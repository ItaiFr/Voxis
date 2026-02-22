"""
Voxis - Voice dictation & AI assistant for Windows
=====================================================

Modes:
  Ctrl+Alt+D  Fast Dictate   → speech to text (local cleanup)
  Ctrl+Alt+F  Polish Dictate → speech to text (AI cleans up grammar)
  Ctrl+Alt+R  AI Command     → speech prompt to Claude/ChatGPT
  Ctrl+Alt+W  Rewrite        → highlight text, speak changes, AI rewrites
  Ctrl+Alt+V  Toggle Window  → show/hide the Voxis window
  Ctrl+Alt+Q  Quit           → exit Voxis
"""

import json
import os
import queue
import re
import sys
import time
import threading
import wave
import tempfile
import logging
import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import keyboard
import pyperclip

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("voxis.log")]
)
log = logging.getLogger("Voxis")

# ─── Configuration ───────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "config.json"
HISTORY_PATH = Path(__file__).parent / "history.json"
MAX_HISTORY = 25

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

CFG = load_config()


def load_history() -> list[dict]:
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, "r") as f:
                entries = json.load(f)
            return entries[-MAX_HISTORY:]
        except Exception:
            return []
    return []


def save_history(entries: list[dict]):
    try:
        with open(HISTORY_PATH, "w") as f:
            json.dump(entries[-MAX_HISTORY:], f, indent=2)
    except Exception as e:
        log.warning(f"Could not save history: {e}")


# ─── Whisper "You" bug filter ────────────────────────────────────────────────
# Whisper hallucinates short phrases like "You", "Thank you", "Thanks for
# watching" on silence or very short audio. We filter these out.

HALLUCINATION_PHRASES = {
    "you", "you.", "thank you", "thank you.", "thanks", "thanks.",
    "thanks for watching", "thanks for watching.", "bye", "bye.",
    "thank you for watching", "thank you for watching.",
    "subscribe", "like and subscribe", "see you next time",
    "please subscribe", "the end", "the end.",
}

def is_hallucination(text: str) -> bool:
    """Check if transcription is a known Whisper silence hallucination."""
    return text.strip().lower().rstrip('.!?,') in HALLUCINATION_PHRASES or \
           len(text.strip()) < CFG.get("min_transcription_length", 2)


# ─── Audio Recorder ──────────────────────────────────────────────────────────

class AudioRecorder:
    SAMPLE_RATE = 16000
    CHANNELS = 1

    def __init__(self):
        self.frames: list[np.ndarray] = []
        self.recording = False
        self._stream = None
        self._start_time = 0

    def start(self):
        self.frames = []
        self.recording = True
        self._start_time = time.time()
        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        log.info("Recording started")

    def _callback(self, indata, frame_count, time_info, status):
        if status:
            log.warning(f"Audio status: {status}")
        if self.recording:
            self.frames.append(indata.copy())

    @property
    def duration(self) -> float:
        return time.time() - self._start_time if self._start_time else 0

    def stop(self) -> str:
        """Stop recording and return path to a temp WAV file, or '' if too short."""
        self.recording = False
        duration = self.duration
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        min_dur = CFG.get("min_audio_duration_sec", 0.5)
        if not self.frames or duration < min_dur:
            log.info(f"Recording too short ({duration:.1f}s < {min_dur}s), discarding")
            return ""

        audio = np.concatenate(self.frames, axis=0)

        # Check if audio is mostly silence (RMS below threshold)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.005:
            log.info(f"Recording is silence (RMS={rms:.4f}), discarding")
            return ""

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        log.info(f"Recording saved ({duration:.1f}s): {tmp.name}")
        return tmp.name


# ─── Local Text Cleanup ─────────────────────────────────────────────────────

def basic_cleanup(text: str) -> str:
    if not text:
        return text
    fillers = [
        r'\b(uh|uhh|uhm|um|umm|hmm|hm|ah|ahh|er|erm)\b',
        r'\b(like,?\s+you know)\b',
        r'\b(you know what I mean|I mean like|sort of like)\b',
    ]
    cleaned = text
    for pattern in fillers:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    cleaned = re.sub(
        r'([.!?])\s+([a-z])',
        lambda m: m.group(1) + ' ' + m.group(2).upper(),
        cleaned
    )
    if cleaned and cleaned[-1] not in '.!?':
        cleaned += '.'
    return cleaned


# ─── Whisper Transcriber ─────────────────────────────────────────────────────

class Transcriber:
    def __init__(self, model_size: str = "base.en"):
        self.model_size = model_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            log.info(f"Loading Whisper model '{self.model_size}'...")
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            log.info("Whisper model loaded.")
        return self._model

    def transcribe(self, audio_path: str) -> str:
        model = self._load_model()
        segments, info = model.transcribe(audio_path, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments)
        log.info(f"Transcribed: {text[:120]}")
        return text.strip()


# ─── Browser AI Client ───────────────────────────────────────────────────────

class BrowserAI:
    def __init__(self, provider: str = "claude"):
        self.provider = provider
        self.selectors = CFG["selectors"][provider]
        self.profile_dir = str(Path(CFG["browser_profile_dir"]).resolve())
        self._playwright = None
        self._browser = None
        self._page = None

    def _ensure_browser(self):
        if self._page and not self._page.is_closed():
            return
        from playwright.sync_api import sync_playwright
        if self._playwright is None:
            self._playwright = sync_playwright().start()
        log.info(f"Launching browser for {self.provider}...")
        self._browser = self._playwright.chromium.launch_persistent_context(
            user_data_dir=self.profile_dir,
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
            viewport={"width": 1280, "height": 900},
        )
        self._page = self._browser.new_page()
        self._page.goto(self.selectors["url"], wait_until="domcontentloaded")
        self._page.wait_for_timeout(3000)
        log.info(f"Browser ready at {self.selectors['url']}")

    def send_prompt(self, prompt: str, timeout_sec: int = 120) -> str:
        self._ensure_browser()
        page = self._page
        sel = self.selectors

        try:
            page.goto(sel["url"], wait_until="domcontentloaded")
            page.wait_for_timeout(2000)
        except Exception as e:
            log.warning(f"Navigation issue: {e}")

        input_el = page.wait_for_selector(sel["input"], timeout=15000)
        if input_el:
            input_el.click()
            page.wait_for_timeout(300)
            if self.provider == "claude":
                input_el.fill("")
                page.keyboard.type(prompt, delay=10)
            else:
                input_el.fill(prompt)
            page.wait_for_timeout(500)

        send_btn = page.wait_for_selector(sel["send_button"], timeout=5000)
        if send_btn:
            send_btn.click()
        else:
            page.keyboard.press("Enter")

        log.info("Prompt sent, waiting for response...")
        page.wait_for_timeout(3000)

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            streaming = page.query_selector(sel.get("stop_streaming", ".__nonexistent__"))
            if streaming is None:
                page.wait_for_timeout(1000)
                break
            page.wait_for_timeout(500)

        responses = page.query_selector_all(sel["response"])
        if responses:
            text = responses[-1].inner_text()
            log.info(f"Got response: {text[:120]}")
            return text.strip()
        else:
            log.warning("No response found on page.")
            return "[No response detected — check the browser window]"

    def close(self):
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()


# ─── Notification Helper ─────────────────────────────────────────────────────

try:
    from plyer import notification as _plyer_notif
except ImportError:
    _plyer_notif = None

def notify(title: str, message: str):
    try:
        if _plyer_notif:
            _plyer_notif.notify(title=title, message=message[:200],
                                app_name="Voxis", timeout=3)
    except Exception:
        pass


# ─── GUI ─────────────────────────────────────────────────────────────────────

import tkinter as tk
from tkinter import ttk

# Colors
BG       = "#1a1a2e"
BG2      = "#16213e"
BG3      = "#0f3460"
FG       = "#e0e0e0"
FG_DIM   = "#8899aa"
ACCENT   = "#e94560"
ACCENT2  = "#00d2ff"
GREEN    = "#00c853"
SURFACE  = "#222244"

class VoxisGUI:
    """Main GUI window with history, status, and controls."""

    def __init__(self, app: "VoxisApp"):
        self.app = app
        self.root = tk.Tk()
        self.root.title("Voxis")
        self.root.geometry("520x640")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.root.minsize(400, 400)

        # Try to set icon
        try:
            self._set_icon()
        except Exception:
            pass

        # Handle close = hide
        self.root.protocol("WM_DELETE_WINDOW", self.hide)

        self._build_ui()
        self.history_items: list[dict] = []
        self._load_saved_history()

    def _set_icon(self):
        """Set a simple app icon."""
        from PIL import Image, ImageDraw, ImageTk
        img = Image.new("RGBA", (32, 32), (26, 26, 46, 255))
        draw = ImageDraw.Draw(img)
        draw.ellipse([10, 2, 22, 16], fill=(233, 69, 96, 255))
        draw.rectangle([13, 16, 19, 22], fill=(233, 69, 96, 255))
        draw.line([16, 24, 16, 30], fill=(233, 69, 96, 255), width=2)
        self._icon_photo = ImageTk.PhotoImage(img)
        self.root.iconphoto(False, self._icon_photo)

    def _build_ui(self):
        # ── Top bar ──
        top = tk.Frame(self.root, bg=BG2, pady=10, padx=15)
        top.pack(fill="x")

        tk.Label(top, text="Voxis", font=("Segoe UI", 18, "bold"),
                 fg=ACCENT, bg=BG2).pack(side="left")

        self.status_label = tk.Label(top, text="Ready", font=("Segoe UI", 11),
                                     fg=GREEN, bg=BG2)
        self.status_label.pack(side="right")

        self.recording_dot = tk.Label(top, text="", font=("Segoe UI", 14),
                                      fg=ACCENT, bg=BG2)
        self.recording_dot.pack(side="right", padx=(0, 8))

        # ── Hotkey reference ──
        ref = tk.Frame(self.root, bg=BG, pady=6, padx=15)
        ref.pack(fill="x")

        hotkeys = [
            (CFG["hotkey_dictate"], "Dictate", FG),
            (CFG["hotkey_polish"], "Polish", FG),
            (CFG["hotkey_ai"], "AI Cmd", FG),
            (CFG["hotkey_rewrite"], "Rewrite", FG),
        ]
        for key, label, color in hotkeys:
            f = tk.Frame(ref, bg=BG)
            f.pack(side="left", padx=(0, 14))
            tk.Label(f, text=key.upper(), font=("Consolas", 9, "bold"),
                     fg=ACCENT2, bg=BG).pack(side="left")
            tk.Label(f, text=f" {label}", font=("Segoe UI", 9),
                     fg=FG_DIM, bg=BG).pack(side="left")

        # ── Separator ──
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=15, pady=(4, 0))

        # ── History header ──
        hist_header = tk.Frame(self.root, bg=BG, padx=15)
        hist_header.pack(fill="x", pady=(8, 4))
        tk.Label(hist_header, text="History", font=("Segoe UI", 12, "bold"),
                 fg=FG, bg=BG).pack(side="left")

        clear_btn = tk.Button(hist_header, text="Clear All", font=("Segoe UI", 9),
                              fg=FG_DIM, bg=BG2, activeforeground=FG,
                              activebackground=BG3, bd=0, padx=10, pady=2,
                              cursor="hand2", command=self.clear_history)
        clear_btn.pack(side="right")

        # ── History scrollable area ──
        container = tk.Frame(self.root, bg=BG)
        container.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        self.canvas = tk.Canvas(container, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)

        self.history_frame = tk.Frame(self.canvas, bg=BG)
        self.history_frame.bind("<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas_window = self.canvas.create_window((0, 0), window=self.history_frame,
                                                        anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>",
            lambda e: self.canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        # ── Empty state ──
        self.empty_label = tk.Label(self.history_frame,
            text="No history yet.\nPress a hotkey to start recording!",
            font=("Segoe UI", 11), fg=FG_DIM, bg=BG, justify="center", pady=40)
        self.empty_label.pack(fill="x")

        # ── Bottom bar ──
        bottom = tk.Frame(self.root, bg=BG2, pady=8, padx=15)
        bottom.pack(fill="x", side="bottom")

        # Provider toggle buttons
        provider_frame = tk.Frame(bottom, bg=BG2)
        provider_frame.pack(side="left")

        tk.Label(provider_frame, text="AI:", font=("Segoe UI", 9, "bold"),
                 fg=FG_DIM, bg=BG2).pack(side="left", padx=(0, 6))

        self.claude_btn = tk.Button(provider_frame, text="Claude",
                                    font=("Segoe UI", 9), bd=0, padx=10, pady=3,
                                    cursor="hand2",
                                    command=lambda: self.app.switch_provider("claude"))
        self.claude_btn.pack(side="left", padx=(0, 2))

        self.chatgpt_btn = tk.Button(provider_frame, text="ChatGPT",
                                     font=("Segoe UI", 9), bd=0, padx=10, pady=3,
                                     cursor="hand2",
                                     command=lambda: self.app.switch_provider("chatgpt"))
        self.chatgpt_btn.pack(side="left")

        self._update_provider_buttons(CFG.get("ai_provider", "claude"))

        login_btn = tk.Button(bottom, text="Open Browser / Login",
                              font=("Segoe UI", 9), fg=FG, bg=BG3,
                              activeforeground=FG, activebackground=ACCENT,
                              bd=0, padx=12, pady=4, cursor="hand2",
                              command=lambda: threading.Thread(
                                  target=self.app.open_browser, daemon=True).start())
        login_btn.pack(side="right")

        hide_btn = tk.Button(bottom, text="Hide Window",
                             font=("Segoe UI", 9), fg=FG, bg=BG3,
                             activeforeground=FG, activebackground=ACCENT,
                             bd=0, padx=12, pady=4, cursor="hand2",
                             command=self.hide)
        hide_btn.pack(side="right", padx=(0, 8))

        quit_btn = tk.Button(bottom, text="Quit",
                             font=("Segoe UI", 9), fg=FG, bg=ACCENT,
                             activeforeground=FG, activebackground="#c0392b",
                             bd=0, padx=12, pady=4, cursor="hand2",
                             command=self.quit)
        quit_btn.pack(side="right", padx=(0, 8))

    def _update_provider_buttons(self, provider: str):
        """Highlight the active provider button."""
        if provider == "claude":
            self.claude_btn.config(fg=FG, bg=ACCENT, activeforeground=FG, activebackground=ACCENT)
            self.chatgpt_btn.config(fg=FG_DIM, bg=BG3, activeforeground=FG, activebackground=BG3)
        else:
            self.claude_btn.config(fg=FG_DIM, bg=BG3, activeforeground=FG, activebackground=BG3)
            self.chatgpt_btn.config(fg=FG, bg=GREEN, activeforeground=FG, activebackground=GREEN)

    def _on_canvas_resize(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    # ── History management ──

    def _load_saved_history(self):
        entries = load_history()
        for entry in entries:
            self._add_history_entry(entry["mode"], entry["text"],
                                    timestamp=entry.get("timestamp"),
                                    persist=False)

    def add_history(self, mode: str, text: str):
        """Add a new entry to the history (called from any thread)."""
        self.root.after(0, self._add_history_entry, mode, text)

    def _add_history_entry(self, mode: str, text: str, timestamp: str = None,
                           persist: bool = True):
        # Remove empty state label
        if self.empty_label:
            self.empty_label.destroy()
            self.empty_label = None

        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        mode_colors = {
            "dictate": ("#4fc3f7", "Dictate"),
            "polish":  ("#ab47bc", "Polish"),
            "ai":      ("#e94560", "AI Cmd"),
            "rewrite": ("#ffb74d", "Rewrite"),
        }
        color, label = mode_colors.get(mode, (FG_DIM, mode))

        # Card frame
        card = tk.Frame(self.history_frame, bg=SURFACE, padx=10, pady=8)
        card.pack(fill="x", pady=(0, 6))

        # Header row
        header = tk.Frame(card, bg=SURFACE)
        header.pack(fill="x")

        tk.Label(header, text=label, font=("Segoe UI", 9, "bold"),
                 fg=color, bg=SURFACE).pack(side="left")
        tk.Label(header, text=f"  {timestamp}", font=("Consolas", 8),
                 fg=FG_DIM, bg=SURFACE).pack(side="left")

        # Copy button
        text_to_copy = text
        copy_btn = tk.Button(header, text="Copy", font=("Segoe UI", 8),
                             fg=FG, bg=BG3, activeforeground=FG,
                             activebackground=GREEN, bd=0, padx=8, pady=1,
                             cursor="hand2",
                             command=lambda t=text_to_copy, b=None: self._copy_text(t, copy_btn))
        copy_btn.pack(side="right")

        # Text content
        display_text = text if len(text) <= 500 else text[:500] + "..."
        text_label = tk.Label(card, text=display_text, font=("Segoe UI", 10),
                              fg=FG, bg=SURFACE, wraplength=440, justify="left",
                              anchor="w")
        text_label.pack(fill="x", pady=(4, 0))

        # Store reference
        self.history_items.append({"card": card, "text": text, "mode": mode,
                                   "timestamp": timestamp})

        # Persist to file
        if persist:
            entries = [{"mode": h["mode"], "text": h["text"],
                        "timestamp": h["timestamp"]} for h in self.history_items]
            save_history(entries)

        # Scroll to bottom
        self.root.after(50, lambda: self.canvas.yview_moveto(1.0))

    def _copy_text(self, text: str, btn: tk.Button):
        pyperclip.copy(text)
        original_text = btn.cget("text")
        btn.config(text="Copied!", fg=GREEN)
        self.root.after(1500, lambda: btn.config(text=original_text, fg=FG))

    def clear_history(self):
        for item in self.history_items:
            item["card"].destroy()
        self.history_items.clear()
        save_history([])
        self.empty_label = tk.Label(self.history_frame,
            text="No history yet.\nPress a hotkey to start recording!",
            font=("Segoe UI", 11), fg=FG_DIM, bg=BG, justify="center", pady=40)
        self.empty_label.pack(fill="x")

    # ── Status updates ──

    def set_status(self, text: str, color: str = FG):
        self.root.after(0, lambda: self._update_status(text, color))

    def _update_status(self, text: str, color: str):
        self.status_label.config(text=text, fg=color)

    def set_recording(self, is_recording: bool, mode: str = ""):
        self.root.after(0, lambda: self._update_recording(is_recording, mode))

    def _update_recording(self, is_recording: bool, mode: str):
        if is_recording:
            self.recording_dot.config(text="⏺")
            self.status_label.config(text=f"Recording ({mode})...", fg=ACCENT)
            self._blink_dot()
        else:
            self.recording_dot.config(text="")

    def _blink_dot(self):
        if self.recording_dot.cget("text") == "⏺":
            current = self.recording_dot.cget("fg")
            new_color = BG2 if current == ACCENT else ACCENT
            self.recording_dot.config(fg=new_color)
            self.root.after(500, self._blink_dot)

    # ── Window visibility ──

    def show(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def hide(self):
        self.root.withdraw()

    def toggle(self):
        if self.root.state() == "withdrawn" or not self.root.winfo_viewable():
            self.show()
        else:
            self.hide()

    def quit(self):
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ─── Main Application ────────────────────────────────────────────────────────

class VoxisApp:
    MODE_LABELS = {
        "dictate": "Dictate",
        "polish": "Polish",
        "ai": "AI Cmd",
        "rewrite": "Rewrite",
    }

    def __init__(self):
        self.recorder = AudioRecorder()
        self.transcriber = Transcriber(CFG.get("whisper_model", "base.en"))
        self.browser_ai = BrowserAI(CFG.get("ai_provider", "claude"))
        self.command_prefix = CFG.get("command_prefix", "hey claude").lower()
        self.polish_prompt = CFG.get("polish_prompt", "")
        self.rewrite_prompt_template = CFG.get("rewrite_prompt", "")
        self.is_recording = False
        self.current_mode = None
        self._lock = threading.Lock()
        self._rewrite_original = ""  # Text captured for rewrite mode

        # Dedicated thread for all Playwright/browser operations
        self._browser_queue = queue.Queue()
        self._browser_thread = threading.Thread(target=self._browser_worker, daemon=True)
        self._browser_thread.start()

        # GUI created later in run()
        self.gui: VoxisGUI = None

    def _browser_worker(self):
        """Single thread that handles all browser operations."""
        while True:
            func, args, result_q = self._browser_queue.get()
            try:
                result = func(*args)
                result_q.put(("ok", result))
            except Exception as e:
                result_q.put(("error", e))

    def _run_in_browser_thread(self, func, *args):
        """Run a function on the dedicated browser thread and return the result."""
        result_q = queue.Queue()
        self._browser_queue.put((func, args, result_q))
        status, value = result_q.get()
        if status == "error":
            raise value
        return value

    def toggle_dictate(self):
        self._toggle("dictate")

    def toggle_polish(self):
        self._toggle("polish")

    def toggle_ai(self):
        self._toggle("ai")

    def toggle_rewrite(self):
        """
        Rewrite mode:
        1st press: captures the currently highlighted/selected text, starts recording voice instructions
        2nd press: stops recording, sends highlighted text + instructions to AI
        """
        with self._lock:
            if self.is_recording and self.current_mode == "rewrite":
                self._stop_and_process()
            elif not self.is_recording:
                # Step 1: Capture highlighted text via clipboard
                old_clipboard = pyperclip.paste()
                keyboard.send("ctrl+c")
                time.sleep(0.15)
                selected = pyperclip.paste()

                if selected == old_clipboard or not selected.strip():
                    self._set_status("⚠️ No text selected — highlight text first", ACCENT)
                    notify("Voxis", "⚠️ Highlight some text first, then press Ctrl+Alt+W")
                    return

                self._rewrite_original = selected.strip()
                self.current_mode = "rewrite"
                self.is_recording = True
                self.recorder.start()
                if self.gui:
                    self.gui.set_recording(True, "Rewrite")
                self._set_status("🎙️ Recording rewrite instructions...", ACCENT)
                notify("Voxis", "🎙️ Speak your rewrite instructions... Press Ctrl+Alt+W to stop.")

    def toggle_window(self):
        if self.gui:
            self.gui.root.after(0, self.gui.toggle)

    def quit_app(self):
        log.info("Quit hotkey pressed.")
        if self.gui:
            self.gui.root.after(0, self.gui.quit)

    def _toggle(self, mode: str):
        with self._lock:
            if self.is_recording:
                self._stop_and_process()
            else:
                self.current_mode = mode
                self.is_recording = True
                self.recorder.start()
                label = self.MODE_LABELS.get(mode, mode)
                if self.gui:
                    self.gui.set_recording(True, label)
                self._set_status(f"🎙️ Recording ({label})...", ACCENT)
                notify("Voxis", f"🎙️ Recording ({label})... Press hotkey to stop.")

    def _stop_and_process(self):
        audio_path = self.recorder.stop()
        self.is_recording = False
        if self.gui:
            self.gui.set_recording(False)

        if not audio_path:
            self._set_status("⚠️ No audio captured (too short or silent)", "#ff9800")
            notify("Voxis", "⚠️ No audio captured.")
            return

        mode = self.current_mode
        rewrite_original = self._rewrite_original if mode == "rewrite" else ""
        self._rewrite_original = ""
        threading.Thread(target=self._process, args=(audio_path, mode, rewrite_original),
                         daemon=True).start()

    def _process(self, audio_path: str, mode: str, rewrite_original: str = ""):
        try:
            self._set_status("⏳ Transcribing...", ACCENT2)

            text = self.transcriber.transcribe(audio_path)

            # Clean up temp file
            try:
                os.unlink(audio_path)
            except OSError:
                pass

            # Bug fix: filter hallucinated transcriptions
            if not text or is_hallucination(text):
                self._set_status("⚠️ Nothing detected — try speaking louder", "#ff9800")
                notify("Voxis", "⚠️ Could not detect speech. Try again.")
                log.info(f"Filtered hallucination: '{text}'")
                return

            if mode == "dictate":
                cleaned = basic_cleanup(text)
                pyperclip.copy(cleaned)
                self._set_status("✅ Copied to clipboard", GREEN)
                notify("Voxis", f"📋 Copied:\n{cleaned[:100]}")
                if self.gui:
                    self.gui.add_history("dictate", cleaned)

            elif mode == "polish":
                self._set_status("✨ AI is polishing text...", ACCENT2)
                prompt = self.polish_prompt + text
                response = self._run_in_browser_thread(self.browser_ai.send_prompt, prompt)
                pyperclip.copy(response)
                self._set_status("✅ Polished text copied", GREEN)
                notify("Voxis", f"📋 Polished:\n{response[:100]}")
                if self.gui:
                    self.gui.add_history("polish", response)

            elif mode == "rewrite":
                self._set_status("✏️ AI is rewriting...", ACCENT2)
                prompt = self.rewrite_prompt_template.format(
                    original=rewrite_original, instructions=text
                )
                response = self._run_in_browser_thread(self.browser_ai.send_prompt, prompt)
                pyperclip.copy(response)
                self._set_status("✅ Rewritten text copied", GREEN)
                notify("Voxis", f"📋 Rewritten:\n{response[:100]}")
                if self.gui:
                    self.gui.add_history("rewrite", response)

            elif mode == "ai":
                prompt = text
                if prompt.lower().startswith(self.command_prefix):
                    prompt = prompt[len(self.command_prefix):].strip()
                if not prompt:
                    self._set_status("⚠️ Empty prompt", "#ff9800")
                    return

                self._set_status("🤖 Waiting for AI response...", ACCENT2)
                response = self._run_in_browser_thread(self.browser_ai.send_prompt, prompt)
                pyperclip.copy(response)
                self._set_status("✅ AI response copied", GREEN)
                notify("Voxis", f"📋 AI:\n{response[:100]}")
                if self.gui:
                    self.gui.add_history("ai", response)

        except Exception as e:
            log.error(f"Processing error: {e}", exc_info=True)
            self._set_status(f"❌ Error: {str(e)[:60]}", ACCENT)
            notify("Voxis", f"❌ Error: {str(e)[:100]}")

    def _set_status(self, text: str, color: str = FG):
        log.info(text)
        if self.gui:
            self.gui.set_status(text, color)

    def switch_provider(self, provider: str):
        if provider == self.browser_ai.provider:
            return
        log.info(f"Switching AI provider to {provider}")
        self._run_in_browser_thread(self.browser_ai.close)
        self.browser_ai = BrowserAI(provider)
        CFG["ai_provider"] = provider
        try:
            with open(CONFIG_PATH, "w") as f:
                json.dump(CFG, f, indent=4)
        except Exception as e:
            log.warning(f"Could not save config: {e}")
        if self.gui:
            self.gui.root.after(0, self.gui._update_provider_buttons, provider)
        self._set_status(f"Switched to {provider.title()} — click 'Open Browser / Login'", GREEN)

    def open_browser(self):
        try:
            self._run_in_browser_thread(self.browser_ai._ensure_browser)
            self._set_status("🌐 Browser ready", GREEN)
            notify("Voxis", "🌐 Browser opened. Log in if needed.")
        except Exception as e:
            self._set_status(f"❌ Browser error", ACCENT)
            notify("Voxis", f"❌ Browser error: {str(e)[:100]}")

    def run(self):
        log.info("=" * 60)
        log.info("Voxis Starting")
        log.info("=" * 60)

        # Register global hotkeys
        keyboard.add_hotkey(CFG["hotkey_dictate"], self.toggle_dictate, suppress=True)
        keyboard.add_hotkey(CFG["hotkey_polish"], self.toggle_polish, suppress=True)
        keyboard.add_hotkey(CFG["hotkey_ai"], self.toggle_ai, suppress=True)
        keyboard.add_hotkey(CFG["hotkey_rewrite"], self.toggle_rewrite, suppress=True)
        keyboard.add_hotkey(CFG["hotkey_toggle_window"], self.toggle_window, suppress=True)
        keyboard.add_hotkey(CFG.get("hotkey_quit", "ctrl+alt+q"), self.quit_app, suppress=True)

        # Pre-load whisper in background
        threading.Thread(target=self.transcriber._load_model, daemon=True).start()

        # Create and run GUI
        self.gui = VoxisGUI(self)
        self.gui.set_status("Ready", GREEN)
        notify("Voxis", "✅ Voxis is running! Use hotkeys to start.")

        log.info(f"  Dictate : {CFG['hotkey_dictate']}")
        log.info(f"  Polish  : {CFG['hotkey_polish']}")
        log.info(f"  AI Cmd  : {CFG['hotkey_ai']}")
        log.info(f"  Rewrite : {CFG['hotkey_rewrite']}")
        log.info(f"  Window  : {CFG['hotkey_toggle_window']}")
        log.info(f"  Quit    : {CFG.get('hotkey_quit', 'ctrl+alt+q')}")

        self.gui.run()  # Blocks until window closed

        # Cleanup
        keyboard.unhook_all()
        try:
            self._run_in_browser_thread(self.browser_ai.close)
        except Exception:
            pass
        log.info("Voxis exited.")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = VoxisApp()
    app.run()

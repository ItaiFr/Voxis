# Voxis

Voice-to-text dictation and AI assistant for Windows.  
Uses your existing Claude or ChatGPT subscription — no API key needed.

---

## What It Does

Press a hotkey anywhere on your computer, speak, press again to stop.  
The result is copied to your clipboard, ready to paste.

| Hotkey | Mode | Description |
|--------|------|-------------|
| `Ctrl+Alt+D` | **Dictate** | Speech → text (fast, runs locally) |
| `Ctrl+Alt+F` | **Polish** | Speech → AI-cleaned text (fixes grammar, removes filler words) |
| `Ctrl+Alt+R` | **AI Command** | Speech → sends to Claude/ChatGPT → copies AI response |
| `Ctrl+Alt+W` | **Rewrite** | Highlight text → speak changes → AI rewrites it |
| `Ctrl+Alt+V` | **Show/Hide** | Toggle the Voxis window |
| `Ctrl+Alt+Q` | **Quit** | Exit Voxis completely |

---

## Install (one time)

**Requirements:** Windows 10/11, Python 3.10+, a microphone.

1. Download this folder
2. Double-click **`setup.bat`**
3. Wait for it to finish (downloads ~75 MB for the speech model on first run)

---

## Usage

1. Double-click **`run.bat`** — the Voxis window opens
2. Click **"Open Browser / Login"** and log in to Claude.ai or ChatGPT
3. Minimize the browser (don't close it)
4. You're ready! Use the hotkeys from any app:

**Dictate:** `Ctrl+Alt+D` → speak → `Ctrl+Alt+D` → `Ctrl+V` to paste

**Polish:** `Ctrl+Alt+F` → speak → `Ctrl+Alt+F` → wait for AI → `Ctrl+V`

**AI Command:** `Ctrl+Alt+R` → speak your question → `Ctrl+Alt+R` → wait → `Ctrl+V`

**Rewrite:** Select text → `Ctrl+Alt+W` → speak what to change → `Ctrl+Alt+W` → wait → `Ctrl+V`

All results appear in the Voxis window history with copy buttons.

---

## The Rewrite Feature

This is like having an AI editor on standby:

1. **Highlight** any text in any app (email, document, browser, etc.)
2. Press **`Ctrl+Alt+W`** — Voxis captures the highlighted text
3. **Speak** your instructions (e.g. "make this more formal" or "shorten to two sentences")
4. Press **`Ctrl+Alt+W`** again to stop recording
5. The AI rewrites the text → it's on your clipboard → **`Ctrl+V`** to paste

---

## Settings

Edit **`config.json`** to customize:

- **Hotkeys** — change any key combination
- **AI provider** — set `"ai_provider"` to `"claude"` or `"chatgpt"`
- **Whisper model** — `"tiny.en"` (fastest), `"base.en"` (default), `"small.en"` (most accurate)
- **Polish prompt** — customize how the AI cleans up your text
- **Rewrite prompt** — customize the rewrite instructions template

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No audio captured | Check microphone is set as default in Windows Sound settings |
| "You" or gibberish output | Fixed in this version — very short/silent recordings are filtered |
| Browser won't open | Run: `venv\Scripts\activate && python -m playwright install chromium` |
| AI not responding | Open browser window — you may need to log in again |
| Hotkeys not working | Another app may be using the same shortcut — change in config.json |

---

## Files

```
Voxis/
├── voxis.py            Main app
├── config.json         Settings
├── requirements.txt    Dependencies
├── setup.bat           One-time setup
├── run.bat             Daily launcher
├── diagnose.bat        Troubleshooting helper
├── browser_profile/    Saves your login (auto-created)
└── voxis.log           Debug log
```

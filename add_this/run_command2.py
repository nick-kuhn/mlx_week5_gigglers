import sounddevice as sd
import numpy as np
import torch
import whisper
from pynput import keyboard
import time
import os
import webbrowser
import pyautogui

# Load your fine-tuned model
model = whisper.load_model("tiny")  # Use the *same* architecture as you trained
model.load_state_dict(torch.load("whisper_ask_ft.pth", map_location="cpu"))
model.eval()

fs = 16000  # Sample rate
channels = 1
audio = []
recording = False
stream = None
audio_lock = None  # Initialized when script starts

def audio_callback(indata, frames, t, status):
    global audio, recording
    if recording:
        with audio_lock:
            audio.append(indata.copy())

def start_recording():
    global audio, recording, stream, audio_lock
    audio = []
    recording = True
    stream = sd.InputStream(samplerate=fs, channels=channels, callback=audio_callback)
    stream.start()
    print("Recording... (hold SPACE)")

def stop_recording():
    global recording, stream, audio
    recording = False
    time.sleep(0.2)  # Let buffer clear
    stream.stop()
    stream.close()
    with audio_lock:
        if len(audio) == 0:
            print("No audio captured!")
            return None
        audio_np = np.concatenate(audio, axis=0)
    print(f"Captured {len(audio_np)} audio samples")
    return audio_np

def transcribe_audio_np(audio_np, fs):
    # If 2D, make 1D (whisper expects shape (samples,))
    if audio_np.ndim > 1:
        audio_np = audio_np.flatten()
    audio_np = audio_np.astype(np.float32)
    # Whisper's pad_or_trim expects 16,000 Hz
    audio = whisper.pad_or_trim(audio_np)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(language="en", without_timestamps=True)
    result = whisper.decode(model, mel, options)
    return result.text.lower()

# ----- Command Handlers -----
def open_spotify(song=None):
    print("Opening Spotify...")
    os.system("open -a Spotify")
    if song:
        time.sleep(2)
        os.system(f"open 'spotify:search:{song.replace(' ', '%20')}'")
        print(f"Searched for song: {song}")

def open_notes(note=None):
    if note:
        add_quick_note_mac(note)
    else:
        print("Opening Notes...")
        os.system("open -a Notes")

def add_quick_note_mac(note):
    import subprocess
    # AppleScript: create and focus the new note
    applescript = f'''
    tell application "Notes"
        activate
        set theNote to make new note at folder "Notes" of account "iCloud" with properties {{name:"Quick Note", body:"{note}"}}
        show theNote
    end tell
    '''
    print("Creating and focusing a quick note in the Notes app...")
    subprocess.run(['osascript', '-e', applescript])

def open_calendar():
    print("Opening Calendar...")
    os.system("open -a Calendar")

def google_search(query):
    print("Googling:", query)
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    webbrowser.open(url)

def open_maps(destination):
    print("Opening Maps to:", destination)
    url = f"https://www.google.com/maps/search/{destination.replace(' ', '+')}"
    webbrowser.open(url)

def ask_chatgpt_auto(prompt):
    url = "https://chat.openai.com/"
    print("Opening ChatGPT in your default browser...")
    webbrowser.open(url)
    time.sleep(2)
    print("Click inside the ChatGPT prompt box in your browser!")
    for i in range(2, 0, -1):
        print(f"Typing in {i}...", end="\r")
        time.sleep(1)
    pyautogui.typewrite(prompt)
    pyautogui.press('enter')
    print("Prompt sent to ChatGPT!")

def take_screenshot():
    print("Taking screenshot...")
    os.system("screencapture -x ~/Desktop/voice_screenshot.png")
    print("Screenshot saved to Desktop.")

def open_gmail_email(prompt=None):
    url = "https://mail.google.com/"
    print("Opening Gmail in your default browser...")
    webbrowser.open(url)
    if prompt:
        print("Wait a few seconds for Gmail to load, then click in the compose box!")
        for i in range(4, 0, -1):
            print(f"Typing in {i}...", end="\r")
            time.sleep(1)
        pyautogui.typewrite(prompt)
        print("\nPrompt typed as email body.")

def mute_audio():
    print("Muting audio...")
    os.system("""osascript -e 'set volume output muted true'""")

def unmute_audio():
    print("Unmuting audio...")
    os.system("""osascript -e 'set volume output muted false'""")

# ----- Main Command Dispatcher -----
def handle_command(command):
    print("Recognized:", command)
    command = command.strip()

    if command.startswith("open spotify"):
        song = command.replace("open spotify", "").replace("and play", "").strip()
        open_spotify(song)
    elif command.startswith("open notes and write down"):
        note = command.replace("open notes and write down", "").strip()
        open_notes(note)
    elif command.startswith("open notes"):
        open_notes()  # Just open Notes, no text writing
    elif command.startswith("take a note"):
        note = command.replace("take a note", "").strip()
        open_notes(note)
    elif command.startswith("open calendar"):
        open_calendar()
    elif command.startswith("open google and search"):
        query = command.replace("open google and search", "").strip()
        google_search(query)
    elif command.startswith("google"):
        query = command.replace("google", "").strip()
        google_search(query)
    elif command.startswith("search"):
        query = command.replace("search", "").strip()
        google_search(query)
    elif command.startswith("open maps and search"):
        destination = command.replace("open maps and search", "").strip()
        open_maps(destination)
    elif command.startswith("open maps") or command.startswith("maps"):
        destination = command.replace("open maps", "").replace("maps", "").strip()
        open_maps(destination)
    elif command.startswith("ask chatgpt"):
        prompt = command.replace("ask chatgpt", "").strip()
        ask_chatgpt_auto(prompt)
    elif "screenshot" in command:
        take_screenshot()
    elif command.startswith("open gmail and write an email saying"):
        prompt = command.replace("open gmail and write an email saying", "").strip()
        open_gmail_email(prompt)
    elif "unmute" in command:
        unmute_audio()
    elif "mute" in command:
        mute_audio()
    elif command.startswith("lock"):
        lock_computer()
    else:
        print("No recognized command.")

# ----- Keyboard Listener -----
def on_press(key):
    if key == keyboard.Key.space and not getattr(on_press, "is_recording", False):
        on_press.is_recording = True
        start_recording()

def on_release(key):
    if key == keyboard.Key.space and getattr(on_press, "is_recording", False):
        on_press.is_recording = False
        audio_np = stop_recording()
        if audio_np is not None:
            command = transcribe_audio_np(audio_np, fs)
            handle_command(command)
    if key == keyboard.Key.esc:
        # Stop listener
        return False

if __name__ == "__main__":
    import threading
    audio_lock = threading.Lock()  # Needed for thread-safe audio list
    print("Hold SPACE to record. Release to stop and transcribe. Press ESC to quit.")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

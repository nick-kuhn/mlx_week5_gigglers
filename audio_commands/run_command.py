import sounddevice as sd
import soundfile as sf
import torch
import whisper
from pynput import keyboard
import time
import os
import webbrowser
import pyautogui
import subprocess

# Load your fine-tuned model
model = whisper.load_model("tiny")  # Use the *same* architecture as you trained
model.load_state_dict(torch.load("whisper_askgpt_ft.pth", map_location="cpu"))
model.eval()
recording = False

def record(filename="latestcommand.wav", duration=5, samplerate=16000):
    print("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"Saved to {filename}")
    return filename

def transcribe(path):
    audio = whisper.load_audio(path)
    audio = whisper.pad_or_trim(audio)
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
    print("Opening Notes...")
    os.system("open -a Notes")
    if note:
        print(f"Writing note: {note}")
        # To automate note writing, use AppleScript (macOS only), or pyautogui

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
    # macOS screenshot command
    os.system("screencapture -x ~/Desktop/voice_screenshot.png")
    print("Screenshot saved to Desktop.")

def open_email():
    print("Opening Mail app...")
    os.system("open -a Mail")
    # To automate writing an email, you'd need more scripting or pyautogui

def mute_unmute():
    print("Toggling mute/unmute...")
    # macOS mute using AppleScript
    os.system("""osascript -e 'set volume output muted not (output muted of (get volume settings))'""")

def lock_computer():
    print("Locking computer...")
    os.system('/System/Library/CoreServices/Menu\\ Extras/User.menu/Contents/Resources/CGSession -suspend')

# ----- Main Command Dispatcher -----

def handle_command(command):
    print("Recognized:", command)
    command = command.strip()

    # 1. Spotify
    if command.startswith("open spotify"):
        song = command.replace("open spotify", "").replace("and play", "").strip()
        open_spotify(song)
    # 2. Notes
    elif command.startswith("open notes") or command.startswith("take a note"):
        note = command.replace("open notes", "").replace("take a note", "").strip()
        open_notes(note)
    # 3. Calendar
    elif command.startswith("open calendar"):
        open_calendar()
    # 4. Google search
    elif command.startswith("google") or command.startswith("search"):
        # Accepts "google cats" or "search cats"
        query = command.replace("google", "").replace("search", "").strip()
        google_search(query)
    # 5. Maps
    elif command.startswith("open maps") or command.startswith("maps"):
        destination = command.replace("open maps", "").replace("maps", "").strip()
        open_maps(destination)
    # 6. ChatGPT
    elif command.startswith("ask chatgpt"):
        prompt = command.replace("ask chatgpt", "").strip()
        ask_chatgpt_auto(prompt)
    # 7. Screenshot
    elif "screenshot" in command:
        take_screenshot()
    # 8. Email
    elif command.startswith("open email") or command.startswith("write an email") or command.startswith("open mail"):
        open_email()
    # 9. Mute/unmute
    elif "mute" in command:
        mute_unmute()
    # 10. Lock computer
    elif command.startswith("lock"):
        lock_computer()
    else:
        print("No recognized command.")

# ----- Keyboard Listener -----

def on_press(key):
    global recording
    if key == keyboard.Key.ctrl and not recording:
        recording = True
        filename = record(duration=5)
        command = transcribe(filename)
        handle_command(command)
        recording = False

if __name__ == "__main__":
    print("Hold down CTRL to issue a command.")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
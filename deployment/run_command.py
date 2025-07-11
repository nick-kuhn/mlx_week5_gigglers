import numpy as np
import torch
import sys
from pathlib import Path
from pynput import keyboard
import time
import os
import webbrowser
import pyautogui
import subprocess
import tempfile
import signal
import threading
import whisper
#import openai_whisper as whisper


# Add project root to path for model loading
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import your custom model
try:
    from model.model import WhisperEncoderClassifier
    USE_CUSTOM_MODEL = True
except ImportError:
    print("⚠️  Custom model not found, using base Whisper")
    USE_CUSTOM_MODEL = False

# Detect available audio recording methods
def detect_audio_backend():
    """Detect which audio recording backend is available"""
    try:
        import sounddevice as sd
        # Try to query devices to see if PortAudio is working
        sd.query_devices()
        print("✅ PortAudio (sounddevice) available")
        return "portaudio"
    except (ImportError, Exception) as e:
        print(f"❌ PortAudio not available: {e}")
    
    try:
        # Check if ffmpeg is available
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ ffmpeg available")
            return "ffmpeg"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ ffmpeg not available")
    
    raise RuntimeError("No audio recording backend available. Please install either PortAudio (sounddevice) or ffmpeg.")

# Initialize audio backend
AUDIO_BACKEND = detect_audio_backend()
print(f"🎤 Using audio backend: {AUDIO_BACKEND}")

# Load model (either custom trained model or base Whisper)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if USE_CUSTOM_MODEL:
    try:
        # Try to load your custom trained model
        from transformers import WhisperProcessor
        
        # Paths for custom model
        MODEL_DIR = PROJECT_ROOT / "whisper_models" / "whisper_tiny"
        CHECKPOINT_PATH = PROJECT_ROOT / "model" / "api_net.pt"
        
        if CHECKPOINT_PATH.exists():
            print("🔧 Loading custom trained model...")
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            classes = checkpoint['classes']
            label_to_idx = checkpoint['label_to_idx']
            idx_to_label = {v: k for k, v in label_to_idx.items()}
            
            # Initialize processor and model
            processor = WhisperProcessor.from_pretrained(MODEL_DIR)
            model = WhisperEncoderClassifier(MODEL_DIR, num_classes=len(classes)).to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            print(f"✅ Custom model loaded with {len(classes)} classes")
            USE_CUSTOM_INFERENCE = True
        else:
            print(f"⚠️  Custom model not found at {CHECKPOINT_PATH}, using base Whisper")
            USE_CUSTOM_INFERENCE = False
    except Exception as e:
        print(f"⚠️  Error loading custom model: {e}, using base Whisper")
        USE_CUSTOM_INFERENCE = False
else:
    USE_CUSTOM_INFERENCE = False

if not USE_CUSTOM_INFERENCE:
    # Load base Whisper model
    print("🔧 Loading base Whisper model...")
    model = whisper.load_model("tiny").to(DEVICE)
    print("✅ Base Whisper model loaded")

fs = 16000  # Sample rate
channels = 1

# Audio recording variables (different for each backend)
if AUDIO_BACKEND == "portaudio":
    import sounddevice as sd
    audio = []
    recording = False
    stream = None
    audio_lock = threading.Lock()
else:  # ffmpeg backend
    recording = False
    recording_process = None
    temp_audio_file = None
    audio_lock = threading.Lock()

# PortAudio recording functions
if AUDIO_BACKEND == "portaudio":
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
        print("Recording... (hold CTRL)")

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

# ffmpeg recording functions
else:
    def start_recording():
        global recording, recording_process, temp_audio_file
        with audio_lock:
            recording = True
            # Create temporary file for audio
            temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio_file.close()
            
            # Start ffmpeg recording
            cmd = [
                'ffmpeg', '-f', 'alsa', '-i', 'default',
                '-ar', str(fs), '-ac', str(channels),
                '-y', temp_audio_file.name
            ]
            recording_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL
            )
        print("Recording... (hold CTRL)")

    def stop_recording():
        global recording, recording_process, temp_audio_file
        with audio_lock:
            if not recording or recording_process is None:
                return None
                
            recording = False
            
            # Stop ffmpeg gracefully
            recording_process.send_signal(signal.SIGINT)
            recording_process.wait()
            
            # Read the recorded audio file
            try:
                import librosa
                audio_np, _ = librosa.load(temp_audio_file.name, sr=fs, mono=True)
                
                # Clean up temporary file
                os.unlink(temp_audio_file.name)
                
                if len(audio_np) == 0:
                    print("No audio captured!")
                    return None
                    
                print(f"Captured {len(audio_np)} audio samples")
                return audio_np
                
            except ImportError:
                print("❌ librosa not available for audio loading. Please install: pip install librosa")
                return None
            except Exception as e:
                print(f"❌ Error reading audio file: {e}")
                # Clean up on error
                try:
                    os.unlink(temp_audio_file.name)
                except:
                    pass
                return None

def transcribe_audio_np(audio_np, fs):
    """Transcribe audio using either custom model or base Whisper"""
    # If 2D, make 1D (whisper expects shape (samples,))
    if audio_np.ndim > 1:
        audio_np = audio_np.flatten()
    audio_np = audio_np.astype(np.float32)
    
    if USE_CUSTOM_INFERENCE:
        # Use custom trained model
        try:
            # Preprocess audio with Whisper processor
            inputs = processor(
                audio_np, 
                sampling_rate=fs, 
                return_tensors="pt"
            )
            input_features = inputs["input_features"].to(DEVICE)
            
            # Run inference
            with torch.no_grad():
                logits = model(input_features)
                predicted_idx = logits.argmax(dim=1).item()
                predicted_class = idx_to_label[predicted_idx]
            
            print(f"🎯 Custom model prediction: {predicted_class}")
            return predicted_class.replace('<', '').replace('>', '')  # Remove brackets for command matching
            
        except Exception as e:
            print(f"❌ Error with custom model, falling back to base Whisper: {e}")
            # Fall back to base Whisper
            pass
    
    # Use base Whisper model
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
    """Open notes with pre-written text"""
    # Default text to pre-populate
    draft_text = """(Draft)
To Nick, Api, Yali,

How boring was this weeks project right? Wish it was over already!

Ben"""
    
    if note:
        add_quick_note_mac(note)
    else:
        print("Opening Notes with draft text...")
        
        # Try different approaches based on platform
        if os.system("which open > /dev/null 2>&1") == 0:  # macOS
            add_quick_note_mac(draft_text)
        else:  # Linux/Windows - create temp file and open with text editor
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                    temp_file.write(draft_text)
                    temp_file_path = temp_file.name
                
                # Linux
                if os.system("which gedit > /dev/null 2>&1") == 0:
                    subprocess.Popen(['gedit', temp_file_path])
                elif os.system("which kate > /dev/null 2>&1") == 0:
                    subprocess.Popen(['kate', temp_file_path])
                elif os.system("which nano > /dev/null 2>&1") == 0:
                    subprocess.Popen(['gnome-terminal', '--', 'nano', temp_file_path])
                # Windows
                elif os.name == 'nt':
                    os.system(f"notepad '{temp_file_path}'")
                else:
                    print("No text editor found")
                    os.unlink(temp_file_path)
                    
            except Exception as e:
                print(f"Error opening notes: {e}")

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
    try:
        # Linux with ALSA
        if os.system("which amixer > /dev/null 2>&1") == 0:
            os.system("amixer -D pulse sset Master toggle 2>/dev/null || amixer sset Master toggle")
        # macOS
        elif os.system("which osascript > /dev/null 2>&1") == 0:
            os.system("""osascript -e 'set volume output muted true'""")
        else:
            print("Mute not implemented for this system")
    except Exception as e:
        print(f"Error muting audio: {e}")

def unmute_audio():
    print("Unmuting audio...")
    try:
        # Linux with ALSA
        if os.system("which amixer > /dev/null 2>&1") == 0:
            os.system("amixer -D pulse sset Master on 2>/dev/null || amixer sset Master on")
        # macOS
        elif os.system("which osascript > /dev/null 2>&1") == 0:
            os.system("""osascript -e 'set volume output muted false'""")
        else:
            print("Unmute not implemented for this system")
    except Exception as e:
        print(f"Error unmuting audio: {e}")

def volume_up():
    """Increase system volume"""
    print("Volume up...")
    try:
        # Linux with ALSA
        if os.system("which amixer > /dev/null 2>&1") == 0:
            os.system("amixer -D pulse sset Master 30%+ 2>/dev/null || amixer sset Master 30%+")
        # macOS
        elif os.system("which osascript > /dev/null 2>&1") == 0:
            os.system("""osascript -e 'set volume output volume (output volume of (get volume settings) + 30)'""")
        else:
            print("Volume control not implemented for this system")
    except Exception as e:
        print(f"Error increasing volume: {e}")

def volume_down():
    """Decrease system volume"""
    print("Volume down...")
    try:
        # Linux with ALSA
        if os.system("which amixer > /dev/null 2>&1") == 0:
            os.system("amixer -D pulse sset Master 30%- 2>/dev/null || amixer sset Master 30%-")
        # macOS
        elif os.system("which osascript > /dev/null 2>&1") == 0:
            os.system("""osascript -e 'set volume output volume (output volume of (get volume settings) - 30)'""")
        else:
            print("Volume control not implemented for this system")
    except Exception as e:
        print(f"Error decreasing volume: {e}")

def lock_computer():
    """Lock the computer (works on Linux and macOS)"""
    try:
        # Try Linux first (most common)
        if os.system("which gnome-screensaver-command > /dev/null 2>&1") == 0:
            os.system("gnome-screensaver-command -l")
            print("Computer locked (GNOME)")
        elif os.system("which xdg-screensaver > /dev/null 2>&1") == 0:
            os.system("xdg-screensaver lock")
            print("Computer locked (XDG)")
        elif os.system("which loginctl > /dev/null 2>&1") == 0:
            os.system("loginctl lock-session")
            print("Computer locked (systemd)")
        # macOS fallback
        elif os.system("which pmset > /dev/null 2>&1") == 0:
            os.system("pmset displaysleepnow")
            print("Display locked (macOS)")
        else:
            print("Unable to lock computer - no supported lock command found")
    except Exception as e:
        print(f"Error locking computer: {e}")

# ----- Main Command Dispatcher -----
def handle_command(command):
    print("Recognized:", command)
    command = command.strip()

    # Handle custom model commands (specific tokens)
    if USE_CUSTOM_INFERENCE:
        # Map custom model outputs to actions
        if command in ["close_browser", "close browser"]:
            print("Closing browser...")
            # Add browser closing logic here
            return
        elif command in ["google", "search"]:
            print("Opening Google...")
            google_search("")
            return
        elif command in ["maximize_window", "maximize window"]:
            print("Maximizing window...")
            # Add window maximizing logic here
            return
        elif command in ["mute"]:
            mute_audio()
            return
        elif command in ["no_action"]:
            print("No action taken")
            return
        elif command in ["open_browser", "open browser"]:
            print("Opening browser...")
            webbrowser.open("https://www.google.com")
            return
        elif command in ["open_notepad", "open notepad"]:
            open_notes()
            return
        elif command in ["play_music", "play music"]:
            open_spotify()
            return
        elif command in ["stop_music", "stop music"]:
            print("Stopping music...")
            # Add music stopping logic here
            return
        elif command in ["switch_window", "switch window"]:
            print("Switching window...")
            # Add window switching logic (Alt+Tab equivalent)
            try:
                pyautogui.hotkey('alt', 'tab')
            except:
                print("Could not switch window")
            return
        elif command in ["volume_down", "volume down"]:
            volume_down()
            return
        elif command in ["volume_up", "volume up"]:
            volume_up()
            return

    # Handle natural language commands (base Whisper)
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
    if key == keyboard.Key.ctrl_l and not getattr(on_press, "is_recording", False):
        on_press.is_recording = True
        start_recording()

def on_release(key):
    if key == keyboard.Key.ctrl_l and getattr(on_press, "is_recording", False):
        on_press.is_recording = False
        audio_np = stop_recording()
        if audio_np is not None:
            command = transcribe_audio_np(audio_np, fs)
            handle_command(command)
    if key == keyboard.Key.esc:
        # Stop listener
        return False

if __name__ == "__main__":
    print("Hold CTRL to record. Release to stop and transcribe. Press ESC to quit.")
    print(f"Audio backend: {AUDIO_BACKEND}")
    
    if AUDIO_BACKEND == "ffmpeg":
        print("📋 Note: Using ffmpeg backend. Make sure your microphone is set as default ALSA device.")
        print("📋 You may need to install librosa: pip install librosa")
    
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

#!/usr/bin/env python3
"""
Voice Command Classification Deployment Script
Uses custom trained WhisperEncoderClassifier for direct command classification
"""

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
import glob
import random

CONFIDENCE_THRESHOLD = 0.4

# Add project root to path for model loading
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import custom model
try:
    from model.model import WhisperEncoderClassifier
    from transformers import WhisperProcessor
except ImportError as e:
    print(f"❌ Could not import required modules: {e}")
    print("Please ensure transformers and your custom model are available")
    sys.exit(1)

# Original command tokens from training
COMMAND_TOKENS = [
    '<close_browser>', '<google>', '<maximize_window>', '<mute>', '<no_action>',
    '<open_browser>', '<open_notepad>', '<play_music>', '<stop_music>',
    '<switch_window>', '<volume_down>', '<volume_up>'
]

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

# Load custom classification model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths for custom model
MODEL_DIR = PROJECT_ROOT / "whisper_models" / "whisper_tiny"
CHECKPOINT_PATH = PROJECT_ROOT / "deployment" / "best_model.pt"


def load_classification_model():
    """Load the custom trained classification model"""
    if not CHECKPOINT_PATH.exists():
        print(f"❌ Model checkpoint not found at: {CHECKPOINT_PATH}")
        print("Please ensure best_model.pt is in the deployment folder")
        sys.exit(1)
    
    print("🔧 Loading custom classification model...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    classes = checkpoint['classes']
    label_to_idx = checkpoint['label_to_idx']
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    
    # Initialize processor and model
    processor = WhisperProcessor.from_pretrained(MODEL_DIR)
    model = WhisperEncoderClassifier(MODEL_DIR, num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Classification model loaded with {len(classes)} classes")
    print(f"Classes: {classes}")
    
    return model, processor, classes, label_to_idx, idx_to_label

# Load the model
model, processor, classes, label_to_idx, idx_to_label = load_classification_model()

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
        print("🎤 Recording... (hold SPACE)")

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
        print("🎤 Recording... (hold SPACE)")

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

def classify_audio(audio_np):
    """Classify audio using the custom trained model"""
    try:
        # If 2D, make 1D
        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()
        audio_np = audio_np.astype(np.float32)
        
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
            
            # Get confidence
            probs = torch.softmax(logits, dim=1)
            confidence = probs[0, predicted_idx].item()
        
        print(f"🎯 Classified: {predicted_class} (confidence: {confidence:.3f})")
        return predicted_class, confidence
        
    except Exception as e:
        print(f"❌ Error during classification: {e}")
        return None, 0.0

# ----- Command Handlers for Original Token Set -----

def close_browser():
    """Open a special surprise link"""
    print("� Opening special link...")
    try:
        webbrowser.open("https://shattereddisk.github.io/rickroll/rickroll.mp4")
        print("✅ Never gonna give you up! 🎵")
    except Exception as e:
        print(f"Error opening link: {e}")

def open_google():
    """Open Google in default browser"""
    print("🔍 Opening Google...")
    webbrowser.open("https://www.google.com")

def maximize_window():
    """Maximize the current window"""
    print("🔲 Maximizing window...")
    try:
        # Windows/Linux: Windows+Up or F11
        if os.name in ['posix', 'nt']:
            pyautogui.hotkey('win', 'up')
        # macOS: Different approach needed
        else:
            print("Maximize not implemented for macOS")
    except Exception as e:
        print(f"Error maximizing window: {e}")

def mute_audio():
    """Mute system audio"""
    print("🔇 Muting audio...")
    try:
        # Linux
        if os.system("which pactl > /dev/null 2>&1") == 0:
            os.system("pactl set-sink-mute @DEFAULT_SINK@ toggle")
        # macOS
        elif os.system("which osascript > /dev/null 2>&1") == 0:
            os.system("""osascript -e 'set volume output muted true'""")
        # Windows (requires nircmd or similar)
        else:
            print("Mute not implemented for this system")
    except Exception as e:
        print(f"Error muting audio: {e}")

def no_action():
    """Do nothing - for testing or when no action is intended"""
    print("⚪ No action taken")

def open_browser():
    """Open LinkedIn profile"""
    print("🌐 Opening LinkedIn profile...")
    webbrowser.open("https://www.linkedin.com/in/besart-shyti-20616956/overlay/photo/")

def open_notepad():
    """Open notepad/text editor"""
    print("📝 Opening notepad...")
    try:
        # Linux
        if os.system("which gedit > /dev/null 2>&1") == 0:
            subprocess.Popen(['gedit'])
        elif os.system("which kate > /dev/null 2>&1") == 0:
            subprocess.Popen(['kate'])
        elif os.system("which nano > /dev/null 2>&1") == 0:
            subprocess.Popen(['gnome-terminal', '--', 'nano'])
        # macOS
        elif os.system("which open > /dev/null 2>&1") == 0:
            os.system("open -a TextEdit")
        # Windows
        elif os.name == 'nt':
            os.system("notepad")
        else:
            print("No text editor found")
    except Exception as e:
        print(f"Error opening notepad: {e}")

def play_music():
    """Start music player and begin playback"""
    print("🎵 Playing music...")
    try:
        # First try playerctl for direct playback control
        if os.system("which playerctl > /dev/null 2>&1") == 0:
            # Try to play if any player is available
            result = os.system("playerctl play 2>/dev/null")
            if result == 0:
                print("✅ Started playback using existing player")
                return
        
        # Find a music file to play
        music_file = find_music_file()
        
        # Try common Linux music players with autoplay
        if os.system("which spotify > /dev/null 2>&1") == 0:
            print("Opening Spotify...")
            subprocess.Popen(['spotify'])
            # Wait a moment for Spotify to load, then try to start playback
            time.sleep(4)
            if os.system("which playerctl > /dev/null 2>&1") == 0:
                # Try multiple times as Spotify can be slow to register
                for i in range(3):
                    result = os.system("playerctl -p spotify play 2>/dev/null")
                    if result == 0:
                        print("✅ Started Spotify playback")
                        return
                    time.sleep(1)
            # Fallback: use keyboard shortcut (spacebar) to start playback
            time.sleep(1)
            pyautogui.press('space')
            print("✅ Attempted to start Spotify with keyboard shortcut")
                
        elif os.system("which rhythmbox > /dev/null 2>&1") == 0:
            print("Opening Rhythmbox...")
            if music_file:
                # Start Rhythmbox and add the file to library/queue
                subprocess.Popen(['rhythmbox', music_file])
                print(f"✅ Playing {os.path.basename(music_file)} in Rhythmbox")
                # Wait a moment for Rhythmbox to load, then try to start playback
                time.sleep(3)
                if os.system("which playerctl > /dev/null 2>&1") == 0:
                    os.system("playerctl -p rhythmbox play 2>/dev/null")
            else:
                subprocess.Popen(['rhythmbox'])
                print("✅ Started Rhythmbox")
                # Wait a moment and try to start playback
                time.sleep(3)
                if os.system("which playerctl > /dev/null 2>&1") == 0:
                    os.system("playerctl -p rhythmbox play 2>/dev/null")
                else:
                    # Fallback: use keyboard shortcut
                    pyautogui.press('space')
            
        elif os.system("which audacious > /dev/null 2>&1") == 0:
            print("Opening Audacious...")
            if music_file:
                subprocess.Popen(['audacious', '--play-pause', music_file])
                print(f"✅ Playing {os.path.basename(music_file)} in Audacious")
            else:
                subprocess.Popen(['audacious', '--play-pause'])
                print("✅ Started Audacious playback")
                
        elif os.system("which clementine > /dev/null 2>&1") == 0:
            print("Opening Clementine...")
            if music_file:
                subprocess.Popen(['clementine', music_file])
                print(f"✅ Playing {os.path.basename(music_file)} in Clementine")
                # Wait and try to start playback
                time.sleep(2)
                if os.system("which playerctl > /dev/null 2>&1") == 0:
                    os.system("playerctl -p clementine play 2>/dev/null")
            else:
                subprocess.Popen(['clementine'])
                print("✅ Started Clementine")
                time.sleep(2)
                if os.system("which playerctl > /dev/null 2>&1") == 0:
                    os.system("playerctl -p clementine play 2>/dev/null")
                else:
                    pyautogui.press('space')
                
        elif os.system("which mpv > /dev/null 2>&1") == 0 and music_file:
            print(f"Playing {os.path.basename(music_file)} with mpv...")
            subprocess.Popen(['mpv', '--no-video', music_file])
            print("✅ Started mpv playback")
            
        elif os.system("which vlc > /dev/null 2>&1") == 0:
            print("Opening VLC...")
            if music_file:
                subprocess.Popen(['vlc', '--intf', 'qt', '--started-from-file', music_file])
                print(f"✅ Playing {os.path.basename(music_file)} in VLC")
            else:
                subprocess.Popen(['vlc'])
                print("✅ Opened VLC")
                # Wait and try to start playback if any media is loaded
                time.sleep(2)
                pyautogui.press('space')
                
        # macOS
        elif os.system("which open > /dev/null 2>&1") == 0:
            print("Opening Music app on macOS...")
            if music_file:
                os.system(f"open '{music_file}'")
                print(f"✅ Playing {os.path.basename(music_file)}")
            else:
                os.system("open -a Music")
                time.sleep(2)
                # Try to start playback with spacebar
                pyautogui.press('space')
                print("✅ Attempted to start Music app playback")
            
        else:
            print("❌ No music player found")
            if music_file:
                print(f"Found music file: {music_file}")
                print("Try installing: spotify, rhythmbox, audacious, clementine, mpv, or vlc")
            
    except Exception as e:
        print(f"❌ Error playing music: {e}")

def find_music_file():
    """Find a music file to play from common directories"""
    import glob
    import random
    
    music_dirs = [
        os.path.expanduser("~/Music"),
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Documents"),
        "/home/*/Music"
    ]
    
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.ogg', '*.m4a', '*.aac', '*.wma']
    
    for music_dir in music_dirs:
        if os.path.exists(music_dir):
            all_files = []
            for ext in audio_extensions:
                files = glob.glob(os.path.join(music_dir, "**", ext), recursive=True)
                all_files.extend(files)
            
            if all_files:
                # Return a random music file for variety
                return random.choice(all_files)
    
    return None

def stop_music():
    """Stop music playback"""
    print("⏹️ Stopping music...")
    try:
        # Try to pause/stop using playerctl (works with most MPRIS-compatible players)
        if os.system("which playerctl > /dev/null 2>&1") == 0:
            result = os.system("playerctl pause 2>/dev/null")
            if result == 0:
                print("✅ Paused music using playerctl")
                return
            else:
                # Try to stop instead of pause
                result = os.system("playerctl stop 2>/dev/null")
                if result == 0:
                    print("✅ Stopped music using playerctl")
                    return
        
        # Try killing common music players as fallback
        music_players = ['spotify', 'rhythmbox', 'audacious', 'clementine', 'vlc', 'mpv']
        for player in music_players:
            result = os.system(f"pkill -f {player} 2>/dev/null")
            if result == 0:
                print(f"✅ Stopped {player}")
                return
        
        # macOS fallback
        if os.system("which osascript > /dev/null 2>&1") == 0:
            os.system("""osascript -e 'tell application "Music" to pause'""")
            print("✅ Paused Music app on macOS")
            
        else:
            print("⚠️ No running music players found to stop")
            
    except Exception as e:
        print(f"❌ Error stopping music: {e}")

def switch_window():
    """Switch to next window (Alt+Tab)"""
    print("🔄 Switching window...")
    try:
        pyautogui.hotkey('alt', 'tab')
    except Exception as e:
        print(f"Error switching window: {e}")

def volume_down():
    """Decrease system volume"""
    print("🔉 Volume down...")
    try:
        # Linux
        if os.system("which pactl > /dev/null 2>&1") == 0:
            os.system("pactl set-sink-volume @DEFAULT_SINK@ -10%")
        # macOS
        elif os.system("which osascript > /dev/null 2>&1") == 0:
            os.system("""osascript -e 'set volume output volume (output volume of (get volume settings) - 10)'""")
        else:
            print("Volume control not implemented for this system")
    except Exception as e:
        print(f"Error decreasing volume: {e}")

def volume_up():
    """Increase system volume"""
    print("🔊 Volume up...")
    try:
        # Linux
        if os.system("which pactl > /dev/null 2>&1") == 0:
            os.system("pactl set-sink-volume @DEFAULT_SINK@ +10%")
        # macOS
        elif os.system("which osascript > /dev/null 2>&1") == 0:
            os.system("""osascript -e 'set volume output volume (output volume of (get volume settings) + 10)'""")
        else:
            print("Volume control not implemented for this system")
    except Exception as e:
        print(f"Error increasing volume: {e}")

# Command mapping
COMMAND_FUNCTIONS = {
    '<close_browser>': close_browser,
    '<google>': open_google,
    '<maximize_window>': maximize_window,
    '<mute>': mute_audio,
    '<no_action>': no_action,
    '<open_browser>': open_browser,
    '<open_notepad>': open_notepad,
    '<play_music>': play_music,
    '<stop_music>': stop_music,
    '<switch_window>': switch_window,
    '<volume_down>': volume_down,
    '<volume_up>': volume_up
}

def execute_command(predicted_class, confidence):
    """Execute the predicted command"""
    # Check confidence threshold
    if confidence < CONFIDENCE_THRESHOLD:  # Adjust threshold as needed
        print(f"⚠️ Low confidence ({confidence:.3f}), ignoring command")
        return
    
    # Execute command if it exists in our mapping
    if predicted_class in COMMAND_FUNCTIONS:
        try:
            COMMAND_FUNCTIONS[predicted_class]()
        except Exception as e:
            print(f"❌ Error executing command {predicted_class}: {e}")
    else:
        print(f"⚠️ Unknown command: {predicted_class}")

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
            predicted_class, confidence = classify_audio(audio_np)
            if predicted_class:
                execute_command(predicted_class, confidence)
    if key == keyboard.Key.esc:
        # Stop listener
        return False

if __name__ == "__main__":
    print("🎯 Voice Command Classifier Deployment")
    print("=" * 50)
    print("Commands available:")
    for i, cmd in enumerate(COMMAND_TOKENS, 1):
        print(f"  {i:2d}. {cmd}")
    print("=" * 50)
    print("Hold SPACE to record. Release to classify and execute. Press ESC to quit.")
    print(f"Audio backend: {AUDIO_BACKEND}")
    
    if AUDIO_BACKEND == "ffmpeg":
        print("📋 Note: Using ffmpeg backend. Make sure your microphone is set as default ALSA device.")
        print("📋 You may need to install librosa: pip install librosa")
    
    print("\n🎤 Ready for voice commands...")
    
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

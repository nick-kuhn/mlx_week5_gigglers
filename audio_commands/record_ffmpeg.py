import subprocess
import os

def record(filename, duration=15, samplerate=16000, folder='data'):
    """Record audio using ffmpeg command"""
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    
    print(f"Recording for {duration} seconds...")
    
    # Use ffmpeg to record audio
    cmd = [
        'ffmpeg',
        '-f', 'alsa',           # input format
        '-i', 'default',        # default audio input
        '-t', str(duration),    # duration
        '-ar', str(samplerate), # sample rate
        '-ac', '1',             # mono
        '-y',                   # overwrite output file
        filepath
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Saved to {filepath}")
        else:
            print(f"Error: {result.stderr}")
    except FileNotFoundError:
        print("Error: ffmpeg command not found. Please install ffmpeg:")
        print("sudo apt-get install ffmpeg")

if __name__ == "__main__":
    while True:
        filename = input("Enter filename (e.g. ask_chatgpt_001.wav) or ENTER to quit: ")
        if not filename.strip():
            break
        record(filename, duration=15)

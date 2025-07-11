import sounddevice as sd
import soundfile as sf
import os

def record(filename, duration=15, samplerate=16000, folder='data'):
    os.makedirs(folder, exist_ok=True)  # Make sure 'data/' exists
    filepath = os.path.join(folder, filename)
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(filepath, audio, samplerate)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    while True:
        filename = input("Enter filename (e.g. ask_chatgpt_001.wav) or ENTER to quit: ")
        if not filename.strip():
            break
        record(filename, duration=15)
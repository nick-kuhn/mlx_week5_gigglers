#!/usr/bin/env python3
"""
Voice Command Dataset Recorder using ffmpeg for audio capture.
"""
import os
import subprocess
import signal
import time
import csv
import random
from pathlib import Path
from datetime import datetime

# Global constants
COMMAND_TOKENS: list[str] = [
    '<close_browser>', '<google>', '<maximize_window>', '<mute>', '<no_action>',
    '<open_browser>', '<open_notepad>', '<play_music>', '<stop_music>',
    '<switch_window>', '<volume_down>', '<volume_up>'
]
SAMPLERATE: int = 16000
CHANNELS: int = 1
CSV_FILE: str = "data/validation/recordings.csv"
RECORDINGS_DIR: Path = Path("data/validation")


class VoiceCommandRecorder:
    """
    Handles recording voice commands via ffmpeg and managing metadata CSV.
    """

    def __init__(self) -> None:
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        self.csv_file = CSV_FILE
        self.recordings_dir = RECORDINGS_DIR
        self.process = None
        self.start_time = 0.0
        self.current_filepath: Path | None = None
        self.init_csv()

    def init_csv(self) -> None:
        """
        Create the CSV file with headers if it doesn't exist.
        """
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'command_token', 'timestamp', 'duration_seconds'])
            print(f"📝 Created new CSV file: {self.csv_file}")
        else:
            print(f"📝 Using existing CSV file: {self.csv_file}")

    def get_next_filename(self, token: str) -> str:
        """
        Generate a unique filename for the given command token.
        """
        clean = token.replace('<', '').replace('>', '')
        existing = list(self.recordings_dir.glob(f"*_{clean}_*.wav"))
        count = len(existing) + 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"voice_{clean}_{count:03d}_{timestamp}.wav"

    def start_recording(self) -> None:
        """
        Spawn an ffmpeg process to start recording audio.
        """
        assert self.current_filepath is not None
        cmd = [
            'ffmpeg', '-f', 'alsa', '-i', 'default',
            '-ar', str(SAMPLERATE), '-ac', str(CHANNELS),
            '-y', str(self.current_filepath)
        ]
        # Suppress ffmpeg output AND redirect stdin to prevent it from consuming input
        self.process = subprocess.Popen(cmd, 
                                       stdout=subprocess.DEVNULL, 
                                       stderr=subprocess.DEVNULL,
                                       stdin=subprocess.DEVNULL)
        self.start_time = time.time()
        print("🎤 Recording started... press ENTER to stop")

    def stop_recording(self) -> float:
        """
        Stop the ffmpeg process and return the duration recorded.
        """
        if self.process:
            # Send SIGINT to allow ffmpeg to finalise the file
            self.process.send_signal(signal.SIGINT)
            self.process.wait()
        end_time = time.time()
        duration = end_time - self.start_time
        print("⏹️  Recording stopped!")
        return duration

    def save_recording(self, filepath: Path, token: str, duration: float) -> bool:
        """
        Append the recording metadata to the CSV.
        """
        timestamp = datetime.now().isoformat()
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([str(filepath), token, timestamp, f"{duration:.2f}"])
            print(f"💾 Saved {duration:.2f}s to {filepath}")
            print(f"📝 Updated {self.csv_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to save CSV entry: {e}")
            return False

    def record_for_command(self, token: str) -> str:
        """
        Record audio for a specific command token.
        """
        print(f"\n🎯 RECORDING FOR COMMAND: {token}")
        filename = self.get_next_filename(token)
        self.current_filepath = self.recordings_dir / filename
        print(f"📁 Will save as: {filename}")
        
        try:
            while True:
                user_input = input("Press ENTER to start recording (or 'q' to quit): ").strip()
                if user_input.lower() == 'q':
                    print("🚪 Exiting session...")
                    return "exit"
                
                # Start recording
                self.start_recording()
                
                # Simple approach: record for a fixed duration or until user stops
                print("🎤 Recording started! Choose an option:")
                print("  1. Press ENTER to stop now")
                print("  2. Type a number (1-10) to record for that many seconds")
                print("  3. Type 'q' to quit")
                
                try:
                    stop_input = input("Your choice: ").strip()
                    
                    if stop_input.lower() == 'q':
                        self.stop_recording()
                        print("🚪 Exiting session...")
                        return "exit"
                    elif stop_input.isdigit():
                        # Record for specified seconds
                        seconds = int(stop_input)
                        if 1 <= seconds <= 10:
                            print(f"⏱️  Recording for {seconds} seconds...")
                            time.sleep(seconds)
                        else:
                            print("⏱️  Recording for 3 seconds (default)...")
                            time.sleep(3)
                    # If empty (just ENTER) or anything else, stop immediately
                    
                except KeyboardInterrupt:
                    pass
                
                # Stop recording
                duration = self.stop_recording()
                
                if self.save_recording(self.current_filepath, token, duration):
                    return "success"
                return "error"

        except KeyboardInterrupt:
            if self.process:
                self.stop_recording()
            return "exit"

    def show_example_phrases(self, token: str) -> None:
        """
        Print example phrases for guidance.
        """
        examples = {
            '<close_browser>': ["Close the browser", "Shut down browser", "Exit web browser"],
            # ... other tokens omitted for brevity ...
        }
        print("   Examples:")
        for ex in examples.get(token, ["Say the command naturally"]):
            print(f"   • {ex}")

    def show_csv_stats(self) -> None:
        """
        Display statistics of all recordings.
        """
        if not os.path.exists(self.csv_file):
            print("📊 No recordings yet!")
            return
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print("📊 No recordings yet!")
            return
        counts: dict[str,int] = {}
        total = 0.0
        for row in rows:
            counts[row['command_token']] = counts.get(row['command_token'], 0) + 1
            total += float(row['duration_seconds'])
        print(f"\n📊 Total recordings: {len(rows)}, total duration: {total:.1f}s")
        for t in COMMAND_TOKENS:
            print(f"  {t}: {counts.get(t, 0)} recordings")


def main() -> None:
    recorder = VoiceCommandRecorder()
    print("🎤 Voice Command Dataset Recorder")
    while True:
        print("\n1: Random commands  2: Specific command  3: Stats  4: Quit")
        choice = input("Choose (1-4): ").strip()
        if choice == '4':
            break
        if choice == '3':
            recorder.show_csv_stats()
        elif choice == '1':
            while True:
                cmd = random.choice(COMMAND_TOKENS)
                res = recorder.record_for_command(cmd)
                if res != "success":
                    break
        elif choice == '2':
            for i, t in enumerate(COMMAND_TOKENS, 1):
                print(f"{i}. {t}")
            idx = input("Cmd #: ").strip()
            try:
                token = COMMAND_TOKENS[int(idx)-1]
                while recorder.record_for_command(token) == "success":
                    pass
            except Exception:
                print("❌ Invalid selection")
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()

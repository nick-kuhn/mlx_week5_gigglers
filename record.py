import sounddevice as sd
import soundfile as sf
import os
import threading
import time
import keyboard
import csv
import random
from pathlib import Path
from datetime import datetime

# Define the 12 command tokens
COMMAND_TOKENS = [
    '<close_browser>', '<google>', '<maximize_window>', '<minimize_window>', '<mute>', '<no_action>', 
    '<open_browser>', '<open_notepad>', '<play_music>', '<stop_music>', 
    '<switch_window>', '<volume_down>', '<volume_up>'
]

class VoiceCommandRecorder:
    def __init__(self, samplerate=16000, channels=1):
        self.samplerate = samplerate
        self.channels = channels
        self.recording = False
        self.audio_data = []
        self.stream = None
        self.csv_file = "recordings.csv"
        self.recordings_dir = Path("data/recordings")
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.init_csv()
        
    def init_csv(self):
        """Initialize the recordings.csv file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'command_token', 'timestamp', 'duration_seconds'])
            print(f"📝 Created new CSV file: {self.csv_file}")
        else:
            print(f"📝 Using existing CSV file: {self.csv_file}")
    
    def get_next_filename(self, command_token):
        """Generate the next filename for a given command token."""
        # Remove angle brackets from command token for filename
        clean_command = command_token.replace('<', '').replace('>', '')
        
        # Count existing files for this command
        existing_files = list(self.recordings_dir.glob(f"*_{clean_command}_*.wav"))
        count = len(existing_files) + 1
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"voice_{clean_command}_{count:03d}_{timestamp}.wav"
        return filename
    
    def audio_callback(self, indata, frames, time, status):
        """Callback function for the audio stream."""
        if status:
            print(f"Audio callback status: {status}")
        if self.recording:
            self.audio_data.append(indata.copy())
    
    def start_recording(self):
        """Start recording audio."""
        self.recording = True
        self.audio_data = []
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            callback=self.audio_callback
        )
        self.stream.start()
        print("🎤 Recording started! Release SPACEBAR to stop...")
    
    def stop_recording(self):
        """Stop recording audio."""
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("⏹️  Recording stopped!")
    
    def save_recording(self, filename, command_token):
        """Save the recorded audio and update CSV."""
        if not self.audio_data:
            print("No audio data to save!")
            return False
        
        # Concatenate all audio chunks
        import numpy as np
        audio_array = np.concatenate(self.audio_data, axis=0)
        
        # Save to file
        filepath = self.recordings_dir / filename
        sf.write(filepath, audio_array, self.samplerate)
        duration = len(audio_array) / self.samplerate
        
        # Update CSV
        timestamp = datetime.now().isoformat()
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([str(filepath), command_token, timestamp, f"{duration:.2f}"])
        
        print(f"💾 Saved {duration:.2f} seconds of audio to {filepath}")
        print(f"📝 Added entry to {self.csv_file}")
        return True
    
    def record_for_command(self, command_token):
        """Record audio for a specific command using walkie-talkie style."""
        print("\n" + "="*60)
        print(f"🎯 RECORDING FOR COMMAND: {command_token}")
        print("="*60)
        
        # Generate filename
        filename = self.get_next_filename(command_token)
        
        print(f"📁 Will save as: {filename}")
        print("\n🎵 Walkie-Talkie Recording Mode")
        print("Hold SPACEBAR to record your voice command")
        print("Release SPACEBAR to stop and save")
        print("Press 'q' to exit session and return to main menu")
        print("\nExample phrases you could say:")
        self.show_example_phrases(command_token)
        
        print(f"\n💡 Say something like: 'Please {command_token.replace('<', '').replace('>', '').replace('_', ' ')} now'")
        print("Ready? Hold SPACEBAR when you want to start recording...")
        
        try:
            while True:
                # Check if spacebar is pressed
                if keyboard.is_pressed('space'):
                    if not self.recording:
                        self.start_recording()
                    time.sleep(0.1)
                else:
                    if self.recording:
                        self.stop_recording()
                        if self.save_recording(filename, command_token):
                            print("✅ Recording saved successfully!")
                            return "success"  # Recording completed successfully
                        else:
                            print("❌ Failed to save recording")
                            return "error"  # Error occurred
                    time.sleep(0.1)
                    
                # Check for quit
                if keyboard.is_pressed('q'):
                    if self.recording:
                        self.stop_recording()
                    print("🚪 Exiting session...")
                    return "exit"  # User wants to exit session
                    
        except KeyboardInterrupt:
            print("\n🛑 Recording interrupted by user")
            if self.recording:
                self.stop_recording()
            return "exit"  # User pressed Ctrl+C, exit session
    
    def show_example_phrases(self, command_token):
        """Show example phrases for each command."""
        examples = {
            '<close_browser>': ["Close the browser", "Shut down browser", "Exit web browser"],
            '<google>': ["Open Google", "Search on Google", "Go to Google"],
            '<maximize_window>': ["Maximize window", "Make window bigger", "Full screen"],
            '<minimize_window>': ["Minimize the window", "Make it smaller", "Hide the window"],
            '<mute>': ["Mute audio", "Turn off sound", "Silence"],
            '<no_action>': ["Do nothing", "No action needed", "Stay as is"],
            '<open_browser>': ["Open browser", "Launch web browser", "Start browser"],
            '<open_notepad>': ["Open notepad", "Launch text editor", "Start notepad"],
            '<play_music>': ["Play music", "Start music", "Begin playback"],
            '<stop_music>': ["Stop music", "Pause music", "End playback"],
            '<switch_window>': ["Switch window", "Change window", "Next window"],
            '<volume_down>': ["Turn down volume", "Lower volume", "Decrease sound"],
            '<volume_up>': ["Turn up volume", "Increase volume", "Louder"]
        }
        
        command_examples = examples.get(command_token, ["Say the command naturally"])
        print("   Example phrases:")
        for example in command_examples:
            print(f"   • '{example}'")
    
    def show_csv_stats(self):
        """Show statistics about current recordings."""
        if not os.path.exists(self.csv_file):
            print("📊 No recordings yet!")
            return
        
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            print("📊 No recordings yet!")
            return
        
        # Count recordings per command
        command_counts = {}
        total_duration = 0
        
        for row in rows:
            command = row['command_token']
            duration = float(row['duration_seconds'])
            command_counts[command] = command_counts.get(command, 0) + 1
            total_duration += duration
        
        print(f"\n📊 RECORDING STATISTICS")
        print("="*40)
        print(f"Total recordings: {len(rows)}")
        print(f"Total duration: {total_duration:.1f} seconds")
        print(f"Average duration: {total_duration/len(rows):.1f} seconds")
        print("\nRecordings per command:")
        
        for command in COMMAND_TOKENS:
            count = command_counts.get(command, 0)
            print(f"  {command}: {count} recordings")

def main():
    recorder = VoiceCommandRecorder()
    
    print("🎤 Voice Command Dataset Recorder")
    print("="*50)
    
    while True:
        print("\nChoose recording mode:")
        print("1. Record random commands (recommended for balanced dataset)")
        print("2. Record specific command repeatedly")
        print("3. Show recording statistics")
        print("4. Quit")
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '4':
            print("👋 Goodbye!")
            break
        elif choice == '3':
            recorder.show_csv_stats()
            continue
        elif choice == '1':
            # Random command mode with continuous recording
            print("\n🎲 RANDOM COMMAND RECORDING MODE")
            print("="*50)
            print("You'll be prompted to record different commands randomly.")
            print("Press 'q' during any recording to exit back to main menu.")
            print("Press 'Ctrl+C' to exit back to main menu.")
            print("\nStarting random command session...")
            
            session_count = 0
            try:
                while True:
                    # Select random command
                    command_token = random.choice(COMMAND_TOKENS)
                    print(f"\n🎲 Session #{session_count + 1} - Random command: {command_token}")
                    
                    # Record for this command
                    result = recorder.record_for_command(command_token)
                    
                    if result == "success":
                        session_count += 1
                        print(f"✅ Session #{session_count} completed! Preparing next random command...")
                        time.sleep(1)  # Brief pause before next command
                    elif result == "exit":
                        # User pressed 'q' or Ctrl+C, exit the session
                        print("🚪 Exiting random command session...")
                        break
                    else:  # result == "error"
                        print("🔄 Error occurred, trying next command...")
                        time.sleep(0.5)
                    
            except KeyboardInterrupt:
                print(f"\n\n🛑 Random recording session ended! Completed {session_count} recordings.")
            
            print(f"📊 Session summary: {session_count} recordings completed.")
            print("Returning to main menu...")
            
        elif choice == '2':
            # Specific command mode with continuous recording
            print("\nAvailable commands:")
            for i, cmd in enumerate(COMMAND_TOKENS, 1):
                print(f"  {i:2d}. {cmd}")
            
            try:
                cmd_choice = input(f"\nEnter command number (1-{len(COMMAND_TOKENS)}): ").strip()
                cmd_index = int(cmd_choice) - 1
                
                if 0 <= cmd_index < len(COMMAND_TOKENS):
                    command_token = COMMAND_TOKENS[cmd_index]
                    
                    print(f"\n🎯 SPECIFIC COMMAND RECORDING MODE")
                    print("="*50)
                    print(f"Recording multiple samples for: {command_token}")
                    print("Press 'q' during any recording to exit back to main menu.")
                    print("Press 'Ctrl+C' to exit back to main menu.")
                    print(f"\nStarting recording session for {command_token}...")
                    
                    session_count = 0
                    try:
                        while True:
                            print(f"\n🎯 Sample #{session_count + 1} for {command_token}")
                            
                            # Record for this specific command
                            result = recorder.record_for_command(command_token)
                            
                            if result == "success":
                                session_count += 1
                                print(f"✅ Sample #{session_count} completed! Preparing next sample...")
                                time.sleep(1)  # Brief pause before next sample
                            elif result == "exit":
                                # User pressed 'q' or Ctrl+C, exit the session
                                print(f"🚪 Exiting recording session for {command_token}...")
                                break
                            else:  # result == "error"
                                print("🔄 Error occurred, trying again...")
                                time.sleep(0.5)
                            
                    except KeyboardInterrupt:
                        print(f"\n\n🛑 Recording session ended! Completed {session_count} samples for {command_token}.")
                    
                    print(f"📊 Session summary: {session_count} samples recorded for {command_token}.")
                    print("Returning to main menu...")
                    
                else:
                    print("❌ Invalid command number!")
                    
            except ValueError:
                print("❌ Please enter a valid number!")
        else:
            print("❌ Invalid choice! Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Recording session ended!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Make sure you have the required dependencies:")
        print("uv add keyboard sounddevice soundfile numpy")

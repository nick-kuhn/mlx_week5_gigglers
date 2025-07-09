#!/usr/bin/env python3
"""
Audio Generation Script for Command Dataset

This script processes commands.csv to:
1. Extract sentences with special tokens
2. Remove special tokens to create clean sentences
3. Generate audio files from clean sentences
4. Create a new CSV with audio paths, original sentences, clean sentences, and class labels
"""

import csv
import os
import re
import pyttsx3
from pathlib import Path
import argparse
from tqdm import tqdm
import time
import random

# Add these:
from pydub import AudioSegment

class AudioGenerator:
    def __init__(
        self,
        output_dir: str = "ben_branch/data/generated_audio",
        voice_id: int = 0,
        rate: int = 150,
        volume: float = 1.0,
        random_voices: bool = True
    ) -> None:
        """
        Initialize the audio generator.

        Args:
            output_dir: Directory to save audio files.
            voice_id: Voice ID to use (ignored if random_voices=True).
            rate: Speech rate in words per minute.
            volume: Volume level (0.0 to 1.0).
            random_voices: Whether to randomly select voices.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialise TTS engine and get all voices
        self.engine = pyttsx3.init()
        all_voices = self.engine.getProperty("voices")

        # Keep only English voices
        self.voices = [
            v for v in all_voices
            if "English" in v.name or "english" in v.name
        ]

        self.voice_id = voice_id
        self.rate = rate
        self.volume = volume
        self.random_voices = random_voices

        if not self.voices:
            raise RuntimeError("No English voices found!")

        if self.random_voices:
            print(f"Random English voice selection: {len(self.voices)} voices available.")
            for i, v in enumerate(self.voices):
                print(f"  {i}: {v.name}")
        else:
            if 0 <= self.voice_id < len(self.voices):
                print(f"Voice set to: {self.voices[self.voice_id].name}")
            else:
                print(f"Warning: Voice ID {voice_id} not in English list, defaulting to 0.")
                self.voice_id = 0

    
    def set_voice(self, voice_id):
        """Set the voice to use for TTS."""
        if self.voices and 0 <= voice_id < len(self.voices):
            self.voice_id = voice_id
            print(f"Voice set to: {self.voices[voice_id].name}")
        else:
            print(f"Warning: Voice ID {voice_id} not available. Using default voice.")
    
    def set_rate(self, rate):
        """Set the speech rate."""
        self.rate = rate
    
    def set_volume(self, volume):
        """Set the volume level."""
        volume = max(0.0, min(1.0, volume))  # Clamp between 0.0 and 1.0
        self.volume = volume
    
    def extract_special_token(self, sentence):
        """
        Extract the special token from a sentence.
        
        Args:
            sentence (str): Sentence containing special token
            
        Returns:
            str: The special token (e.g., '<volume_up>')
        """
        # Find all tokens enclosed in angle brackets
        tokens = re.findall(r'<[^>]+>', sentence)
        
        # Return the first token found, or empty string if none
        return tokens[0] if tokens else ""
    
    def remove_special_tokens(self, sentence):
        """
        Remove all special tokens from a sentence.
        
        Args:
            sentence (str): Sentence with special tokens
            
        Returns:
            str: Clean sentence without special tokens
        """
        # Remove all tokens enclosed in angle brackets
        clean_sentence = re.sub(r'<[^>]+>', '', sentence)
        
        # Clean up extra whitespace
        clean_sentence = ' '.join(clean_sentence.split())
        
        return clean_sentence.strip()
    
    def generate_audio_filename(self, row_id, class_label):
        """
        Generate a filename for the audio file.
        
        Args:
            row_id (int): Row ID from CSV
            class_label (str): Class label (special token)
            
        Returns:
            str: Filename for the audio file
        """
        # Remove angle brackets from class label for filename
        clean_label = class_label.replace('<', '').replace('>', '')
        filename = f"audio_{row_id:04d}_{clean_label}.wav"
        return filename
    

    def generate_audio(self, text: str, filename: str) -> tuple[bool, dict]:
        """
        Generate an audio file from text using the filtered English voices,
        then resample the result to 16 kHz.

        Args:
            text: The text to synthesise.
            filename: The target WAV filename.

        Returns:
            A tuple (success, voice_info), where success is True if the file
            was generated (and resampled) without error, and voice_info gives
            which voice was used.
        """
        try:
            # Prepare file path and TTS engine
            filepath = self.output_dir / filename
            engine = self.engine

            # Choose voice
            if self.random_voices:
                vid = random.randrange(len(self.voices))
            else:
                vid = self.voice_id

            # Apply TTS properties
            engine.setProperty("voice", self.voices[vid].id)
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)

            voice_info = {"voice_id": vid, "voice_name": self.voices[vid].name}

            # Generate raw WAV
            engine.save_to_file(text, str(filepath))
            engine.runAndWait()

            # Resample to 16 kHz
            try:
                audio = AudioSegment.from_file(str(filepath))
                audio = audio.set_frame_rate(16000)
                audio.export(str(filepath), format="wav")
            except Exception as e:
                print(f"Warning: could not resample {filename}: {e}")

            return True, voice_info

        except Exception as e:
            print(f"Error generating {filename}: {e}")
            return False, {"voice_id": -1, "voice_name": "Error"}

    
    def process_csv(self, input_csv="commands.csv", output_csv="audio_dataset.csv"):
        """
        Process the commands CSV file and generate audio files.
        
        Args:
            input_csv (str): Path to input CSV file
            output_csv (str): Path to output CSV file
        """
        print(f"Processing {input_csv}...")
        print(f"Output directory: {self.output_dir}")
        print(f"Output CSV: {output_csv}")
        
        # Read input CSV
        try:
            with open(input_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except FileNotFoundError:
            print(f"Error: {input_csv} not found!")
            return
        
        print(f"Found {len(rows)} rows to process")
        
        # Prepare output data
        output_data = []
        
        # Process each row
        for i, row in enumerate(tqdm(rows, desc="Generating audio")):
            # Extract data from row
            row_id = i + 1  # Use sequential ID
            original_sentence = row.get('sentence', '').strip()
            
            if not original_sentence:
                print(f"Warning: Empty sentence in row {row_id}")
                continue
            
            # Extract special token (class label)
            class_label = self.extract_special_token(original_sentence)
            
            if not class_label:
                print(f"Warning: No special token found in row {row_id}: {original_sentence}")
                continue
            
            # Remove special tokens to get clean sentence
            clean_sentence = self.remove_special_tokens(original_sentence)
            
            if not clean_sentence:
                print(f"Warning: Empty clean sentence in row {row_id}")
                continue
            
            # Generate audio filename
            audio_filename = self.generate_audio_filename(row_id, class_label)
            
            # Generate audio file
            success, voice_info = self.generate_audio(clean_sentence, audio_filename)
            
            if success:
                # Prepare output row
                output_row = {
                    'audio_path': str(self.output_dir / audio_filename),
                    'original_sentence': original_sentence,
                    'clean_sentence': clean_sentence,
                    'class_label': class_label,
                    'voice_id': voice_info['voice_id'],
                    'voice_name': voice_info['voice_name']
                }
                output_data.append(output_row)
            else:
                print(f"Failed to generate audio for row {row_id}")
        
        # Write output CSV
        if output_data:
            try:
                with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['audio_path', 'original_sentence', 'clean_sentence', 'class_label', 'voice_id', 'voice_name']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(output_data)
                
                print(f"\nSuccess! Generated {len(output_data)} audio files")
                print(f"Output CSV saved to: {output_csv}")
                
                # Print some statistics
                self.print_statistics(output_data)
                
            except Exception as e:
                print(f"Error writing output CSV: {e}")
        else:
            print("No audio files were generated successfully!")
    
    def print_statistics(self, data):
        """Print statistics about the generated dataset."""
        print("\n=== Dataset Statistics ===")
        
        # Count class labels
        class_counts = {}
        voice_counts = {}
        for row in data:
            label = row['class_label']
            class_counts[label] = class_counts.get(label, 0) + 1
            
            voice_name = row.get('voice_name', 'Unknown')
            voice_counts[voice_name] = voice_counts.get(voice_name, 0) + 1
        
        print(f"Total samples: {len(data)}")
        print(f"Number of classes: {len(class_counts)}")
        print("\nClass distribution:")
        for label, count in sorted(class_counts.items()):
            print(f"  {label}: {count} samples")
        
        if self.random_voices:
            print(f"\nVoice distribution (random voices enabled):")
            for voice_name, count in sorted(voice_counts.items()):
                print(f"  {voice_name}: {count} samples")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Generate audio files from commands CSV')
    
    parser.add_argument('--input', '-i', default='commands.csv', 
                       help='Input CSV file (default: commands.csv)')
    parser.add_argument('--output', '-o', default='audio_dataset.csv',
                       help='Output CSV file (default: audio_dataset.csv)')
    parser.add_argument('--audio-dir', '-d', default='ben_branch/data/generated_audio',
                       help='Directory for audio files (default: ben_branch/data/generated_audio)')
    parser.add_argument('--voice', '-v', type=int, default=0,
                       help='Voice ID to use (default: 0)')
    parser.add_argument('--rate', '-r', type=int, default=200,
                       help='Speech rate in WPM (default: 200)')
    parser.add_argument('--volume', type=float, default=1.0,
                       help='Volume level 0.0-1.0 (default: 1.0)')
    parser.add_argument('--disable-random-voices', action='store_true',
                       help='Randomly select voices for each audio file')
    parser.add_argument('--list-voices', action='store_true',
                       help='List available voices and exit')
    
    args = parser.parse_args()
    
    # List voices if requested
    if args.list_voices:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        print("Available voices:")
        for i, voice in enumerate(voices):
            print(f"  {i}: {voice.name} ({getattr(voice, 'gender', 'Unknown')})")
        return
    
    # Initialize audio generator
    print("Initializing audio generator...")
    generator = AudioGenerator(
        output_dir=args.audio_dir,
        voice_id=args.voice,
        rate=args.rate,
        volume=args.volume,
        random_voices=not args.disable_random_voices
    )
    
    # Process the CSV
    start_time = time.time()
    generator.process_csv(args.input, args.output)
    end_time = time.time()
    
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()

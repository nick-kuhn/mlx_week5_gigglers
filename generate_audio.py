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
from gtts import gTTS
from pathlib import Path
import argparse
from tqdm import tqdm
import time
import random
from pydub import AudioSegment

class AudioGenerator:
    def __init__(
        self,
        output_dir: str = "ben_branch/data/generated_audio",
    ) -> None:
        """
        Initialize the audio generator.

        Args:
            output_dir: Directory to save audio files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def extract_special_token(self, sentence):
        """
        Extract the special token from a sentence.
        Args:
            sentence (str): Sentence containing special token
        Returns:
            str: The special token (e.g., '<volume_up>')
        """
        tokens = re.findall(r'<[^>]+>', sentence)
        return tokens[0] if tokens else ""

    def remove_special_tokens(self, sentence):
        """
        Remove all special tokens from a sentence.
        Args:
            sentence (str): Sentence with special tokens
        Returns:
            str: Clean sentence without special tokens
        """
        clean_sentence = re.sub(r'<[^>]+>', '', sentence)
        clean_sentence = ' '.join(clean_sentence.split())
        return clean_sentence.strip()

    def generate_audio_filename(self, row_id, class_label):
        clean_label = class_label.replace('<', '').replace('>', '')
        filename = f"audio_{row_id:04d}_{clean_label}.wav"
        return filename

    def generate_audio(self, text: str, filename: str) -> tuple[bool, dict]:
        """
        Generate an audio file from text using gTTS, then resample the result to 16 kHz.
        Args:
            text: The text to synthesise.
            filename: The target WAV filename.
        Returns:
            A tuple (success, info), where success is True if the file was generated (and resampled) without error.
        """
        try:
            filepath = self.output_dir / filename
            mp3_path = filepath.with_suffix('.mp3')
            # Generate mp3 with gTTS
            tts = gTTS(text=text, lang='en')
            tts.save(str(mp3_path))
            # Convert to wav and resample to 16kHz
            audio = AudioSegment.from_file(str(mp3_path))
            audio = audio.set_frame_rate(16000)
            audio.export(str(filepath), format="wav")
            os.remove(mp3_path)
            return True, {"info": "gTTS, 16kHz"}
        except Exception as e:
            print(f"Error generating {filename}: {e}")
            return False, {"info": "Error"}

    def process_csv(self, input_csv="commands.csv", output_csv="audio_dataset.csv"):
        print(f"Processing {input_csv}...")
        print(f"Output directory: {self.output_dir}")
        print(f"Output CSV: {output_csv}")
        try:
            with open(input_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except FileNotFoundError:
            print(f"Error: {input_csv} not found!")
            return
        print(f"Found {len(rows)} rows to process")
        output_data = []
        for i, row in enumerate(tqdm(rows, desc="Generating audio")):
            row_id = i + 1
            original_sentence = row.get('sentence', '').strip()
            if not original_sentence:
                print(f"Warning: Empty sentence in row {row_id}")
                continue
            class_label = self.extract_special_token(original_sentence)
            if not class_label:
                print(f"Warning: No special token found in row {row_id}: {original_sentence}")
                continue
            clean_sentence = self.remove_special_tokens(original_sentence)
            if not clean_sentence:
                print(f"Warning: Empty clean sentence in row {row_id}")
                continue
            audio_filename = self.generate_audio_filename(row_id, class_label)
            success, info = self.generate_audio(clean_sentence, audio_filename)
            if success:
                output_row = {
                    'audio_path': str(self.output_dir / audio_filename),
                    'original_sentence': original_sentence,
                    'clean_sentence': clean_sentence,
                    'class_label': class_label,
                    'info': info['info']
                }
                output_data.append(output_row)
            else:
                print(f"Failed to generate audio for row {row_id}")
        if output_data:
            try:
                with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['audio_path', 'original_sentence', 'clean_sentence', 'class_label', 'info']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(output_data)
                print(f"\nSuccess! Generated {len(output_data)} audio files")
                print(f"Output CSV saved to: {output_csv}")
                self.print_statistics(output_data)
            except Exception as e:
                print(f"Error writing output CSV: {e}")
        else:
            print("No audio files were generated successfully!")

    def print_statistics(self, data):
        print("\n=== Dataset Statistics ===")
        class_counts = {}
        for row in data:
            label = row['class_label']
            class_counts[label] = class_counts.get(label, 0) + 1
        print(f"Total samples: {len(data)}")
        print(f"Number of classes: {len(class_counts)}")
        print("\nClass distribution:")
        for label, count in sorted(class_counts.items()):
            print(f"  {label}: {count} samples")


def main():
    parser = argparse.ArgumentParser(description='Generate audio files from commands CSV')
    parser.add_argument('--input', '-i', default='commands.csv', 
                       help='Input CSV file (default: commands.csv)')
    parser.add_argument('--output', '-o', default='audio_dataset.csv',
                       help='Output CSV file (default: audio_dataset.csv)')
    parser.add_argument('--audio-dir', '-d', default='ben_branch/data/generated_audio',
                       help='Directory for audio files (default: ben_branch/data/generated_audio)')
    args = parser.parse_args()
    print("Initializing audio generator...")
    generator = AudioGenerator(
        output_dir=args.audio_dir,
    )
    start_time = time.time()
    generator.process_csv(args.input, args.output)
    end_time = time.time()
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

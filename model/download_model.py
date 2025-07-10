# whisper/scratch/download_model.py
"""Download and save Whisper-tiny under whisper_models/whisper_tiny."""

from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def main() -> None:
    # Resolve repo root: <project>/model
    repo_root = Path(__file__).resolve().parent.parent
    local_dir = repo_root / "whisper_models" / "whisper_tiny"
    local_dir.mkdir(parents=True, exist_ok=True)

    # Pull from HF hub
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

    # Save locally
    processor.save_pretrained(local_dir)
    model.save_pretrained(local_dir)

    print(f"✅ Model + processor saved to {local_dir}")

if __name__ == "__main__":
    main()

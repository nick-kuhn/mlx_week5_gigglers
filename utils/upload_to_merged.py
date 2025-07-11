
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from datasets import Audio
import argparse
from huggingface_hub import HfApi
import pandas as pd

def create_merged_dataset(repo_name):
    """Create a new repo from the local merged_dataset.csv."""
    print(f"Creating new repo '{repo_name}' from 'data/merged_dataset/merged_dataset.csv'")
    # 1. Load the merged dataset you saved on disk
    ds = load_dataset("csv", data_files="data/merged_dataset/merged_dataset.csv", split="train")

    # 2. Cast the audio column back to proper Audio objects if needed
    print("Casting audio column...")
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    # 3. Push to a fresh repo (create it on-the-fly)
    print(f"Uploading {len(ds)} rows to '{repo_name}'...")
    ds.push_to_hub(repo_name)
    print(f"✅ Successfully created '{repo_name}'")

def upload_from_csv(repo_name, csv_path):
    """Append the rows in `csv_path` to the remote repo `repo_name`."""
    if csv_path is None:
        raise ValueError("--add_from_csv requires a CSV path")

    # 1. Load new rows
    print(f"Loading new rows from '{csv_path}'...")
    new_rows = load_dataset("csv", data_files=csv_path, split="train")
    new_rows = new_rows.cast_column("audio", Audio(sampling_rate=16_000))
    print(f"Loaded {len(new_rows)} new rows.")

    # 2. Load existing remote dataset. If the repo is empty or doesn't exist yet,
    #    fallback to just the new rows.
    try:
        print(f"Loading existing dataset from '{repo_name}'...")
        existing = load_dataset(repo_name, split="train")
        print("Concatenating datasets...")
        combined = concatenate_datasets([existing, new_rows])
        commit_msg = f"Append {len(new_rows)} new rows from {csv_path}"
    except FileNotFoundError:
        print(f"Repository '{repo_name}' not found. Creating it from scratch.")
        combined = new_rows
        commit_msg = f"Initial upload from {csv_path}"

    # 3. Push combined dataset
    print(f"Uploading {len(combined)} total rows to '{repo_name}'...")
    combined.push_to_hub(repo_name, commit_message=commit_msg)
    print("✅ Upload complete!")

def upload_validation_split(repo_name, csv_path):
    """Adds or overwrites the 'validation' split in a remote repo."""
    if csv_path is None:
        raise ValueError("--add_validation_split requires a CSV path")

    # 1. Load the existing remote dataset to get the target schema
    try:
        print(f"Loading existing dataset from '{repo_name}' to get schema...")
        existing_ds_dict = load_dataset(repo_name)
        target_features = existing_ds_dict['train'].features
    except Exception as e:
        print(f"❌ Error loading existing dataset: {e}")
        print("Please ensure the repository exists and has a 'train' split.")
        return

    # 2. Load new validation rows from the CSV into a DataFrame
    print(f"Loading new validation rows from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} validation rows.")

    # 3. Transform the DataFrame to match the target schema
    print("Transforming new data to match target schema...")
    new_rows = []
    for _, row in df.iterrows():
        new_rows.append({
            'audio': row['filename'],  # This will be cast to Audio feature
            'class_label': row['command_token'],
            # Add other columns from the target schema with defaults
            'type': 'recording',
            'original_filename': row.get('filename'),
            'timestamp': row.get('timestamp'),
            'duration_seconds': row.get('duration_seconds'),
            'original_sentence': None,
            'clean_sentence': None,
            'info': None,
        })
    
    # 4. Create a new Dataset and cast it to the target schema
    validation_ds = Dataset.from_list(new_rows)
    print("Casting new validation split to the exact target features...")
    validation_ds = validation_ds.cast(target_features)

    # 5. Add/overwrite the validation split in the DatasetDict
    print(f"Adding/overwriting 'validation' split...")
    existing_ds_dict['validation'] = validation_ds
    
    # 6. Push updated DatasetDict
    commit_msg = f"Add/overwrite validation split with {len(validation_ds)} rows from {csv_path}"
    print(f"Uploading updated dataset to '{repo_name}'...")
    existing_ds_dict.push_to_hub(repo_name, commit_message=commit_msg)
    
    print("✅ Upload complete!")
    print("\nFinal splits:")
    for split_name, split_data in existing_ds_dict.items():
        print(f"  - {split_name}: {len(split_data)} samples")


if __name__ == "__main__":
    # Check if user is logged in to Huggingface
    try:
        api = HfApi()
        user = api.whoami()
        print(f"✅ Logged in to Hugging Face Hub as: {user['name']}")
    except Exception:
        print("❌ Could not connect to Hugging Face Hub.")
        print("Please login first by running: huggingface-cli login")
        exit()

    parser = argparse.ArgumentParser(
        description="Create a new merged voice-command dataset repo or append rows to an existing one on the Hugging Face Hub.")

    # The user must pick exactly ONE primary action
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--create_repo",
        metavar="REPO_ID",
        help="Create the specified repo and upload the local data/merged_dataset folder.",
    )
    group.add_argument(
        "--add_from_csv",
        metavar="CSV_PATH",
        help="CSV file with additional rows to append to an existing repo (default repo: ntkuhn/mlx_voice_commands_mixed).",
    )
    group.add_argument(
        "--add_validation_split",
        metavar="CSV_PATH",
        help="CSV file with validation rows to add/overwrite as the 'validation' split.",
    )

    # Optional – only used when appending rows
    parser.add_argument(
        "--upload_repo",
        metavar="REPO_ID",
        help="Target repository when using --add_from_csv (defaults to ntkuhn/mlx_voice_commands_mixed)",
        default="ntkuhn/mlx_voice_commands_mixed",
    )

    args = parser.parse_args()

    if args.create_repo:
        create_merged_dataset(args.create_repo)

    # The append path – only runs when --add_from_csv was supplied (mutually exclusive with --create_repo)
    if args.add_from_csv:
        upload_from_csv(args.upload_repo, args.add_from_csv)
    
    if args.add_validation_split:
        upload_validation_split(args.upload_repo, args.add_validation_split)
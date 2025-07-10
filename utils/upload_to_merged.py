
from datasets import load_dataset, concatenate_datasets
from datasets import Audio
import argparse

def create_merged_dataset(repo_name):
    # 1. Load the merged dataset you saved on disk
    ds = load_dataset("csv", data_files="data/merged_dataset/merged_dataset.csv", split="train")

    # 2. Cast the audio column back to proper Audio objects if needed

    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    # 3. Push to a fresh repo (create it on-the-fly)
    ds.push_to_hub(repo_name)

def upload_from_csv(repo_name, csv_path):
    """Append the rows in `csv_path` to the remote repo `repo_name`.

    Steps
    -----
    1. Load the additional rows from the given CSV.
    2. Load the existing remote dataset (non-streaming so we reuse shard files).
    3. Concatenate both datasets.
    4. Push the combined dataset back to the hub with a helpful commit message.
    """

    if csv_path is None:
        raise ValueError("--add_from_csv requires a CSV path")

    # 1. Load new rows
    new_rows = load_dataset("csv", data_files=csv_path, split="train")
    new_rows = new_rows.cast_column("audio", Audio(sampling_rate=16_000))

    # 2. Load existing remote dataset. If the repo is empty or doesn't exist yet,
    #    fallback to just the new rows.
    try:
        existing = load_dataset(repo_name, split="train")
        combined = concatenate_datasets([existing, new_rows])
        commit_msg = f"Append {len(new_rows)} new rows from {csv_path}"
    except FileNotFoundError:
        # Repo does not exist yet – create it with the new rows only
        combined = new_rows
        commit_msg = f"Initial upload from {csv_path}"

    # 3. Push combined dataset
    combined.push_to_hub(repo_name, commit_message=commit_msg)

if __name__ == "__main__":
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


#


# Features

## Generate Audio Dataset
Execute `uv run generate_audio.py` to generate audio files and a csv `audio_dataset.csv` from input file `commands.csv` (this may take a bit). 


# Environment Setup Guide

This guide shows you how to create and manage a Python 3.10 environment using **uv**, install dependencies from `pyproject.toml` in your project root, and integrate with Jupyter/VS Code.

---

## Prerequisites

- **uv** installed:
  ```bash
  curl -Ls https://astral.sh/uv/install.sh | bash
  export PATH="$HOME/.cargo/bin:$PATH"
  ```
- **Python 3.10** available as `python3.10`

---

## 1. Create & activate the environment

From the project root (`~/mlx/week5/mlx_week5_gigglers/`) run:

```bash
uv venv .venv --python=python3.10
source .venv/bin/activate  # prompt becomes (.venv)
```

---

## 2. Install project dependencies

1. In the project root, with the venv activated:
   ```bash
   # Compile dependencies into a lock file
   uv pip compile --output-file uv.lock pyproject.toml

   # Sync the environment to match the lock file
   uv pip sync uv.lock
   ```
2. Now the `.venv` has all required packages.

> **Tip**: Whenever you add new dependancies (see 5 below) to the `pyproject.toml`, rerun these commands to refresh `uv.lock` and reinstall.

---

## 3. Register the Jupyter kernel

Run this **once** to expose the env to JupyterLab or VS Code:

```bash
python -m ipykernel install --user   --name=week5_env   --display-name "Python (.venv)"
```

Select **"Python (.venv)"** in your notebook UI.

---

## 4. Day-to-day usage

```bash
# Activate the environment
source .venv/bin/activate

# Work as usual (run scripts, notebooks, FastAPI, etc.)

# Deactivate when done
deactivate
```

---

## 5. Adding new dependencies

Whenever you need to add a new package:

1. Add it under `[project].dependencies` in `pyproject.toml`.
2. Re-run:
   ```bash
   uv pip compile pyproject.toml --output-file uv.lock
   uv pip sync uv.lock
   ```

---

## 6. Git ignore

Keep the venv and lockfile out of Git by adding to `.gitignore`:

```
.venv/
uv.lock
```

The `pyproject.toml` stays committed so everyone sees the same dependency list.

---

Happy coding! 🚀

### To dp:
- cnn
- transformer
- cnn transformer
- ensemble

### Run train whisper
```
python ben_branch/whisper/scratch/train_whspr_classifier.py 
```

### TODO:
- fix audio gen
- fix class names! currently its making 12 classes and giving only 'volume' rather than 'volume_up' etc
- Look at other TODO in train_whspr_classifier.py


#


# Features

## Generate Audio Dataset
Execute `uv run generate_audio.py` to generate audio files and a csv `audio_dataset.csv` from input file `commands.csv` (this may take a bit). 

## Audio Recordings
Run `record.py` to generate more examples of voice commands by speaking into the microphone. 

## Upload to database
To push more entries to the database, run `utils/upload_to_merged.py` with option `--add_from_csv <csv_path>`. 

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
uv venv .venv --python=python3.10
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

# Workflow

### Generate then downsample the robot audio

```
python generate_audio.py
python misc/downsample.py
```
### Generate some real world audio from you as the human
```
audio_commands/record_linux.py # for linux
record.py # for mac (windows untested)
```
This will save metadata to recordings.csv and audio files to data. Note the linux vs mac saves to slightly different places, so be sure to check and potentially manually merge csvs and move files.

### Download the model
```
python model/download_model.py 

```
### Train classifier using whisper's encoder
```
python model/train_whspr_classifier.py 
```
This will run inference every epoch on 50% of the real world human data. For simplicity, at the moment it takes every other entry in metadata as validation, so the inverse entries can be sued for testing.

### Deploy 

There are two deployment options for the voice command system:

#### 1. Keyword-Based Deployment (`deployment/run_command.py`)
A traditional keyword-matching system that responds to specific spoken phrases:

**Available Commands:**
- "open browser" - Opens default browser
- "close browser" - Closes current browser window  
- "google" - Opens Google search
- "play music" - Starts music playback
- "stop music" - Stops music playback
- "volume up" - Increases system volume
- "volume down" - Decreases system volume
- "mute" - Toggles audio mute
- "maximize window" - Maximizes current window
- "switch window" - Alt+Tab to next window
- "open notepad" - Opens text editor

**Usage:**
```bash
python deployment/run_command.py
```
Hold CTRL to record, release to process. Uses exact keyword matching with speech-to-text.

#### 2. AI Classifier Deployment (`deployment/deploy_classifier.py`)
Uses the trained WhisperEncoderClassifier model for intelligent command classification:

**Available Command Tokens:**
- `<close_browser>` - Opens rickroll video (easter egg!)
- `<google>` - Opens Google search
- `<maximize_window>` - Maximizes current window
- `<mute>` - Toggles system audio mute
- `<no_action>` - Does nothing, pretty cool.
- `<open_browser>` - Opens web browser (auto locating to our fave person)
- `<open_notepad>` - Opens text editor
- `<play_music>` - Intelligently starts music playback
- `<stop_music>` - Stops music playback
- `<switch_window>` - Switches to next window
- `<volume_down>` - Decreases system volume
- `<volume_up>` - Increases system volume

**Features:**
- Uses custom trained classifier for robust command recognition
- Confidence threshold filtering (default: 0.4)
- Automatic audio backend detection (PortAudio/ffmpeg)
- Enhanced music playback with multiple player support
- Cross-platform compatibility (Linux/macOS/Windows)

**Usage:**
```bash
python deployment/deploy_classifier.py
```
Hold CTRL to record, release to classify and execute. Requires `best_model.pt` in deployment folder.

**Audio Backend Support:**
- **PortAudio** (preferred): Uses `sounddevice` for real-time audio capture
- **ffmpeg** (fallback): Uses ALSA on Linux for audio recording

Both systems provide real-time voice command execution with visual feedback and error handling. 


### TODO:
- fix audio gen
- fix class names! currently its making 12 classes and giving only 'volume' rather than 'volume_up' etc
- Look at other TODO in train_whspr_classifier.py
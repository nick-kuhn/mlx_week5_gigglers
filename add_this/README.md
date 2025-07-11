Steps:
1. Set up and activate virtual environment:
    python3 -m venv venv
    source venv/bin/activate
2. Install all the requirements/dependencies:
    pip install -r requirements.txt
3. Generate audio samples using the "record.py" file, make sure to add the file name and transcription in the "labels.csv" file
4. Then train Whisper tiny on these samples using "train_whisper.py"
5. Finally use "run_command2.py" to run the actual commands

Key:
***[...takes prompt...] = run_command2.py listens to words after command is spoken
***[...does not take prompt...] = run_command2.py does not care about what you say after command is spoken

Commands:
1. Open Spotify & Search a Song -> "Open Spotify and play [...takes prompt...]"
2. Open Calendar -> "Open Calendar [...does not take prompt...]"
3. Open Google and Search -> "Open Google and Search [...takes prompt...]"
4. Open Maps and Search -> "Open Maps and Search [...takes prompt (i.e. destination)...]"
5. Take a Screenshot -> "Take a Screenshot [...does not take prompt...]"
6. Unmute -> "Unmute [...does not take prompt...]"
7. Mute -> "Mute [...does not take prompt...]"
8. Lock my computer -> "Lock [...does not take prompt...]"
9. Chat GPT Search -> "Ask Chat GPT [...takes prompt...]"

To-do's:
- Increase the speed of recording
- Add open gmail and write an email saying... functionality
- Add open notes and write something... functionality
- Improve the terminal stuff

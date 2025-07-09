import torch
import whisper

model = whisper.load_model('tiny')
audio = whisper.load_audio('name.wav') # audio that contains the new word/token that you want whisper to learn/fine-tune
audio = whisper.pad_or_trim(audio)
lg_ml = whisper.log_mel_spectrogram(audio)
tknsr = whisper.tokenizer.get_tokenizer(multilingual=True)

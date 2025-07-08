import torch
import whisper
from pydub import AudioSegment

# Step 1: Convert .m4a to .wav (only needed ONCE, not every run)
audio_seg = AudioSegment.from_file("mlx.m4a", format="m4a")
audio_seg.export("mlx.wav", format="wav")

# Step 2: Use whisper.load_audio to load wav file as numpy array
audio = whisper.load_audio("mlx.wav")
audio = whisper.pad_or_trim(audio)
lg_ml = whisper.log_mel_spectrogram(audio)

# Load Whisper model and audio
model = whisper.load_model('tiny')
# audio = AudioSegment.from_file("basic.m4a", format="m4a")
# audio.export("name.wav", format="wav")  # Convert to WAV format for Whisper
# #audio = whisper.load_audio('name.wav')
# audio = whisper.pad_or_trim(audio)
# lg_ml = whisper.log_mel_spectrogram(audio)
tknsr = whisper.tokenizer.get_tokenizer(multilingual=True)

# Decode using Whisper (baseline)
opt = whisper.DecodingOptions()
res = whisper.decode(model, lg_ml.to(model.device), opt)
print('Baseline:', res.text)  # Example: "Hello my name is Bass."
print('------')

# Prepare tokens for custom training
ids = []
ids += [tknsr.sot]
ids += [tknsr.language_token]
ids += [tknsr.transcribe]
ids += [tknsr.no_timestamps]
ids += tknsr.encode(' I love MLX because it is teaching me a lot.')
ids += [tknsr.eot]

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
criterion = torch.nn.CrossEntropyLoss()

# Training step
model.train()
tks = torch.tensor(ids).unsqueeze(0).to(model.device)
mel = whisper.log_mel_spectrogram(audio).unsqueeze(0).to(model.device)

pred = model(tokens=tks, mel=mel)
trgt = tks[:, 1:].contiguous()
pred = pred[:, :-1, :].contiguous()

print('Ids Target:', trgt.squeeze().tolist())
print('Ids Output:', torch.argmax(pred, dim=-1).squeeze().tolist())
print('Txt Target:', tknsr.decode(trgt.squeeze().tolist()))
print('Txt Output:', tknsr.decode(torch.argmax(pred, dim=-1).squeeze().tolist()))

loss = criterion(pred.transpose(1, 2), trgt)
print('Loss:', loss.item())
print('------')
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Evaluation step
model.eval()
prd = model(tokens=tks, mel=mel)
prd = prd[:, :-1, :].contiguous()

print('Ids Target:', trgt.squeeze().tolist())
print('Ids Output:', torch.argmax(prd, dim=-1).squeeze().tolist())
print('Txt Target:', tknsr.decode(trgt.squeeze().tolist()))
print('Txt Output:', tknsr.decode(torch.argmax(prd, dim=-1).squeeze().tolist()))
loss = criterion(prd.transpose(1, 2), trgt)
print('Loss:', loss.item())

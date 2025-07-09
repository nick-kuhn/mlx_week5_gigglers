import torch
import whisper

model = whisper.load_model('tiny')
audio = whisper.load_audio('name.wav') # audio that contains the new word/token that you want whisper to learn/fine-tune
audio = whisper.pad_or_trim(audio)
lg_ml = whisper.log_mel_spectrogram(audio)
tknsr = whisper.tokenizer.get_tokenizer(multilingual=True)

opt = whisper.DecodingOptions()
res = whisper.decode(model, lg_ml.to(model.device),opt)
print("baseline:", res.text) # Hello, my name is Bass. 

ids = []
ids += [tknsr.sot]
ids += [tknsr.language_token]
ids += [tknsr.transcribe]
ids += [tknsr.no_timestamps]
ids += tknsr.encode("Hello, my name is Bes.")
ids += [tknsr.eot]

optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
criterion = torch.nn.CrossEntropyLoss()

model.train()
tks = torch.tensor(ids).unsqueeze(0).to(model.device)
mel = whisper.log_mel_spectrogram(audio).unsqueeze(0).to(model.device)

pred = model(tokens = tks, mel=mel)
trgt = tks[:,1:].contiguous()
pred = pred[:,:-1,:].continguous()

print("Ids Target:", trgt.squeeze().tolist())
print("Ids Output:", torch.argmax(pred, dim=-1).squeeze().tolist())
print("Txt Target:", tknsr.decode(trgt.squeeze().tolist()))
print("Txt Output:", tknsr.decode(torch.argmax(pred, dim=-1).squeeze().tolist()))

loss = criterion(pred.transpose(1,2), trgt)
print("Loss:", loss.item())
optimizer.zero_grad()
loss.backward()
optimizer.step()

model.eval()
prd = model(tokens = tks, mel = mel)
prd = prd[:,:-1,:].contiguous()

print("Ids Target:", trgt.squeeze().tolist())
print("Ids Output:", torch.argmax(pred, dim=-1).squeeze().tolist())
print("Txt Target:", tknsr.decode(trgt.squeeze().tolist()))
print("Txt Output:", tknsr.decode(torch.argmax(pred, dim=-1).squeeze().tolist()))

loss = criterion(prd.transpose(1,2), trgt)
print("Loss:", loss.item())
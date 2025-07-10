import torch
import whisper
import pandas as pd
from pydub import AudioSegment

# Parameters
audio_folder = "data"
metadata_path = "labels.csv"
learning_rate = 1e-5
epochs = 3

# Load dataset metadata
df = pd.read_csv(metadata_path)

# Load model and tokenizer
model = whisper.load_model("tiny")   # you can use "base", "tiny", etc.
tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(epochs):
    total_loss = 0
    for i, row in df.iterrows():
        # 1. Load and preprocess audio
        audio_file = f"{audio_folder}/{row['filename']}"
        audio = whisper.load_audio(audio_file)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).unsqueeze(0).to(model.device)

        # 2. Prepare tokens
        text = row['transcription']
        ids = []
        ids += [tokenizer.sot]
        ids += [tokenizer.language_token]
        ids += [tokenizer.transcribe]
        ids += [tokenizer.no_timestamps]
        ids += tokenizer.encode(" " + text.strip())
        ids += [tokenizer.eot]
        tokens = torch.tensor(ids).unsqueeze(0).to(model.device)

        # 3. Forward pass
        pred = model(tokens=tokens, mel=mel)
        target = tokens[:, 1:].contiguous()
        pred = pred[:, :-1, :].contiguous()
        
        # --- Decode model prediction (output) for inspection ---
        out_ids = torch.argmax(pred, dim=-1).squeeze().tolist()
        tgt_ids = target.squeeze().tolist()
        decoded_output = tokenizer.decode(out_ids)
        decoded_target = tokenizer.decode(tgt_ids)
        
        # --- Print or log ---
        print(f"File: {row['filename']}")
        print(f"Target transcript:     {decoded_target}")
        print(f"Predicted transcript: {decoded_output}")
        print(f"Loss: {criterion(pred.transpose(1, 2), target).item():.4f}")
        print('-'*40)

        # 4. Loss & backward
        loss = criterion(pred.transpose(1, 2), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Progress print
        if (i+1) % 10 == 0 or (i+1) == len(df):
            print(f"Epoch {epoch+1} Example {i+1}/{len(df)}: Loss={loss.item():.4f}")

    print(f"Epoch {epoch+1}: Avg loss = {total_loss/len(df):.4f}")

# Save your model if you want
torch.save(model.state_dict(), "whisper_ask_ft.pth")
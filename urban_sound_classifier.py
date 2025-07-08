### try to classify sounds into 10 classes
# using both the CNN and transformers
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import numpy as np
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics import f1_score

# pre-set all the relevant parameters for later sweep
@dataclass
class TrainingHyperparameters:
    num_epochs = 10
    batch_size = 128
    lr = 3e-4
    weight_decay = 1e-4
    
@dataclass
class ModelHyperparameters:
    patch_size = 8  # for patchigy: should divide both n_mels (128) and n_frames (128)
    embed_dim = 64
    ff_dim = 2048
    num_heads = 8
    num_layers = 3 # number of the encoder blocks


### 1. load in the dataset
ds = load_dataset("danavery/urbansound8K") #no need to shuffle the dataset as indicated on the huggdsingface, 10-fold cross-validation by default

### 2. pre-processing the dataset; get corresponding power spectrum --> get the images of sound
def audio_to_spectrum(
    audio_array: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
    target_duration: float = 4.064
) -> np.ndarray:
    # ensure float32
    audio_array = audio_array.astype(np.float32)
    # first downsample the audio
    if orig_sr != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)
    # zero-padding or truncate to a fixed length of 4s
    target_len = int(target_sr * target_duration)
    if len(audio_array) < target_len:
        audio_array = np.pad(audio_array, (0, target_len - len(audio_array)), mode='constant')
    else:
        audio_array = audio_array[:target_len]
    # then run the fft
    mel_spec = librosa.feature.melspectrogram(
        y=audio_array, 
        sr=target_sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    # convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    # normalize the powe spectrum for each audio file --- z-score
    log_mel_spec = (log_mel_spec - np.mean(log_mel_spec))/(np.std(log_mel_spec) + 1e-6)
    return log_mel_spec

# Custom Dataset for spectrograms and labels
class SpectrogramDataset(Dataset):
    def __init__(self, spectra: List[np.ndarray], labels: List[int], class_names: Optional[List[str]] = None) -> None:
        self.spectra = spectra
        self.labels = labels
        self.class_names = class_names
    def __len__(self) -> int:
        return len(self.spectra)
    def __getitem__(self, idx: int) -> Any:
        spec = np.expand_dims(self.spectra[idx], axis=0)  # [1, n_mels, n_frames]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.class_names is not None:
            class_name = self.class_names[idx]
            return torch.tensor(spec, dtype=torch.float32), label, class_name
        return torch.tensor(spec, dtype=torch.float32), label
# Example: To get the class name for a sample, pass class_names to the dataset and unpack the third value.
# spec, label, class_name = dataset[idx]

# Preprocess and store all power spectra
power_spectra: List[np.ndarray] = []
labels: List[int] = []
folds: List[int] = []
class_names: List[str] = []  # Store the class name for each example
train_data = list(ds['train'])
for example in train_data:
    audio = example['audio']['array']
    sr = example['audio']['sampling_rate']
    spectrum = audio_to_spectrum(audio, sr)
    power_spectra.append(spectrum)
    labels.append(example['classID'])   # Use integer label for training
    folds.append(example['fold'])
    class_names.append(example['class'])  # Store the class name for reference


# ### have a peek of a random power spectrogram
# i = 0
# plt.figure(figsize=(10,4))
# plt.imshow(power_spectra[i], aspect='auto', origin='lower', cmap='hot')
# plt.title(f'Log-Mel Spectrogram -- class: {labels[i]}')
# plt.xlabel('Time Frames')
# plt.ylabel('Mel Bands')
# plt.tight_layout()
# plt.show()



### 3. using these sound images as input for the classifers
# 3.1 transformer encoder
# Define the neural network architecture
class ViT(nn.Module):
    def __init__(self,img_width,img_channels,patch_size,embed_dim,num_heads,num_layers,num_classes,ff_dim):
        super().__init__() #call the parent class's __init__
        # carry some parameters
        self.patch_size = patch_size
        # get the embedding layer
        self.patch_embedding = nn.Linear(img_channels*patch_size*patch_size,embed_dim) #will do the broadcast to the data tensor, so the input and output dim don't need to be matched
        # get the CLS which "summerize" the information of the whole sequence
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        # get the positional encoding [including the patch_embeddings plus 1--the cls token]
        self.position_embedding = nn.Parameter(
            torch.randn(1,(img_width//patch_size)*(img_width//patch_size)+1,embed_dim)
                                               )
        # build the encoder, first define the encoder_layer, then just stack them together num_layers times
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, 
        )
        # construct the last layer for the classification task
        self.fc = nn.Linear(embed_dim, num_classes)
    # construct the forward model, how information will be passed through --- the real structure
    def forward(self, x):
        # flatten the patch matrix into a vector, also stack together all patches
        b, c, nh, nw, ph, pw = x.shape  #[64,1,4,4,7,7]
        # stack the patches together
        x = x.reshape(b, nh*nw, ph, pw) #[64,1,16,7,7]
        # flatte each patch into one vector
        x = x.reshape(b, nh*nw, ph*pw)  #[64, 1, 16, 49]
        # each flatten patch will be embedded to the embed_dim
        x = self.patch_embedding(x)
        # pre-pend the CLS ("secretory") token in front of the embedding
        cls_tokens = self.cls_token.repeat(b,1,1) #CLS token for the whole batch
        x = torch.cat((cls_tokens, x), dim=1)
        # add the position embeddings, which are learnable parameters
        x = x + self.position_embedding
        # go into the transformer attention blocks
        x = self.transformer_encoder(x)
        # only select the hidden vector of the CLS token for making the prediction
        x = x[:,0]
        # go throught the finnal fc layer for the classification task
        x = self.fc(x)
        return x


def patchify(batch_data, patch_size):
    """
    patchify the batch of images
    """
    b,c,h,w = batch_data.shape  #[batch_size,channels,height,width] 
    ph = patch_size
    pw = patch_size
    nh, nw = h//ph, w//pw
    batch_patches = torch.reshape(batch_data, (b,c,nh,ph,nw,pw))
    batch_patches = torch.permute(batch_patches,(0,1,2,4,3,5)) #[64,1,4,4,7,7]
    # flatten the pixels in each patch
    return batch_patches


def train_model(model, train_loader, criterion, optimizer, device, patch_size):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(train_loader):
        data, target = batch[0], batch[1]
        data, target = data.to(device), target.to(device)
        # data.shape = [64, 1, 28, 28] #tensor: [batch_size,channels,height,width]
        # target is a tensor of shape [64]
        # ### try to plot a random image to have a peek
        # img = data[0].cpu().squeeze()
        # plt.imshow(img, cmap='gray')
        # plt.show()
        # patchify the image into a grid (in order to implement vision transformer)
        data = patchify(data,patch_size) #[64,1,4,4,7,7]
        # ### try to check on the pathify
        # img_patches = patches[0,0]
        # fig,axes = plt.subplots(4,4,figsize=(7,7))
        # for i in range (4):
        #     for j in range(4):
        #         axes[i,j].imshow(img_patches[i,j],cmap='gray')
        #         axes[i,j].axis('off')
        # plt.show()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 99:
            print(f'Batch: {batch_idx + 1}, Loss: {running_loss/100:.3f}, '
                  f'Accuracy: {100.*correct/total:.2f}%')
            running_loss = 0.0


def evaluate_model(model, test_loader, device, patch_size):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch[0], batch[1]
            data, target = data.to(device), target.to(device)
            data = patchify(data, patch_size)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    accuracy = 100. * correct / total
    f1 = f1_score(all_targets, all_preds, average='macro')  # or 'weighted'
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Test F1 Score (macro): {f1:.4f}')
    return accuracy, f1

def main():
    img_width = power_spectra[0].shape[-1]
    img_channels = 1 # only 1 channel not RGB 3 channels
    num_classes = 10
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    n_folds = 10
    all_indices = np.arange(len(power_spectra))
    fold_ids = np.array(folds)
    for val_fold in range(1, n_folds + 1):
        print(f'\n=== Fold {val_fold} as validation ===')
        train_indices = all_indices[fold_ids != val_fold]
        val_indices = all_indices[fold_ids == val_fold]
        train_dataset = SpectrogramDataset([power_spectra[i] for i in train_indices], [labels[i] for i in train_indices], class_names=[class_names[i] for i in train_indices])
        val_dataset = SpectrogramDataset([power_spectra[i] for i in val_indices], [labels[i] for i in val_indices], class_names=[class_names[i] for i in val_indices])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        # Initialize model, loss function, and optimizer from scratch for each fold
        model = ViT(
            img_width=img_width,
            img_channels=img_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_classes=num_classes,
            num_layers=num_layers,
            ff_dim=ff_dim,
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Training loop
        print('Starting training...')
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            train_model(model, train_loader, criterion, optimizer, device, patch_size)
            accuracy, f1 = evaluate_model(model, val_loader, device, patch_size)
    # Save the last trained model (optional)
    torch.save(model.state_dict(), 'sound_classifier_transformer.pth')
    print('Model saved to sound_classifier_transformer.pth')

if __name__ == '__main__':
    main() 
# take input: x: [B, 1, n_mels, frames]
# patch the input
# add pos embeds
# pass through norm or something?
# pass to self attention block
# norm?
# mlp?
# classification

import torch
import torch.nn as nn

def patch_img(imgs, patch_size, stride):
    """Turn:
    [[ 1,  2,  3,  4],
    [ 5,  6,  7,  8],
    [ 9, 10, 11, 12],
    [13, 14, 15, 16]]
    into:
    [[1,2,5,6],
    [3,4,7,8],
    [9,10,13,14],
    [11,12,15,16]]
    img = a torch tensor
    """
    batch_columns = []
    for img in imgs:
        # With a 4x4 grid example
        # First unfold does by row, patch every second/path_size num into an array
        # Second unfold does by column.
        # Is now 4 arrays (1 per patch) of 2x2 arrays (individual patches)
        patches = img.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
        # Flatten each patch
        columns = patches.contiguous().view(-1, patch_size * patch_size)
        batch_columns.append(columns)
    return torch.stack(batch_columns)
        
NUM_HEADS = 8
EMBED_DIM = 128

class Tranformer(nn.Module):
    def __init__(self, x, embed_dim, patch_size, stride, num_blocks):
        super().__init__()
        self.patch = patch_img()
        self.pos_enc = nn.Embedding(17, embed_dim)
        self.mh_att = MultiHeadAttention()
        self.linear = nn.Linear() 

    def forward(self, x, patch_size, stride, num_blocks):
        x = self.patch(x, patch_size, stride) # will it magically handle shapes (including batch?)
        x = self.pos_enc(x)
        for _ in num_blocks:
            x = x.MultiHeadAttention(x)

        # Now take the output from the last block and do:    
        x = self.linear(x, 10) #project to 10 classes
        # Apply softmax here or will CES loss do this automatically?
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.proj = nn.Linear()
        self.self_att = SelfAttention()
        self.norm = nn.LayerNorm()
        self.ffn = FeedForward()

    def forward(self, x, num_heads, embed_dim):
        
        for _ in num_heads:
            x = self.proj(x, num_heads/embed_dim)
            x_att = self.self_att(x)
        x = self.norm(x + x_att)
        x = self.ffn(x)


            # ffn
            # resid
            # norm
            # concat back to embed_dim
        return 




class SelfAttention(nn.Module):
    def __init__(self, x):
        super().__init__()

class FeedForward(nn.module):
    def __init__(self):
        super().__init__
        self


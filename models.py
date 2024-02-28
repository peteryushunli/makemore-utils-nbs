import sys
sys.path.append('/Users/shwetank/code/makemore-utils-nbs')
from utils import create_dataset, CharDataset, evaluate_loss, print_samples
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import torch

class Bigram(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.bigram_embedding = nn.Embedding(vocab_size, vocab_size) # Because stop word will never be an input

    def forward(self, x, targets = None):
        logits = self.bigram_embedding(x) # Outputs Batch, Time, Channel (Vocab Size) 
        if targets == None:
            loss = None
        else:
            self.B,self.T,self.C = logits.shape
            logits = logits.view(self.B*self.T,self.C)
            targets = targets.contiguous().view(self.B*self.T)
            loss = F.cross_entropy(logits, targets, ignore_index=-1)
            
        return logits, loss
    
class MLP(nn.Module):
    """
    takes the previous block_size tokens, encodes them with a lookup table,
    concatenates the vectors and predicts the next token with an MLP.

    Reference:
    Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
    """

    def __init__(self, max_length, vocab_size, embedding_dimension, hidden_dimension):
        super().__init__()
        self.block_size = max_length + 1
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size + 1,  embedding_dimension) # token embeddings table
        # +1 in the line above for a special <BLANK> token that gets inserted if encoding a token
        # before the beginning of the input sequence
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * embedding_dimension, hidden_dimension),
            nn.Tanh(),
            nn.Linear(hidden_dimension, self.vocab_size)
        )

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):

        # print('idx0:', idx)
        # print('targets:', targets)
        # gather the word embeddings of the previous 3 words - No actually this uses entire sequence
        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size # special <BLANK> token
            embs.append(tok_emb)
            # print(f'idx{k}:', idx)

        # print('block_size:', self.block_size)
        # print('k:', k)
        # print('idx:', idx)
        # print(embs)
        # sys.exit(1)

        # concat all of the embeddings together and pass through an MLP
        x = torch.cat(embs, -1) # x is (b, t, n_embd * block_size) 
        logits = self.mlp(x) # logits are (b, t, vocab_size)
        # print('logits shape:', logits.shape)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

#----------Transformer using Pytorch attention block----------
    
class  MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, emb_dim, dropout):
        super().__init__()
        self.heads = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, bias=False)

    def forward(self, x):
        x = self.heads(x,x,x)
        return x
    
class Feedforward(nn.Module):
    def __init__(self, emb_dim, dropout):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, 4*emb_dim),
            nn.ReLU(),
            nn.Linear(4*emb_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.ff(x)
    
class Block(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout):
        super().__init__()
        self.head_size = emb_dim // num_heads
        self.sa_head = MultiHeadAttention(num_heads, emb_dim, dropout)
        self.ff = Feedforward(emb_dim, dropout)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x, targets=None):
        sa_out, sq_wte = self.sa_head(self.ln1(x))
        x = x + sa_out
        x = x + self.ff(self.ln2(x))
        return x
    
class Pyt_Attention_Xformer(nn.Module):
    def __init__(self, emb_dim,  vocab_size, num_heads, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size + 1, emb_dim)
        self.pos_embedding = nn.Embedding(vocab_size + 1, emb_dim)
        # self.sa_head = MultiHeadAttention(emb_dim//4, 4, emb_dim, block_length)
        # self.ff = Feedforward(emb_dim)
        self.blocks = nn.Sequential(
            Block(emb_dim, num_heads, dropout), 
            Block(emb_dim, num_heads, dropout),
            Block(emb_dim, num_heads, dropout),
            nn.LayerNorm(emb_dim)
        )
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, targets=None):
        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(x)
        x = tok_emb + pos_emb # B, T, emb_dim
        # x = self.sa_head(x) # B, T, head_size
        # x =  self.ff(x)
        x = self.blocks(x)
        logits = self.lm_head(x) # B, T, vocab_size

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # print(logits.view(-1, logits.size(-1)).shape, targets.view(-1).shape)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return(logits,loss)
    
class Head(nn.Module):
    def __init__(self, emb_dim, head_size, block_length, dropout):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        # Define a register buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_length,block_length)))
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        ## Initialize the vector
        k = self.key(x) # B,T,head_size
        q = self.query(x) # B,T,head_size
        v = self.value(x) # B,T,head_size
        wei = k @ torch.transpose(q, 1, 2) * C**-0.5 # B,T,head_size * B,head_size,T -> B,T,T
        wei = torch.masked_fill(wei, self.tril[:T,:T] == 0, float('-Inf')) # Only selecting till the Tth column will be esp important during generation o/w will expect maxx_length columns at everytime step and throw an error
        wei = self.dropout(F.softmax(wei, dim=-1)) # B,T,T
        # print(k.shape, q.shape, v.shape, wei.shape)
        out = wei @ v #B,T,T * B,T,H -> B,T,H i.e. 32,16,16 * 32,16,8 -> 32,16,8
        return out
    
class MultiHeadAttentionModuleList(nn.Module):
    def __init__(self, head_size, num_heads, emb_dim, block_length, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(emb_dim, head_size, block_length, dropout) for i in range(num_heads)]) # B,T,head_size*num_heads
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x
    
class BlockScratch(nn.Module):
    def __init__(self, emb_dim, num_heads, block_length, dropout):
        super().__init__()
        self.head_size = emb_dim // num_heads
        self.sa_head = MultiHeadAttentionModuleList(self.head_size, num_heads, emb_dim, block_length,dropout)
        self.ff = Feedforward(emb_dim, dropout)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x, targets=None):
        sa_out = self.sa_head(self.ln1(x))
        x = x + sa_out
        x = x + self.ff(self.ln2(x))
        return x
    
    
class Xformer_Scratch(nn.Module):
    def __init__(self, emb_dim, vocab_size, num_heads, block_length, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size + 1, emb_dim)
        self.pos_embedding = nn.Embedding(vocab_size + 1, emb_dim)
        # self.sa_head = MultiHeadAttention(emb_dim//4, 4, emb_dim, block_length)
        # self.ff = Feedforward(emb_dim)
        self.blocks = nn.Sequential(
            BlockScratch(emb_dim, num_heads, block_length, dropout), 
            BlockScratch(emb_dim, num_heads, block_length, dropout),
            BlockScratch(emb_dim, num_heads, block_length, dropout),
            nn.LayerNorm(emb_dim)
        )
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, targets=None):
        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(x)
        x = tok_emb + pos_emb # B, T, emb_dim
        # x = self.sa_head(x) # B, T, head_size
        # x =  self.ff(x)
        x = self.blocks(x)
        logits = self.lm_head(x) # B, T, vocab_size

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            # print(logits.view(-1, logits.size(-1)).shape, targets.view(-1).shape)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return(logits,loss)
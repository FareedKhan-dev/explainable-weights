"""wsp/model.py: the paper's starting network, a small causal transformer whose every weight can be set by hand.

Three blocks of attention plus a two-layer MLP, each with a residual connection and a LayerNorm in front, then a
linear head over the task's characters. This is the template every agent run starts from. The agent may change the
shapes and the wiring, but never trains a weight."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.W_q, self.W_k, self.W_v, self.W_o = (nn.Linear(d_model, d_model, bias=False) for _ in range(4))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = (w(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2) for w in (self.W_q, self.W_k, self.W_v))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)          # how well each query matches each key
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        attn = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)       # a token only sees itself and the past
        return self.W_o((attn @ v).transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.fc1, self.fc2 = nn.Linear(d_model, d_ff), nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))                      # read from other positions
        return x + self.fc2(F.relu(self.fc1(self.ln2(x))))  # compute on what was read


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, max_seq_len=32, d_model=64, n_heads=4, n_layers=3, d_ff=256):
        super().__init__()
        self.token_emb, self.pos_emb = nn.Embedding(vocab_size, d_model), nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_ff) for _ in range(n_layers))
        self.final_ln, self.head = nn.LayerNorm(d_model), nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        h = self.token_emb(ids) + self.pos_emb(torch.arange(ids.shape[1], device=ids.device))[None]
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_ln(h))                  # one score per character, at every position

"""
Vanilla Transformer implementation in almost 100 lines of code.
"""
import torch
import torch.nn as nn

class AttentionHead(nn.Module):
    def __init__(self, n_embd, n_headd):
        super().__init__()
        self.qkv = nn.Linear(n_embd, n_headd * 3)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(k.size(-1))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        return torch.bmm(torch.softmax(attn_scores, axis=-1), v)

    def forward(self, x):
        return self.scaled_dot_product_attention(self.qkv(x).chunk(3, dim=-1))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.heads = nn.ModuleList([AttentionHead(n_embd, n_embd // n_head) for _ in range(n_head)])
        self.output_linear = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        return self.output_linear(torch.cat([head(x) for head in self.heads], dim=-1))
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, n_embd, ff_dim):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(n_embd, ff_dim), nn.GELU(), nn.Linear(ff_dim, n_embd), nn.Dropout(0.1))

    def forward(self, x):
        return self.ff(x)

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, n_embd):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_seq_len, n_embd)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(x + self.position_embeddings(torch.arange(x.size(1), device=x.device)))
    
class EncoderLayer(nn.Module):
    def __init__(self, n_embd, n_head, ff_dim):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.multihead_attn = MultiHeadAttention(n_embd, n_head)
        self.layer_norm_2 = nn.LayerNorm(n_embd)
        self.feed_forward = PositionWiseFeedForward(n_embd, ff_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        attn_outputs = self.multihead_attn(self.layer_norm_1(x))
        x = x + self.dropout(attn_outputs)
        return x + self.dropout(self.feed_forward(self.layer_norm_2(x)))
    
class Encoder(nn.Module):
    def __init__(self, n_embd, n_head, ff_dim, n_layer=6):
        super().__init__()
        self.positional_encoding = LearnedPositionalEncoding(100, n_embd)
        self.layers = nn.ModuleList([EncoderLayer(n_embd, n_head, ff_dim) for _ in range(n_layer)])

    def forward(self, x):
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, n_embd, n_head, ff_dim):
        super().__init__()
        self.norm_1 = nn.LayerNorm(n_embd)
        self.masked_attn = MultiHeadAttention(n_embd, n_head)
        self.norm_2 = nn.LayerNorm(n_embd)
        self.cross_attn = MultiHeadAttention(n_embd, n_head)
        self.norm_3 = nn.LayerNorm(n_embd)
        self.feed_forward = PositionWiseFeedForward(n_embd, ff_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, enc_output, mask=None):
        x = self.norm_1(x + self.dropout(self.masked_attn(x, mask)))
        x = self.norm_2(x + self.dropout(self.cross_attn(x, enc_output)))
        return self.norm_3(x + self.dropout(self.feed_forward(x)))

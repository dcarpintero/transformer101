## Transformer101

Vanilla implementation in Pytorch of the Transformer model as introduced in the paper [Attention Is All You Need, 2017](https://arxiv.org/pdf/1706.03762.pdf) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin.

``Scaled Dot-Product Attention`` | ``Multi-Head Attention`` | ``Absolute Positional Encodings`` | ``Learned Positional Encodings`` | ``Dropout`` | ``Layer Normalization`` | ``Residual Connection`` | ``Linear Layer`` | ``Position-Wise Feed-Forward Layer`` | ``GELU`` | ``Softmax`` | ``Encoder`` | ``Decorder`` | ``Transformer``

### 1. Background

Sequence modeling and transduction tasks, such as language modeling and machine translation, were typically addressed with RNNs and CNNs. However, these architectures are limited by: (i) ``long training times``, due to the sequential nature of RNNs, which constrains parallelization, and results in increased memory and computational demands as the text sequence grows; and (ii) ``difficulty in learning dependencies between distant positions``, where CNNs, although much less sequential than RNNs, require a number of steps to integrate information that is, in most cases, correlated (linearly for models like ConvS2S and logarithmically for ByteNet) with the distance between elements in the sequence.

### 2. Technical Approach

The paper 'Attention is All You Need' introduced the novel Transformer model, ``a stacked encoder-decoder architecture that utilizes self-attention mechanisms instead of recurrence and convolution to compute input and output representations``. In this model, each of the six layers of both the encoder and decoder is composed of two main sub-layers: a multihead self-attention sub-layer, which allows the model to focus on different parts of the input sequence, and a position-wise fully connected feed-forward sub-layer.

At its core, the ``self-attention mechanism`` enables the model to weight the relationships between input tokens at different positions, resulting in a more effective handling of long-range dependencies. Additionally, by integrating ``multiple attention heads``, the model gains the ability to simultaneously attend to various aspects of the input data during training.

In the proposed implementation, the input and output tokens are converted to 512-dimensional embeddings, to which ``positional embeddings`` are added, enabling the model to use sequence order information.

### 3. Transformer Model Implementation


```python
import torch
import torch.nn as nn
```


```python
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    n_layer: int = 6        # number of encoder/decoder layers
    n_head: int = 12        # number of attention heads
    d_embd: int = 768       # dimension of the token embeddings
    d_ff: int = 4096        # dimension of the feedforward network
    drop: float = 0.1       # dropout probability
    max_seq_len: int = 512  # maximum sequence length
    pad_token_id: int = 0   # padding token id (usually 0)
```


```python
config = TransformerConfig()
```

#### 3.1 Self-Attention

The intuition behind ``self-attention`` is that averaging token embeddings instead of using a fixed embedding for each token, enables the model to capture how words relate to each other in the input. In practice, said weighted relationships (attention weights) represent the syntactic and contextual structure of the sentence, leading to a more nuanced and rich understanding of the data.

The most common way to implement a self-attention layer relies on ``scaled dot-product attention``, and involves:
1. ``Linear projection`` of each token embedding into three vectors: ``query (q)``, ``key (k)``, ``value (v)``.
2. Compute ``scaled attention scores``: determine the similary between ``q`` and ``k`` by applying the ``dot product``. Since the results of this function are typically large numbers, they are then divided by a scaling factor inferred from the dimensionality of (k). This scaling contributes to stabilize gradients during training.
3. Normalize the ``attention scores`` into ``attention weights`` by applying the ``softmax`` function (this ensures all the values sum to 1).
4. ``Update the token embeddings`` by multiplying the ``attention weights`` by the ``value vector``.

> In addition, the self-attention mechanism of the decoder layer introduces ``masking`` to prevent the decoder from having access to future tokens in the sequence it is generating. In practice, this is implemented with a binary mask that designates which tokens should be attended to (assigned non-zero weights) and which should be ignored (assigned zero weights). In our function, setting the future tokens (upper values) to negative-infinity guarantees that the attention weights become zero after applying the softmax function (e exp -inf == 0). This design aligns with the nature of many tasks like translation, summarization, or text generation, where the output sequence needs to be generated one element at a time, and the prediction of each element should be based only on the previously generated elements.


```python
class AttentionHead(nn.Module):
    """
    Represents a single attention head within a multi-head attention mechanism.
    
    Args:
        config (TransformerConfig): The configuration for the transformer model.
    """
    def __init__(self, config):
        super().__init__()
        self.d_head = config.d_embd // config.n_head
        # step_1: linear projections to query (q), key (k), and value (v) vectors
        self.q = nn.Linear(config.d_embd, self.d_head)
        self.k = nn.Linear(config.d_embd, self.d_head)
        self.v = nn.Linear(config.d_embd, self.d_head)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dim_k = torch.tensor(k.size(-1), dtype=torch.float32)
        # step_2: calculate (q, k) similarity with the dot product, and scale attention scores
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(dim_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        # step_3: normalize the attention scores with the softmax function
        attn_weights = torch.softmax(attn_scores, axis=-1)
        # step_4: update the token embeddings by multiplying attention weights by the value vector
        output = torch.bmm(attn_weights, v)
        return output

    def forward(self, x, mask=None):
        output = self.scaled_dot_product_attention(self.q(x), 
                                                   self.k(x), 
                                                   self.v(x), 
                                                   mask=mask)
        return output
```

#### 3.2 Multi-Headed Attention

In a standard attention mechanism, the ``softmax`` of a single head tends to concentrate on a specific aspect of similarity, potentially overlooking other relevant features in the input. By integrating multiple attention heads, the model gains the ability to simultaneously attend to various aspects of the input data.

The basic approach to implement Multi-Headed Attention comprises:

1. Initialize the ``attention heads``. E.g. BERT has 12 attention heads whereas the embeddings dimension is 768, resulting in 768 / 12 = 64 as the head dimension.
2. ``Concatenate attention heads`` to combines the outputs of the attention heads into a single vector while preserving the dimensionality of the embeddings.
3. Apply a ``linear projection``.

> Note that the softmax function is a probability distribution, which when applied within a single attention head tends to amplify certain features (those with higher scores) while diminishing others. Thus, leading to a focus on specific aspects of similarity.


```python
class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism.

    Args:
        config (TransformerConfig): The configuration for the transformer model.
    """
    def __init__(self, config):
        super().__init__()

        if config.d_embd < 0 or config.n_head < 0:
            raise ValueError("Embedding dimension and number of heads must be greater than 0")
        assert config.d_embd % config.n_head == 0, "d_embd must be divisible by n_head"

        self.heads = nn.ModuleList([AttentionHead(config) for _ in range(config.n_head)])
        self.linear = nn.Linear(config.d_embd, config.d_embd)

    def forward(self, x, mask=None):
        attn_outputs = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        output = self.linear(attn_outputs)
        return output
```


```python
multihead_attn = MultiHeadAttention(config)
attn_output = multihead_attn(torch.rand(1, 10, 768))
attn_output.size()
```




    torch.Size([1, 10, 768])



#### 3.3 Position-Wise Feed-Forward Layer

The Transformer, primarily built upon linear operations like ``dot products`` and ``linear projections``, relies on the ``Position-Wise Feed-Forward Layer`` to introduce non-linearity into the model. This non-linearity enables the model to capture complex data patterns and relationships. The layer typically consists of two linear transformations with a ``non-linear activation function (like ReLU or GELU)``. Each layer in the ``Encoder`` and ``Decoder`` includes one of these feed-forward networks, allowing the model to build increasingly abstract representations of the input data as it passes through successive layers. Note that since this layer processes each embedding independly, the computations can be fully parallelized.

In summary, this Layer comprises:
- ``First linear transformation`` to the input tensor. 
- A non-linear ``activation function`` to allow the model learn more complex patterns.
- ``Second linear transformation``, increasing the model's capacity to learn complex relationships in the data.
- ``Dropout``, a regularization technique used to prevent overfitting. It randomly zeroes some of the elements of the input tensor with a certain probability during training.

> Note that the ``ReLU`` function is a faster function that activates units only when the input is possitive, which can lead to sparse activations (that can be intended in some tasks); whereas ``GELU``, introduced after``ReLU``, offers smoother activation by modeling the input as a stochastic process, providing a probabilistic gate in the activation. In practice, ``GELU`` has been the preferred choice in the BERT and GPT models.


```python
class PositionWiseFeedForward(nn.Module):
    """
    Implements the PositionWiseFeedForward layer.

    Args:
        config (TransformerConfig): The configuration for the transformer model.
    """
    def __init__(self, config):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(config.d_embd, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_embd),
            nn.Dropout(config.drop)
        )

    def forward(self, x):
        return self.ff(x)
```


```python
feed_forward = PositionWiseFeedForward(config)
feed_forward_outputs = feed_forward(torch.rand(1, 10, 768))
feed_forward_outputs.size()
```




    torch.Size([1, 10, 768])



#### 3.4 Positional Encoding

Since the Transformer model contains no recurrence and no convolution, the model is invariant to the position of the tokens. By adding ``positional encoding`` to the input sequence, the Transformer model can differentiate between tokens based on their position in the sequence, which is important for tasks such as language modeling and machine translation. In practice, ``positional encodings`` are added to the input embeddings at the bottoms of the ``encoder`` and ``decoder`` stacks. 

> As outlined in the original [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) paper, ``Sinusoidal Positional Encoding`` and ``Learned Positional Encoding`` produce nearly identical results.




```python
class SinusoidalPositionalEncoding(nn.Module):
    """
    Implements Sinusoidal Positional Encoding.

    Parameters:
        embed_size (int): The size of the input feature dimension.
    """
    def __init__(self, config):
        super().__init__()
        self.d_embd = config.d_embd
        self.max_seq_len = config.max_seq_len

    def forward(self):
        pos = torch.arange(self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_embd, 2) * -(torch.log(torch.tensor(10000.0)) / self.d_embd))
        
        pe = torch.zeros(self.max_seq_len, self.d_embd)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        return pe
```


```python
pos_encoder = SinusoidalPositionalEncoding(config)
pos_encoding = pos_encoder()

pos_encoding.shape  # Should be [max_seq_len, d_embd]
```




    torch.Size([100, 768])




```python
class LearnedPositionalEncoding(nn.Module):
    """
    Implements the LearnedPositionalEncoding layer.

    Args:
        config (TransformerConfig): The configuration for the transformer model.
    """
    def __init__(self, config):
        super().__init__()

        self.pos_embd = nn.Embedding(config.max_seq_len, config.d_embd)
        self.dropout = nn.Dropout(config.drop)

    def forward(self, x):
        embd = x + self.pos_embd(torch.arange(x.size(1), device=x.device))
        return self.dropout(embd)
```


```python
encoding_layer = LearnedPositionalEncoding(config)
encoding_outputs = encoding_layer(torch.rand(1, 100, 768))
encoding_outputs.size()
```




    torch.Size([1, 100, 768])



#### 3.5 Encoder

Each of the six layers of both the encoder and decoder is composed of two main sub-layers: a ``multihead self-attention`` sub-layer, which as explained hereinabove allows the model to focus on different parts of the input sequence, and a ``position-wise fully connected feed-forward`` sub-layer. In addition, the model employs a ``residual connection`` around each of the two sub-layers, followed by ``layer normalization``. In our case, we implement pre layer (instead of post layer) normalization with ``Dropout`` regularization to favour stability during training and prevent overfitting, respectively.

> ``layer normalization`` contributes to having zero mean and unitity variance. This helps to stabilize the learning process, and to reduce the number of training steps.

> ``residual connection`` or ``skip connection`` helps alleaviate the problem of vanishing gradients by passing a tensor to the next layer of the model without processing it and adding it to the processed tensor. In other words, the output of each sub-layer is
LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
itself.


```python
class EncoderLayer(nn.Module):
    """
    Implements a single Encoder layer.

    Args:
        config (TransformerConfig): The configuration for the transformer model.
    """
    def __init__(self, config):
        super().__init__()

        self.norm_1 = nn.LayerNorm(config.d_embd)
        self.masked_attn = MultiHeadAttention(config)

        self.norm_2 = nn.LayerNorm(config.d_embd)
        self.feed_forward = PositionWiseFeedForward(config)
        
        self.dropout = nn.Dropout(config.drop)
        
    def forward(self, x, mask=None):
        attn_outputs = self.multihead_attn(self.norm_1(x), mask=mask)
        x = x + self.dropout(attn_outputs)

        output = x + self.dropout(self.feed_forward(self.norm_2(x)))
        return output
```


```python
encoder_layer = EncoderLayer(config)
encoder_layer(torch.rand(1, 100, 768)).size()
```




    torch.Size([1, 100, 768])




```python
class Encoder(nn.Module):
    """
    Implements the Encoder stack.

    Parameters:
        config (TransformerConfig): The configuration for the transformer model.
    """
    def __init__(self, config):
        super().__init__()
        self.pos_enc = LearnedPositionalEncoding(config)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layer)])

    def forward(self, x, mask=None):
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
```


```python
encoder = Encoder(config)
encoder(torch.rand(1, 100, 768)).size()
```




    torch.Size([1, 100, 768])



#### 3.6 Decoder

The Decoder has two attention sub-layers: ``masked multi-head self-attention layer`` and ``encoder-decoder attention layer``.


```python
class DecoderLayer(nn.Module):
    """
    Implements a single Decoder layer.

    Args:
        config (TransformerConfig): The configuration for the transformer model.
    """
    def __init__(self, config):
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.d_embd)
        self.masked_attn = MultiHeadAttention(config)

        self.norm_2 = nn.LayerNorm(config.d_embd)
        self.cross_attn = MultiHeadAttention(config)

        self.norm_3 = nn.LayerNorm(config.d_embd)
        self.feed_forward = PositionWiseFeedForward(config)
        
        self.dropout = nn.Dropout(config.drop)

    def forward(self, x, enc_output, mask=None):
        attn_output = self.masked_attn(x, mask)
        x = self.norm_1(x + self.dropout(attn_output))

        # cross_attn_output = self.cross_attn(x, enc_output) ??
        cross_attn_output = self.cross_attn(x)
        x = self.norm_2(x + self.dropout(cross_attn_output))

        output = self.norm_3(x + self.dropout(self.feed_forward(x)))
        return output
```


```python
decoder_layer = DecoderLayer(config)
decoder_layer(torch.rand(1, 100, 768), torch.rand(1, 100, 768), mask=None).size()
```




    torch.Size([1, 100, 768])




```python
class Decoder(nn.Module):
    """
    Implements the Decoder stack.

    Args:
        config (TransformerConfig): The configuration for the transformer model.
    """
    def __init__(self, config):
        super().__init__()
        self.pos_enc = LearnedPositionalEncoding(config)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layer)])

    def forward(self, x, enc_output, mask=None):
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, enc_output, mask)
        return x
```


```python
decoder = Decoder(config)
decoder(x=torch.rand(1, 100, 768), enc_output=torch.rand(1, 100, 768)).size()
```




    torch.Size([1, 100, 768])



#### 3.7 Transformer

Note that the ``Decoder`` class takes in an additional argument ``enc_output`` which is the output of the ``Encoder`` stack. This is used in the cross-attention mechanism to calculate the attention scores between the decoder input and the encoder output.

The ``source mask`` is typically used in the encoder to ignore padding tokens in the input sequence., whereas the ``target mask`` is used in the decoder to ignore also padding tokens, and to ensure that predictions for each token can only depend on previous tokens. This enforces causality in the decoder output.


```python
class Transfomer(nn.Module):
    """
    Implements the Transformer architecture.

    Args:
        config (TransformerConfig): The configuration for the transformer model.
    """
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def generate_mask(self, src, tgt):
        src_mask = (src != config.pad_token_id).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != config.pad_token_id).unsqueeze(1).unsqueeze(3)
        tgt_mask = tgt_mask & torch.tril(torch.ones((tgt.size(-1), tgt.size(-1)), device=tgt.device)).bool()
      
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        enc_output = self.encoder(src, mask=src_mask)
        dec_output = self.decoder(tgt, enc_output, mask=tgt_mask)
        
        return dec_output
```

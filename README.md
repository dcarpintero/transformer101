# Transformer101

Vanilla implementation in Pytorch of the Transformer model as introduced in the paper [Attention Is All You Need, 2017](https://arxiv.org/pdf/1706.03762.pdf) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin.

``Scaled Dot-Product Attention`` | ``Multi-Head Attention`` | ``Absolute Positional Encodings`` | ``Learned Positional Encodings`` | ``Dropout`` | ``Layer Normalization`` | ``Residual Connection`` | ``Linear Layer`` | ``Position-Wise Feed-Forward Layer`` | ``GELU`` | ``Softmax`` | ``Encoder`` | ``Decorder`` | ``Transformer``

*Note that this is just an in progress learning project - if you are looking for production grade implementations, refer to the [PyTorch Transformer Class](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html), and [OLMo](https://github.com/allenai/OLMo/), a fully open language model.*

## 1. Background

Sequence modeling and transduction tasks, such as language modeling and machine translation, were typically addressed with RNNs and CNNs. However, these architectures are limited by: (i) ``long training times``, due to the sequential nature of RNNs, which constrains parallelization, and results in increased memory and computational demands as the text sequence grows; and (ii) ``difficulty in learning dependencies between distant positions``, where CNNs, although much less sequential than RNNs, require a number of steps to integrate information that is, in most cases, correlated (linearly for models like ConvS2S and logarithmically for ByteNet) with the distance between elements in the sequence.

## 2. Technical Approach

The paper 'Attention is All You Need' introduced the novel Transformer model, ``a stacked encoder-decoder architecture that utilizes self-attention mechanisms instead of recurrence and convolution to compute input and output representations``. In this model, each of the six layers of both the encoder and decoder is composed of two main sub-layers: a multihead self-attention sub-layer, which allows the model to focus on different parts of the input sequence, and a position-wise fully connected feed-forward sub-layer.

At its core, the ``self-attention mechanism`` enables the model to weight the relationships between input tokens at different positions, resulting in a more effective handling of long-range dependencies. Additionally, by integrating ``multiple attention heads``, the model gains the ability to simultaneously attend to various aspects of the input data during training.

In the proposed implementation, the input and output tokens are converted to 512-dimensional embeddings, to which ``positional embeddings`` are added, enabling the model to use sequence order information.

## 3. Transformer Model Implementation


```python
import math
import torch
import torch.nn as nn
import torch.optim as optim
```


```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """ 
    Transformer (model) configuration
    """

    d_model: int = 768         # dimension of the token embeddings (hideen size of the model)
    n_layer: int = 6           # number of encoder/decoder layers
    n_head: int = 12           # number of self-attention heads
    d_ff: int = 2048           # dimension of the feedforward network
    src_vocab_size: int = 32   # size of the source vocabulary
    tgt_vocab_size: int = 48   # size of the vocabulary
    drop: float = 0.1          # dropout probability
    max_seq_len: int = 100     # maximum sequence length
    pad_token_id: int = 0      # padding token id (usually 0)
    activation: str = "gelu"   # activation function
```


```python
config = ModelConfig()
```

### 3.1 Self-Attention

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
        self.d_head = config.d_model // config.n_head

        self.q = nn.Linear(config.d_model, self.d_head)
        self.k = nn.Linear(config.d_model, self.d_head)
        self.v = nn.Linear(config.d_model, self.d_head)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dim_k = torch.tensor(k.size(-1), dtype=torch.float32)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / torch.sqrt(dim_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, axis=-1)
        output = torch.bmm(attn_weights, v)
        return output

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (torch.Tensor): query embeddings.
            k (torch.Tensor): key embeddings.
            v (torch.Tensor): value embeddings.
            mask (torch.Tensor): attention mask.
        """
        output = self.scaled_dot_product_attention(self.q(q), 
                                                   self.k(k), 
                                                   self.v(v), 
                                                   mask=mask)
        return output
```


```python
attention_head = AttentionHead(config)

x = torch.randn(10, 32, config.d_model)
"""
10: batch size
32: sequence length
config.d_model: hidden size (embedding dimension)
"""

output = attention_head(x, x, x)

print(output.shape)
# Should be [10, 32, d_head == 768 / 12]
# 
# Note that in the linear projection step, q, k, and v are in practice splitted into n_head parts.
# Those will be then concatenated and projected to the final output size 
# in the MultiHeadAttention class (see below).
```

    torch.Size([10, 32, 64])


### 3.2 Multi-Head Attention

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
        assert config.d_model % config.n_head == 0, "d_model must be divisible by n_head"

        self.heads = nn.ModuleList([AttentionHead(config) for _ in range(config.n_head)])
        self.linear = nn.Linear(config.d_model, config.d_model)

    def forward(self, q, k, v, mask=None):
        attn_outputs = torch.cat([h(q, k, v, mask) for h in self.heads], dim=-1)
        output = self.linear(attn_outputs)
        return output
```


```python
multihead_attn = MultiHeadAttention(config)

x = torch.randn(10, 32, config.d_model)
attn_output = multihead_attn(x, x, x)
attn_output.size()
# Should be [10, 32, d_model
# 
# Note that the output size is the same as the input size, 
# as the attention scores of each head are concatenated and projected back to the original size.
```




    torch.Size([10, 32, 768])



### 3.3 Position-Wise Feed-Forward Layer

The Transformer, primarily built upon linear operations like ``dot products`` and ``linear projections``, relies on the ``Position-Wise Feed-Forward Layer`` to introduce non-linearity into the model. This non-linearity enables the model to capture complex data patterns and relationships. The layer typically consists of two linear transformations with a ``non-linear activation function (like ReLU or GELU)``. Each layer in the ``Encoder`` and ``Decoder`` includes one of these feed-forward networks, allowing the model to build increasingly abstract representations of the input data as it passes through successive layers. Note that since this layer processes each embedding independly, the computations can be fully parallelized.

In summary, this Layer comprises:
- ``First linear transformation`` to the input tensor. 
- A non-linear ``activation function`` to allow the model learn more complex patterns.
- ``Second linear transformation``, increasing the model's capacity to learn complex relationships in the data.
- ``Dropout``, a regularization technique used to prevent overfitting. It randomly zeroes some of the elements of the input tensor with a certain probability during training.

> Note that the ``ReLU`` function is a faster function that activates units only when the input is possitive, which can lead to sparse activations (that can be intended in some tasks); whereas ``GELU``, introduced after``ReLU``, offers smoother activation by modeling the input as a stochastic process, providing a probabilistic gate in the activation. In practice, ``GELU`` has been the preferred choice in the BERT and GPT models. Although recent models like LLaMA, PaLM, and [OLMo](https://allenai.org/olmo/olmo-paper.pdf) use the [SwiGLU](https://www.semanticscholar.org/paper/GLU-Variants-Improve-Transformer-Shazeer/bdbf780dfd6b3eb0c9e980887feae5f23af15bc4) activation function


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
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.drop)
        )

    def forward(self, x):
        return self.ff(x)
```


```python
ff = PositionWiseFeedForward(config)
x = torch.randn(10, 32, config.d_model)
ff(x).size()
# Should be [10, 32, d_model]
```




    torch.Size([10, 32, 768])



### 3.4 Positional Encoding

Since the Transformer model contains no recurrence and no convolution, the model is invariant to the position of the tokens. By adding ``positional encoding`` to the input sequence, the Transformer model can differentiate between tokens based on their position in the sequence, which is important for tasks such as language modeling and machine translation. In practice, ``positional encodings`` are added to the input embeddings at the bottoms of the ``encoder`` and ``decoder`` stacks. 

> As outlined in the original [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) paper, ``Sinusoidal Positional Encoding`` and ``Learned Positional Encoding`` produce nearly identical results.




```python
class PositionalEncoding(nn.Module):
    """
    Implements the PositionalEncoding layer.

    Args:
        config (TransformerConfig): The configuration for the transformer model.
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len

        position = torch.arange(self.max_seq_len).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, self.d_model, 2) / self.d_model)

        pe = torch.zeros(1, self.max_seq_len, self.d_model)
        pe[0, :, 0::2] = torch.sin(position / div_term)
        pe[0, :, 1::2] = torch.cos(position / div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
```


```python
pe = PositionalEncoding(config)
x = torch.randn(10, 64, config.d_model)
pe(x).size()
```




    torch.Size([10, 64, 768])



### 3.5 Encoder

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

        self.norm_1 = nn.LayerNorm(config.d_model)
        self.masked_attn = MultiHeadAttention(config)

        self.norm_2 = nn.LayerNorm(config.d_model)
        self.feed_forward = PositionWiseFeedForward(config)
        
        self.dropout = nn.Dropout(config.drop)
        
    def forward(self, x, mask=None):
        attn_outputs = self.masked_attn(x, x, x, mask=mask)
        x = x + self.dropout(attn_outputs)

        output = x + self.dropout(self.feed_forward(self.norm_2(x)))
        return output
```


```python
encoder_layer = EncoderLayer(config)
x = torch.randn(10, 32, config.d_model)
encoder_layer(x).size()
# Should be [10, 32, d_model]
```




    torch.Size([10, 32, 768])




```python
class Encoder(nn.Module):
    """
    Implements the Encoder stack.

    Parameters:
        config (TransformerConfig): The configuration for the transformer model.
    """
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.src_vocab_size, config.d_model)
        self.pe = PositionalEncoding(config)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.n_layer)])

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(config.d_model)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
```


```python
encoder = Encoder(config)
x = torch.randint(0, config.src_vocab_size, (10, 32))
encoder(x).size()
# Should be [10, 32, d_model]
```




    torch.Size([10, 32, 768])



### 3.6 Decoder

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
        self.norm_1 = nn.LayerNorm(config.d_model)
        self.masked_attn = MultiHeadAttention(config)

        self.norm_2 = nn.LayerNorm(config.d_model)
        self.cross_attn = MultiHeadAttention(config)

        self.norm_3 = nn.LayerNorm(config.d_model)
        self.feed_forward = PositionWiseFeedForward(config)
        
        self.dropout = nn.Dropout(config.drop)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_output = self.masked_attn(x, x, x, tgt_mask)
        x = self.norm_1(x + self.dropout(attn_output))

        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm_2(x + self.dropout(attn_output))

        output = self.norm_3(x + self.dropout(self.feed_forward(x)))
        return output
```


```python
decoder_layer = DecoderLayer(config)
x = torch.randn(10, 32, config.d_model)
encoder_output = torch.randn(10, 32, config.d_model)
decoder_layer(x, encoder_output).size()
# Should be [10, 32, d_model]
```




    torch.Size([10, 32, 768])




```python
class Decoder(nn.Module):
    """
    Implements the Decoder stack.

    Args:
        config (TransformerConfig): The configuration for the transformer model.
    """
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.tgt_vocab_size, config.d_model)
        self.pe = PositionalEncoding(config)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layer)])

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x) * math.sqrt(config.d_model)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x
```


```python
decoder = Decoder(config)
x = torch.randint(0, config.tgt_vocab_size, (10, 32))
torch.randn(10, 32, config.d_model).size()
```




    torch.Size([10, 32, 768])



### 3.7 Transformer

Note that the ``Decoder`` class takes in an additional argument ``enc_output`` which is the output of the ``Encoder`` stack. This is used in the cross-attention mechanism to calculate the attention scores between the decoder input and the encoder output.

The ``source mask`` is typically used in the encoder to ignore padding tokens in the input sequence., whereas the ``target mask`` is used in the decoder to ignore also padding tokens, and to ensure that predictions for each token can only depend on previous tokens. This enforces causality in the decoder output.


```python
class Transformer(nn.Module):
    """
    Implements the Transformer architecture.

    Args:
        config (TransformerConfig): The configuration for the transformer model.
    """
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.linear = nn.Linear(config.d_model, config.tgt_vocab_size)

    def generate_mask(self, x):
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
        return mask

    def forward(self, src, tgt):
        tgt_mask = self.generate_mask(tgt)
        
        enc_output = self.encoder(src, mask=None)
        dec_output = self.decoder(tgt, enc_output, src_mask=None, tgt_mask=tgt_mask)
        
        return self.linear(dec_output)
```


```python
model = Transformer(config)
src = torch.randint(0, config.src_vocab_size, (10, 32))
tgt = torch.randint(0, config.tgt_vocab_size, (10, 32))

model(src, tgt).size()
# Should be [10, 32, tgt_vocab_size]
```




    torch.Size([10, 32, 48])



## 4. Training

We will train our transformer model to translate from english to spanish. The dataset has been obtained from https://tatoeba.org


```python
# use pandas to read txt in tabular pairs
import pandas as pd
def load_dataset():
    df = pd.read_csv('data/en-es.txt', sep='\t', header=None)
    en = df[0].tolist()
    es = df[1].tolist()
    return en, es
```


```python
src, tgt = load_dataset()
```


```python
src[3500:3510]
```




    ['We apologize.',
     'We are happy.',
     'We are young.',
     'We can do it.',
     "We can't sue.",
     "We can't win.",
     'We got ready.',
     'We got ready.',
     'We had lunch.',
     'We have some.']




```python
tgt[3500:3510]
```




    ['Pedimos disculpas.',
     'Somos felices.',
     'Somos jóvenes.',
     'Podemos hacerlo.',
     'No podemos demandar.',
     'No podemos ganar.',
     'Nos preparamos.',
     'Estábamos listos.',
     'Almorzamos.',
     'Tenemos algo.']




```python
len(src), len(tgt)
```




    (118964, 118964)




```python
import re

def create_vocab(corpus):
    vocab = set()
    for s in corpus:
        vocab.update(re.findall(r'\w+|[^\w\s]', s))
    w2i = {w: i+4 for i, w in enumerate(vocab)}
    w2i['PAD'] = 0
    w2i['SOS'] = 1
    w2i['EOS'] = 2
    w2i['UNK'] = 3
    i2w = {i: w for w, i in w2i.items()}

    return w2i, i2w
```


```python
src_w2i, src_i2w = create_vocab(src)
tgt_w2i, tgt_i2w = create_vocab(tgt)
print(len(src_w2i), len(tgt_w2i))
```

    14779 28993



```python
def encode(corpus, w2i):
    encoding = []
    for s in corpus:
        s_enc = [w2i[w] for w in re.findall(r'\w+|[^\w\s]', s)]
        s_enc = [w2i['SOS']] + s_enc + [w2i['EOS']]
        encoding.append(s_enc)
    return encoding
```


```python
src_enc = encode(src, src_w2i)
tgt_enc = encode(tgt, tgt_w2i)
```


```python
from sklearn.model_selection import train_test_split

src_train, src_test, tgt_train, tgt_test = train_test_split(src_enc,
                                                            tgt_enc,
                                                            test_size=0.2,
                                                            random_state=42)

```


```python
len(src_train), len(src_test), len(tgt_train), len(tgt_test)
```




    (95171, 23793, 95171, 23793)




```python
def prepare_batch(X, Y):
    max_len_X = max([len(x) for x in X])
    max_len_Y = max([len(y) for y in Y])

    enc_input = torch.zeros(len(X), max_len_X, dtype=torch.long)
    dec_input = torch.zeros(len(Y), max_len_Y, dtype=torch.long)
    output = torch.zeros(len(Y), max_len_Y, dtype=torch.long)

    for i, s in enumerate(X):
        enc_input[i, :len(s)] = torch.tensor(s)

    for i, s in enumerate(Y):
        dec_input[i, :len(s)-1] = torch.tensor(s[:-1])
        output[i, :len(s)-1] = torch.tensor(s[1:])

    return enc_input, dec_input, output
```


```python
from sklearn.utils import shuffle

def batch_generator(X, Y, batch_size):
    X, Y = shuffle(X, Y, random_state=42)
    for i in range(0, len(X), batch_size):
        yield prepare_batch(X[i:i+batch_size], Y[i:i+batch_size])
    
```


```python
batch_size = 4
train_loader = batch_generator(src_train, tgt_train, batch_size)
be, bd, bo = next(train_loader)
```


```python
print(be.size(), bd.size(), bo.size())
```

    torch.Size([4, 12]) torch.Size([4, 11]) torch.Size([4, 11])



```python
print([src_i2w[w.item()] for w in be[0]])
print([tgt_i2w[w.item()] for w in bd[0]])
print([tgt_i2w[w.item()] for w in bo[0]])
```

    ['SOS', 'I', "'", 'm', 'not', 'responsible', 'for', 'what', 'Tom', 'did', '.', 'EOS']
    ['SOS', 'No', 'soy', 'responsable', 'de', 'lo', 'que', 'hizo', 'Tom', '.', 'PAD']
    ['No', 'soy', 'responsable', 'de', 'lo', 'que', 'hizo', 'Tom', '.', 'EOS', 'PAD']



```python
criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```


```python
config.n_head = 8
config.max_seq_len = max([len(s) for s in src + tgt])
config.src_vocab_size = len(src_w2i)
config.tgt_vocab_size = len(tgt_w2i)

model = Transformer(config).to(device)
config
```




    ModelConfig(d_model=768, n_layer=6, n_head=8, d_ff=2048, src_vocab_size=14779, tgt_vocab_size=28993, drop=0.1, max_seq_len=278, pad_token_id=0, activation='gelu')



We perform a dummy run of 2 epochs with a batch of 3 to demonstrate that the model can be trained. We recommend at least 20 epochs and a batch of len(src_train) // batch_size.


```python
model.train()
epochs = 2

for epoch in range(epochs):
    epoch_loss = [] 

    for _ in range(3): # range(len(src_train) // batch_size):
        be, bd, bs = next(train_loader)
        be, bd, bs = be.to(device), bd.to(device), bs.to(device)

        optimizer.zero_grad()
        output = model(be, bd)
        loss = criterion(output.permute(0, 2, 1), bs)
        # src, tgt, output
        print(f'src: {[src_i2w[w.item()] for w in be[0]]}')
        print(f'tgt: {[tgt_i2w[w.item()] for w in bd[0]]}')
        print(f'out: {[tgt_i2w[w.item()] for w in output.argmax(dim=-1)[0]]}')
        print("--")
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
    print("----------------------------------------------------------")
    print(f'Epoch: {epoch}, Loss: {sum(epoch_loss) / len(epoch_loss)}')
    print("----------------------------------------------------------")
print('Training finished')
```

    src: ['SOS', 'I', 'hope', 'Tom', "'", 's', 'right', '.', 'EOS', 'PAD']
    tgt: ['SOS', 'Ojalá', 'que', 'Tom', 'tenga', 'razón', '.', 'PAD', 'PAD', 'PAD']
    out: ['inmortal', 'pong', 'Jugo', 'sencillo', 'enséñeme', 'lesión', 'provocadas', 'escriba', 'llevarte', 'casualmente']
    --
    src: ['SOS', 'Nobody', 'else', 'showed', 'up', '.', 'EOS', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD']
    tgt: ['SOS', 'No', 'apareció', 'nadie', 'más', '.', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD']
    out: ['capacitada', 'narizota', 'preocupas', 'ganchos', 'enfermedades', 'murmurándose', 'copie', 'oficinistas', 'Haré', 'Prueba', 'aceptable', 'hechas']
    --
    src: ['SOS', 'Is', 'the', 'lecture', 'already', 'finished', '?', 'EOS', 'PAD', 'PAD', 'PAD', 'PAD']
    tgt: ['SOS', '¿', 'Ha', 'finalizado', 'ya', 'la', 'conferencia', '?', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD']
    out: ['Esas', 'derretirán', 'NASA', 'contrataron', 'Trabajaré', 'retrasando', 'nuevas', 'reformar', 'perdone', 'tibia', 'inventario', 'olvidarse', 'encontrada']
    --
    ----------------------------------------------------------
    Epoch: 0, Loss: 10.40126864115397
    ----------------------------------------------------------
    src: ['SOS', 'I', 'couldn', "'", 't', 'speak', 'French', '.', 'EOS', 'PAD']
    tgt: ['SOS', 'No', 'sabía', 'hablar', 'francés', '.', 'PAD', 'PAD', 'PAD', 'PAD']
    out: ['guapísima', 'pong', 'meteorito', 'ocasionalmente', 'hablo', 'árbitro', 'fotógrafo', 'tesoros', 'controversias', 'aportando']
    --
    src: ['SOS', 'Come', 'closer', 'to', 'me', '.', 'EOS', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD']
    tgt: ['SOS', 'Acércate', 'más', 'a', 'mí', '.', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD', 'PAD']
    out: ['grafiti', 'dotación', 'aclararse', 'Marque', 'apoyas', 'expulsado', 'reté', 'cercano', 'Accidentalmente', 'asiste', 'diablos', 'divertiría', 'Recientemente']
    --
    src: ['SOS', 'You', "'", 're', 'a', 'good', 'journalist', '.', 'EOS']
    tgt: ['SOS', 'Vos', 'sos', 'un', 'buen', 'periodista', '.', 'PAD']
    out: ['retórica', 'chavo', 'Irás', 'tocador', 'dejarle', 'asignatura', 'participantes', 'logrado']
    --
    ----------------------------------------------------------
    Epoch: 1, Loss: 10.315495808919271
    ----------------------------------------------------------
    Training finished


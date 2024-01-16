## Transformer 101

A vanilla annotated implementation of the Transformer model introduced in the paper [Attention Is All You Need, 2017](https://arxiv.org/pdf/1706.03762.pdf) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin.

Tags: ``[Scaled Dot-Product Attention]`` ``[Multi-Head Attention]`` ``[Absolute Positional Encodings]`` ``[Learned Positional Encodings]`` ``[Dropout]`` ``[Layer Normalization]`` ``[Residual Connection]`` ``[Linear Layer]`` ``[Position-Wise Feed-Forward Layer]`` ``[GELU]`` ``[Softmax]`` ``[Transformer]``

> *This is an in progress project aimed exclusively at educational purposes. The [notebook version](nb.transformer101.ipynb) is intentionally verbose to allow the reader grasp the technical concepts behind the Transformer architecture. The [nano version]() is a lightweight implementation of the Transformer in less than 150 lines of code.*

### 1. Background

Sequence modeling and transduction tasks, such as language modeling and machine translation, were typically addressed with RNNs and CNNs. However, these architectures are limited by: (i) ``long training times``, due to the sequential nature of RNNs, which constrains parallelization, and results in increased memory and computational demands as the text sequence grows; and (ii) ``difficulty in learning dependencies between distant positions``, where CNNs, although much less sequential than RNNs, require a number of steps to integrate information that is, in most cases, correlated (linearly for models like ConvS2S and logarithmically for ByteNet) with the distance between elements in the sequence.

### 2. Technical Approach

The paper [Attention Is All You Need, 2017](https://arxiv.org/pdf/1706.03762.pdf) introduced the novel Transformer model, ``a stacked encoder-decoder architecture that utilizes self-attention mechanisms instead of recurrence and convolution to compute input and output representations``. In this model, each of the six layers of both the encoder and decoder is composed of two main sub-layers: a multihead self-attention sub-layer, which allows the model to focus on different parts of the input sequence, and a position-wise fully connected feed-forward sub-layer.

At its core, the ``self-attention mechanism`` enables the model to weigh the relationships between input tokens at different positions, resulting in a more effective handling of long-range dependencies. Additionally, by integrating ``multiple attention heads``, the model gains the ability to simultaneously attend to various aspects of the input data during training.

In the proposed implementation, the input and output tokens are converted to 512-dimensional embeddings, to which ``positional embeddings`` are added, enabling the model to use sequence order information.


### 3. Architecture

#### 3.1 Self-Attention

The intuition behind ``self-attention`` is that averaging token embeddings instead of using a fixed embedding for each token, enables the model to capture how words relate to each other in the input. In practice, said weighted relationships (attention weights) represent the syntactic and contextual structure of the sentence, leading to a more nuanced and rich understanding of the data.

The most common way to implement a self-attention layer is based on ``scaled dot-product attention``, which involves:
1. ``Linear projection`` of each token embedding into three vectors: ``query (q)``, ``key (k)``, ``value (v)``.
2. Compute ``scaled attention scores``: determine the similary between ``q`` and ``k`` by applying the ``dot product``. Since the results of this function are typically large numbers, they are then divided by a scaling factor inferred from the dimensionality of (k). This scaling contributes to stabilize gradients during training.
3. Normalize the ``attention scores`` into ``attention weights`` by applying the ``softmax`` function (this ensures all the values sum to 1).
4. ``Update the token embeddings`` by multiplying the attention weights by the value vector.

> A mask will be used only in the decoder layer. The main idea is to prevent the decoder from having access to future tokens in the sequence it is generating. In practice, this is implemented with a binary mask that designates which tokens should be attended to (assigned non-zero weights) and which should be ignored (assigned zero weights). In our function, setting the future tokens (upper values) to negative infinity guarantees that the attention weights become zero after applying the softmax function (e exp -inf == 0). This design aligns with the nature of many tasks like translation, summarization, or text generation, where the output sequence needs to be generated one element at a time, and the prediction of each element should be based only on the previously generated elements.

#### 3.2 Multi-Headed Attention

In a standard attention mechanism, the ``softmax`` of a single head tends to concentrate on a specific aspect of similarity, potentially overlooking other relevant features in the input. By integrating multiple attention heads, the model gains the ability to simultaneously attend to various aspects of the input data.

The basic approach to implement Multi-Headed Attention comprises:

1. Initialize the ``attention heads``. E.g. BERT has 12 attention heads whereas the embeddings dimension is 768, resulting in 768 / 12 = 64 as the head dimension.
2. ``Concatenate attention heads`` to combines the outputs of the attention heads into a single vector while preserving the dimensionality of the embeddings.
3. Apply a ``linear projection``.

> Note that the softmax function is a probability distribution, which when applied within a single attention head tends to amplify certain features (those with higher scores) while diminishing others. Thus, leading to a focus on specific aspects of similarity.

#### 3.3 Position-Wise Feed-Forward Layer

The Transformer, primarily built upon linear operations like ``dot products`` and ``linear projections``, relies on the ``Position-Wise Feed-Forward Layer`` to introduce non-linearity into the model. This non-linearity enables the model to capture complex data patterns and relationships. The layer typically consists of two linear transformations with a ``non-linear activation function (like ReLU or GELU)``. Each layer in the ``Encoder`` and ``Decoder`` includes one of these feed-forward networks, allowing the model to build increasingly abstract representations of the input data as it passes through successive layers. Note that since this layer processes each embedding independly, the computations can be fully parallelized.

In summary, this Layer comprises:
- ``First linear transformation`` to the input tensor. 
- A non-linear ``activation function`` to allow the model learn more complex patterns.
- ``Second linear transformation``, increasing the model's capacity to learn complex relationships in the data.
- ``Dropout``, a regularization technique used to prevent overfitting. It randomly zeroes some of the elements of the input tensor with a certain probability during training.

> Note that the ``ReLU`` function is a faster function that activates units only when the input is possitive, which can lead to sparse activations (that can be intended in some tasks); whereas ``GELU``, introduced after``ReLU``, offers smoother activation by modeling the input as a stochastic process, providing a probabilistic gate in the activation. In practice, ``GELU`` has been the preferred choice in the BERT and GPT models.

#### 3.4 Positional Encoding

Since the Transformer model contains no recurrence and no convolution, the model is invariant to the position of the tokens. By adding ``positional encoding`` to the input sequence, the Transformer model can differentiate between tokens based on their position in the sequence, which is important for tasks such as language modeling and machine translation. In practice, ``positional encoding`` are added to the input embeddings at the bottoms of the ``encoder`` and ``decoder`` stacks. 

> As outlined in the original [Attention Is All you Need](https://arxiv.org/pdf/1706.03762.pdf) paper, ``Sinusoidal Positional Encoding`` and ``Learned Positional Encoding`` produces nearly identical results.

#### 3.5 Encoder

Each of the six layers of both the encoder and decoder is composed of two main sub-layers: a ``multihead self-attention`` sub-layer, which as explained hereinabove allows the model to focus on different parts of the input sequence, and a ``position-wise fully connected feed-forward`` sub-layer. In addition, the model employs a ``residual connection`` around each of the two sub-layers, followed by ``layer normalization``. In our case, we implement pre layer (instead of post layer) normalization with ``Dropout`` regularization to favour stability during training and prevent overfitting, respectively.

> ``layer normalization`` contributes to having zero mean and unitity variance. This helps to stabilize the learning process and reducing the number of training steps.

> ``residual connection`` or ``skip connection`` helps alleaviate the problem of vanishing gradients by passing a tensor to the next layer of the model without processing it and adding it to the processes tensor. In other words, the output of each sub-layer is
LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
itself.

#### 3.6 Decoder

The Decoder has two attention sub-layers: ``masked multi-head self-attention layer`` and ``encoder-decoder attention layer``.

#### 3.7 Transformer

[TO-DO]
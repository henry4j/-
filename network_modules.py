"""Neural Network modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

import math
import numpy as np
import tensorflow.google.compat.v2 as tf
from typing import Optional, Tuple, Union


def sand_network_0320(x_seq,
                      x_seq_mask,
                      x_aux,
                      units=1,
                      embedding_dim=128,
                      num_enc_blocks=2,
                      num_dec_blocks=1,
                      num_dec_units=128,
                      num_attn_heads=4,
                      band_mask=(-1, 0),
                      dropout=0.3,
                      int_seq_length=None,
                      v0=None):
  # pylint: disable=g-doc-args
  """Implements [Simply Attend and Diagnose][1].

  For example, given x_seq, x_seq_mask, and x_aux,

  x_seq: tf.keras.Input([None, 180, 16]),
  x_seq_mask: tf.keras.Input([None, 180]),
  x_aux: tf.keras.Input(None, 7),

  where input sequence length: 180, input feature dimesion: 16,
        and auxiliary input dimension: 7,

  this function returns an output tensor of shape (None, units):

  Returns:
    SAnD network.

  [1]: https://arxiv.org/abs/1706.03762.
  """
  x_seq_shape = x_seq.shape.as_list()
  x_seq_length = x_seq_shape[1]
  conv1d = tf.keras.layers.Conv1D(
      filters=embedding_dim, kernel_size=1, input_shape=x_seq_shape[1:])
  x_seq = conv1d(x_seq)
  x_seq = PositionalEncoding(embedding_dim, x_seq_length)(x_seq)
  if num_attn_heads:
    for _ in range(num_enc_blocks):
      if v0:
        x_seq = encoding_v0(x_seq, x_seq_mask, embedding_dim, num_attn_heads,
                            dropout, band_mask)
      else:
        x_seq = encoding(x_seq, x_seq_mask, embedding_dim, num_attn_heads,
                         dropout, band_mask)
    if int_seq_length and int_seq_length < x_seq_length:
      x_seq = Interpolation(x_seq_length, int_seq_length)(x_seq)
  else:
    for _ in range(num_enc_blocks):
      x_seq = tf.keras.layers.LSTM(
          x_seq_length,
          return_sequences=True,
          dropout=dropout,
          recurrent_dropout=dropout)(
              x_seq, mask=x_seq_mask)
  x_seq = tf.keras.layers.Flatten()(x_seq)
  x = tf.keras.layers.concatenate((x_seq, x_aux))
  for _ in range(num_dec_blocks):
    x = tf.keras.layers.Dense(num_dec_units, 'relu')(x)
  x = tf.keras.layers.Dense(units)(x)
  return x


def encoding(x_seq, x_seq_mask, embedding_dim, num_attn_heads, dropout,
             band_mask):
  """Implements the encoding block from [Attention Is All You Need]."""
  attn = SelfAttention(
      embedding_dim,
      num_attn_heads,
      scale=embedding_dim**-0.5,
      band_mask=band_mask,
      dropout=dropout)
  x = Residual(attn, dropout=0.0)(x_seq, x_seq_mask)
  pwff = PositionWiseFeedForward(embedding_dim)
  return Residual(pwff, dropout=dropout)(x)


class Residual(tf.keras.layers.Wrapper):
  """Implements the residual block from [Attention Is All You Need][1].

  N.B. Alternative: Pre-activation by [Identity Mappings in Residual N/W][2].

  [1]: https://arxiv.org/abs/1706.03762
  [2]: https://arxiv.org/abs/1603.05027
  """

  def __init__(self, layer, dropout=0.0, epsilon=1e-6, **kwargs):
    super(Residual, self).__init__(layer, **kwargs)
    self.dropout = dropout
    self.epsilon = epsilon
    if dropout:
      self.dropout_ = tf.keras.layers.Dropout(rate=dropout)
    else:
      self.dropout_ = tf.keras.layers.Lambda(lambda x: x)
    self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=epsilon)

  def call(self, inputs, mask=None, training=None):
    output = self.layer(inputs, mask=mask)
    output = self.dropout_(output, training)
    return self.layer_norm(inputs + output)

  def get_config(self):
    config = super(Residual, self).get_config()
    config.update({
        'dropout': self.dropout,
        'epsilon': self.epsilon,
    })
    return config


class SelfAttention(tf.keras.layers.Attention):
  """Implements self-attention with multi-head, scale, and band_mask options."""

  def __init__(
      self,
      embedding_dim,
      num_heads,
      band_mask=(-1, 0),  # a tuple: num_lower, num_upper args for tf.band_part.
      scale=None,  # e.g., (embedding_dim**-0.5)
      dropout=0.0,  # for attention weights.
      non_trainable=None,
      **kwargs):
    super(SelfAttention, self).__init__(dropout=dropout, **kwargs)
    if embedding_dim % num_heads:
      raise ValueError(
          'embedding_dim({0}) must be divisible by num_heads({1}).'.format(
              embedding_dim, num_heads))
    self.head_dim = embedding_dim // num_heads
    self.embedding_dim = embedding_dim
    self.num_heads = num_heads
    self.band_mask = band_mask
    if not any((scale, non_trainable)):
      self.scale = self.add_weight(
          name='scale',
          shape=(),
          initializer=tf.ones_initializer(),
          dtype=self.dtype,
          trainable=True)
    else:
      self.scale = scale
    if non_trainable:
      self.wq = tf.keras.layers.Lambda(lambda x: x)
      self.wv = tf.keras.layers.Lambda(lambda x: x)
      self.wk = tf.keras.layers.Lambda(lambda x: x)
      self.linear = tf.keras.layers.Lambda(lambda x: x)
    else:
      self.wq = tf.keras.layers.Dense(embedding_dim)
      self.wv = tf.keras.layers.Dense(embedding_dim)
      self.wk = tf.keras.layers.Dense(embedding_dim)
      self.linear = tf.keras.layers.Dense(embedding_dim)

  def build(self, input_shape):
    self.causal_mask = _new_causal_mask(input_shape[1], self.band_mask)
    # Let's override super(tf.keras.layers.Layer, self).build(input_shape).
    tf.keras.layers.Layer.build(self, input_shape)

  def call(self, inputs, mask=None, training=None):
    x = inputs
    x_shape = tf.shape(x)
    q = self.wq(x)
    v = self.wv(x)
    k = self.wk(x)
    q = self._split(q, x_shape)
    v = self._split(v, x_shape)
    k = self._split(k, x_shape)
    masks = [
        None,  #
        None if mask is None else tf.expand_dims(mask, axis=1)
    ]  # Let's mask value tensor.
    x = super(tf.keras.layers.Attention, self).call([q, v, k], masks, training)
    x = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), x_shape)
    return self.linear(x)

  # See also: http://cs?q=f:/third_party/.*/dense_attention.py$+compute.mask.
  def compute_mask(self, inputs, mask=None):
    pass  # Let's not produce a mask.

  # See also: http://cs?q=f:/third_party/.*/dense_attention.py$+validate.*args.
  def _validate_call_args(self, inputs, mask):
    pass  # Let's not validate args.

  def _calculate_scores(self, query, key):
    attn_logits = tf.linalg.matmul(query, key, transpose_b=True)
    if self.scale is not None:
      attn_logits *= self.scale
    if self.causal_mask is not None:
      attn_logits -= 1e9 * tf.cast(
          tf.math.logical_not(self.causal_mask), tf.float32)
    return attn_logits

  def _split(self, x, x_shape):
    x = tf.reshape(x, [x_shape[0], -1, self.num_heads, self.head_dim])
    return tf.transpose(x, [0, 2, 1, 3])

  def get_config(self):
    config = super(SelfAttention, self).get_config()
    config.update({
        'embedding_dim': self.embedding_dim,
        'num_heads': self.num_heads,
        'scale': self.scale,
        'band_mask': self.band_mask,
        'dropout': self.dropout,
    })
    return config


class PositionWiseFeedForward(tf.keras.layers.Layer):
  """Implements the position-wise FF n/w from [Attention Is All You Need][1].

  In addition to attention sub-layers, each of the layers in the encoder,
  and the decoder contains a fully connected FF network, which is applied
  to each position separately and identically. This FF network consists of
  two linear transformations with a ReLU activation in between.

  [1]: https://arxiv.org/abs/1706.03762
  """

  def __init__(self,
               embedding_dim: int,
               hidden_dim: Optional[int] = None,
               **kwargs) -> tf.keras.layers.Layer:
    super(PositionWiseFeedForward, self).__init__(**kwargs)
    self.embedding_dim = embedding_dim
    self.pwff = tf.keras.Sequential([
        tf.keras.layers.Conv1D(
            filters=hidden_dim or 2 * embedding_dim,
            kernel_size=1,
            activation='relu',
            input_shape=[None, embedding_dim]),
        tf.keras.layers.Conv1D(
            filters=embedding_dim,
            kernel_size=1,
            input_shape=[None, 2 * embedding_dim])
    ])

  def call(self, inputs, **kwargs):
    return self.pwff(inputs)

  def get_config(self):
    config = super(PositionWiseFeedForward, self).get_config()
    config.update({
        'embedding_dim': self.embedding_dim,
    })
    return config


class Interpolation(tf.keras.layers.Layer):
  """Implements the dense interpolation from [Simply Attend and Diagnose][1].

  [1]: https://arxiv.org/abs/1711.03905
  """

  def __init__(self, in_seq_length: int, out_seq_length: int,
               **kwargs) -> tf.keras.layers.Layer:
    super(Interpolation, self).__init__(**kwargs)
    self.in_seq_length = in_seq_length
    self.out_seq_length = out_seq_length
    t = (np.arange(in_seq_length) + 1) / np.float32(in_seq_length + 1)
    m = (np.arange(out_seq_length) + 1) / np.float32(out_seq_length + 1)
    m = m[np.newaxis]
    w = np.square(1 - np.abs(m.T - t))
    self.w = tf.constant(w, tf.float32)

  def call(self, inputs):
    return tf.linalg.matmul(self.w, inputs)

  def get_config(self):
    config = super(Interpolation, self).get_config()
    config.update({
        'in_seq_length': self.in_seq_length,
        'out_seq_length': self.out_seq_length,
    })
    return config


class PositionalEncoding(tf.keras.layers.Layer):
  """Implements the positional encoding from [Attention Is All You Need][1].

  PE(p, 2i) = sin(p/10000 ^(2i/d)) and PE(p, 2i+1) = cos(p/10000 ^(2i/d)),
  where p: position, i: the i-th dimension and d: the input embedding dimension.

  [1]: https://arxiv.org/abs/1706.03762
  """

  def __init__(self,
               embedding_dim: int,
               max_seq_length: Optional[int] = 10000,
               dtype: Optional[tf.dtypes.DType] = tf.float32):
    super(PositionalEncoding, self).__init__(dtype=dtype)
    if embedding_dim % 2:
      embedding_dim += 1  # Let's ensure that it is an even number.
    self.embedding_dim = embedding_dim
    self.max_seq_length = max_seq_length
    e = np.arange(
        embedding_dim, step=2)[np.newaxis, :] / np.float32(embedding_dim)
    p = np.arange(max_seq_length)[:, np.newaxis]
    pe = np.empty((1, max_seq_length, embedding_dim))
    pe[0, :, 0::2] = np.sin(p / np.power(10000, e))
    pe[0, :, 1::2] = np.cos(p / np.power(10000, e))
    self.pe = tf.constant(pe, dtype)

  def call(self, x):
    # Forked from Encoder https://www.tensorflow.org/tutorials/text/transformer.
    x *= math.sqrt(self.embedding_dim)
    x_shape = tf.shape(x)
    x += self.pe[:, :x_shape[1], :x_shape[-1]]
    return x

  def get_config(self):
    config = super(PositionalEncoding, self).get_config()
    config.update({
        'embedding_dim': self.self.embedding_dim,
        'max_seq_length': self.max_seq_length
    })
    return config


def densify_sequence_feature(
    sparse_feature: tf.SparseTensor,
    seq_length: int,
    batch_size: Optional[int] = None,
    pad_left: Optional[bool] = False,
    pad_value: Union[None, int, float] = None) -> tf.Tensor:
  """Converts a sparse sequence feature into a dense feature.

  For example, given input sparse_feature:
    [[9],
     [],
     [0, 6, 7, 8],
     [],
     [3, 4, 5]] of dense shape [5, 4],
     seq_length = 2, batch size = 3, pad_left = True, pad_value = -1,

  this function returns output dense feature as follows:
    [[-1,  9],
     [-1, -1],
     [ 7,  8]] of dense shape [3, 2].

  N.B.: Imagine that larger numbers in the input represent newer data points.
        pad_left emits newer data points (9, 8, 7) in the 1st and 3rd rows.

  Args:
    sparse_feature: Sparse feature tensor.
    seq_length: Length for the output dense feature tensor.
    batch_size: Size for the output dense feature tensor.
    pad_left: Whether to pad left.
    pad_value: Default values to fill in the output dense feature tensor.

  Returns:
    A dense feature tensor of shape [batch_size, seq_length].
  """
  if pad_value is None:
    pad_value = tf.zeros([], dtype=sparse_feature.dtype)
  if batch_size is None:
    batch_size = sparse_feature.dense_shape[0]
  seq_feature_lengths = sequence_feature_lengths(sparse_feature)
  densified_feature = tf.sparse.to_dense(
      sparse_feature, default_value=pad_value)
  densified_feature = tf.reverse_sequence(
      densified_feature, seq_feature_lengths, seq_axis=1, batch_axis=0)
  densified_feature = tf.pad(
      densified_feature, [[0, 0], [0, seq_length]],
      mode='CONSTANT',
      constant_values=pad_value)
  densified_feature = tf.slice(densified_feature, [0, 0],
                               [batch_size, seq_length])
  if pad_left:
    densified_feature = tf.reverse(densified_feature, axis=[-1])
  else:
    densified_feature = tf.reverse_sequence(
        densified_feature,
        tf.minimum(seq_feature_lengths[:batch_size], seq_length),
        seq_axis=1,
        batch_axis=0)
  return densified_feature


def sequence_feature_lengths(sparse_feature: tf.SparseTensor) -> tf.Tensor:
  return tf.maximum(
      tf.add(
          tf.math.unsorted_segment_max(sparse_feature.indices[:, 1],
                                       sparse_feature.indices[:, 0],
                                       sparse_feature.dense_shape[0]), 1),
      0)  # Let's zero out integer lowests from empty sequences.


def _new_causal_mask(seq_length: int,
                     band_mask: Optional[Tuple[int]] = (-1, 0),
                     dtype: Optional[tf.dtypes.DType] = tf.bool):
  mask = tf.ones((seq_length, seq_length), dtype)
  mask = tf.linalg.band_part(mask, band_mask[0], band_mask[1])
  return mask


def encoding_v0(x_seq, x_seq_mask, embedding_dim, num_attn_heads, dropout,
                band_mask):
  """Implements the encoding block from [Attention Is All You Need]."""
  attn = SelfAttentionV0(embedding_dim, num_attn_heads, band_mask)
  x = Residual(attn, dropout=dropout)(x_seq, x_seq_mask)
  pwff = PositionWiseFeedForwardV0(embedding_dim)
  return Residual(pwff, dropout=dropout)(x)


class SelfAttentionV0(tf.keras.layers.Layer):
  """Implements multi-head attention with a band mask."""

  def __init__(self, embedding_dim, num_heads, band_mask):
    super(SelfAttentionV0, self).__init__()
    assert embedding_dim % num_heads == 0
    self.head_dim = embedding_dim // num_heads
    self.embedding_dim = embedding_dim
    self.num_heads = num_heads
    self.band_mask = band_mask
    self.wq = tf.keras.layers.Dense(embedding_dim)
    self.wv = tf.keras.layers.Dense(embedding_dim)
    self.wk = tf.keras.layers.Dense(embedding_dim)
    self.linear = tf.keras.layers.Dense(embedding_dim)

  def build(self, input_shape):
    self.causal_mask = 1 - _new_causal_mask(input_shape[1], self.band_mask,
                                            tf.float32)

  def call(self, inputs):
    x = inputs
    x_shape = tf.shape(x)
    q, k, v = self.wq(x), self.wk(x), self.wv(x)
    q, k, v = (self._split(q, x_shape), self._split(k, x_shape),
               self._split(v, x_shape))
    v, unused_att_weights = scaled_dot_product_attention(
        q, k, v, self.causal_mask)
    x = tf.reshape(tf.transpose(v, [0, 2, 1, 3]), x_shape)
    return self.linear(x)

  def _split(self, x, x_shape):
    x = tf.reshape(x, (x_shape[0], -1, self.num_heads, self.head_dim))
    return tf.transpose(x, [0, 2, 1, 3])

  def get_config(self):
    config = super(SelfAttentionV0, self).get_config()
    config.update({
        'embedding_dim': self.embedding_dim,
        'num_heads': self.num_heads,
        'band_mask': self.band_mask,
        'dropout': self.dropout,
    })
    return config


class PositionWiseFeedForwardV0(tf.keras.layers.Layer):
  """Implements the position-wise FF n/w from [Attention Is All You Need][1].

  In addition to attention sub-layers, each of the layers in the encoder,
  and the decoder contains a fully connected FF network, which is applied
  to each position separately and identically. This FF network consists of
  two linear transformations with a ReLU activation in between.

  [1]: https://arxiv.org/abs/1706.03762
  """

  def __init__(self, embedding_dim, hidden_dim=None, **kwargs):
    super(PositionWiseFeedForwardV0, self).__init__(**kwargs)
    self.embedding_dim = embedding_dim
    self.pwff = tf.keras.Sequential([
        tf.keras.layers.Dense(
            hidden_dim or 2 * embedding_dim, activation='relu'),
        tf.keras.layers.Dense(embedding_dim)
    ])

  def call(self, inputs, **kwargs):
    return self.pwff(inputs)

  def get_config(self):
    config = super(PositionWiseFeedForwardV0, self).get_config()
    config.update({
        'embedding_dim': self.embedding_dim,
    })
    return config


# Forked from http://go/transformer-tutorial#scaled_dot_product_attention
def scaled_dot_product_attention(q, k, v, attn_mask=None):
  """Returns a tuple of (scaled dot product output, attention weights).

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    attn_mask: make (1 for padding, 0 else) with shape broadcastable to (...,
      seq_len_q, seq_len_k).

  Returns:
    output, attention_weights
  """
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  attn_logits = tf.linalg.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
  if attn_mask is not None:
    attn_logits += (attn_mask * -1e9)
  attn_weights = tf.nn.softmax(attn_logits, axis=-1)
  return tf.linalg.matmul(attn_weights, v), attn_weights

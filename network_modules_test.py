"""Tests for google3.experimental.users.henrylee.py.input_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.google.compat.v2 as tf

import experimental.users.henrylee.py.network_modules as m

kl = tf.keras.layers


class NetworkModulesTest(tf.test.TestCase):

  def test_self_attention_with_mask(self):
    # Test attention outputs by SelfAttention with Keras mask and
    # scaled dot product with padding mask from [transformer tutorial][1].
    # [1]: http://go/transformer-tutorial#scaled_dot_product_attention.
    batch_size = 3
    q = tf.constant([[[0.5], [0.8], [-0.3]]])
    q3 = tf.repeat(q, repeats=batch_size, axis=0)
    attn_mask = tf.constant([  # 1 for padding indicates a time step to ignore.
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0]],
        [[0.0, 1.0, 1.0]],
    ])
    expected = m.scaled_dot_product_attention(q3, q3, q3, attn_mask)[0]
    mask = tf.constant([  # False in Keras mask indicates a time step to ignore.
        [True, True, True],
        [True, True, False],
        [True, False, False],
    ])
    sa = m.SelfAttention(
        embedding_dim=1, num_heads=1, band_mask=(-1, -1), non_trainable=True)
    actual = sa(q3, mask)
    self.assertAllClose(actual, expected)

  # Forked from http://cs?q=f:third.party+def.test_self_attention_causal.
  def test_causal_self_multi_head_attention(self):
    head_dim, num_heads = 2, 3
    q = tf.constant([[[0.5], [0.8], [-0.3]]])
    q2 = tf.repeat(q, head_dim, -1)
    attn = kl.Attention(causal=True)
    expected = tf.repeat(attn([q2, q2, q2]), num_heads, -1)
    sa = m.SelfAttention(
        head_dim * num_heads, num_heads=num_heads, non_trainable=True)
    actual = sa(tf.repeat(q2, num_heads, -1))
    self.assertAllClose(actual, expected)

  def test_non_causal_self_multi_head_attention(self):
    head_dim, num_heads = 2, 3
    q = tf.constant([[[0.5], [0.8], [-0.3]]])
    q2 = tf.repeat(q, head_dim, -1)
    non_causal_attn = kl.Attention(causal=False)
    expected = tf.repeat(non_causal_attn([q2, q2, q2]), num_heads, -1)
    non_causal_sa = m.SelfAttention(
        head_dim * num_heads,
        num_heads=num_heads,
        band_mask=(-1, -1),
        non_trainable=True)
    actual = non_causal_sa(tf.repeat(q2, num_heads, -1))
    self.assertAllClose(actual, expected)

  def test_positional_encoding(self):
    expected_pe = tf.constant([[  #
        [0., 1., 0., 1.],  # i.e., sine(0), cosine(0), sine(0), and cosine(0).
        [0.84147, 0.54031, 0.0099, 0.9999],
        [0.90929, -0.4161, 0.0199, 0.9998],
        [0.14112, -0.9899, 0.0299, 0.9995],
        [-0.7568, -0.6536, 0.0399, 0.9992],
        [-0.9589, 0.28366, 0.0499, 0.9987],
        [-0.2794, 0.96017, 0.0599, 0.9982],
        [0.65698, 0.75390, 0.0699, 0.9975]
    ]])
    pe = m.PositionalEncoding(embedding_dim=4, max_seq_length=8)
    actual_pe = pe(tf.zeros((1, 8, 4)))
    self.assertAllClose(actual_pe, expected_pe, atol=1e-3)

  def test_position_wise_feed_forward(self):
    # Let's feed-forward a sequence of (5, 3, 7),
    # where batch size: 5, input sequence length: 3, embedding dimension: 7.
    in_seq_shape = (5, 3, 7)
    pwff = m.PositionWiseFeedForward(7, 9, name='pwff')
    out_seq_shape = pwff(tf.zeros(in_seq_shape)).shape
    wb_shape = lambda e: pwff.variables[e].shape.as_list()  # Weights/Biases
    expected_hidden_layer_shapes = [[[1, 7, 9], [9]], [[1, 9, 7], [7]]]
    self.assertAllEqual(wb_shape(0), expected_hidden_layer_shapes[0][0])
    self.assertAllEqual(wb_shape(1), expected_hidden_layer_shapes[0][1])
    self.assertAllEqual(wb_shape(2), expected_hidden_layer_shapes[1][0])
    self.assertAllEqual(wb_shape(3), expected_hidden_layer_shapes[1][1])
    self.assertAllEqual(out_seq_shape, in_seq_shape)

  def test_interpolation(self):
    # Let's interpolate a sequence of (1, 4, 3),
    # where batch size: 1, input sequence length: 4, embedding dimension: 3.
    in_seq = tf.ones((1, 4, 3))
    interpolation = m.Interpolation(in_seq_length=4, out_seq_length=2)
    expected_w = tf.constant([  #
        [0.75111, 0.87111, 0.53777, 0.28444],
        [0.28444, 0.53777, 0.87111, 0.75111]
    ])
    expected_out_seq = tf.constant([[  #
        [2.44444, 2.44444, 2.44444],  #
        [2.44444, 2.44444, 2.44444]
    ]])
    self.assertAllClose(interpolation.w, expected_w, rtol=1e-3)
    self.assertAllClose(interpolation(in_seq), expected_out_seq, rtol=1e-3)

  def test_new_causal_mask(self):
    expected_causal_mask = tf.constant([
        [1, 0, 0, 0, 0],  # Looking back 2 time steps, but not look ahead.
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1]
    ])
    actual = m._new_causal_mask(5, [2, 0], tf.int64)
    self.assertAllEqual(actual, expected_causal_mask)

  def test_densify_sequence_feature(self):
    expected_dense_feature = tf.constant([
        [9, 0, 0, 0, 0],  #
        [0, 0, 0, 0, 0],
        [0, 7, 8, 0, 0]
    ])
    sparse_tensor = tf.SparseTensor(
        indices=[[0, 0], [2, 1], [2, 2]], values=[9, 7, 8], dense_shape=[4, 4])
    actual = m.densify_sequence_feature(sparse_tensor, 5, 3)
    self.assertAllEqual(actual, expected_dense_feature)

  def test_densify_sequence_feature_pad_left(self):
    expected_dense_feature = tf.constant([
        [-1, -1, -1, -1, 90],  #
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, 70, 80]
    ])
    sparse_tensor = tf.SparseTensor(
        indices=[[0, 0], [2, 1], [2, 2]],
        values=[90, 70, 80],
        dense_shape=[4, 4])
    actual = m.densify_sequence_feature(sparse_tensor, 5, 3, True, -1)
    self.assertAllEqual(actual, expected_dense_feature)

  def test_sequence_feature_lengths(self):
    expected_lengths = tf.constant([1, 0, 3, 0])
    sparse_tensor = tf.SparseTensor(
        indices=[[0, 0], [2, 1], [2, 2]], values=[9, 7, 8], dense_shape=[4, 4])
    actual = m.sequence_feature_lengths(sparse_tensor)
    self.assertAllEqual(actual, expected_lengths)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()

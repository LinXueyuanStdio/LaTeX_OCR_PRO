from __future__ import division
import math
import numpy as np
import tensorflow as tf


# taken from https://github.com/tensorflow/tensor2tensor/blob/37465a1759e278e8f073cd04cd9b4fe377d3c740/tensor2tensor/layers/common_attention.py


def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.

    Each channel of the input Tensor is incremented by a sinusoid of a difft
    frequency and phase in one of the positional dimensions.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(a+b) and cos(a+b) can
    be experessed in terms of b, sin(a) and cos(a).

    x is a Tensor with n "positional" dimensions, e.g. one dimension for a
    sequence or two dimensions for an image

    We use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels // (n * 2). For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
        x: a Tensor with shape [batch, d1 ... dn, channels]
        min_timescale: a float
        max_timescale: a float

    Returns:
        a Tensor the same shape as x.

    """
    static_shape = x.get_shape().as_list()  # [20, 14, 14, 512]
    num_dims = len(static_shape) - 2  # 2
    channels = tf.shape(x)[-1]  # 512
    num_timescales = channels // (num_dims * 2)  # 512 // (2*2) = 128
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))  # -0.1 / 127
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)  # len == 128
    for dim in range(num_dims):  # dim == 0; 1
        length = tf.shape(x)[dim + 1]  # 14
        position = tf.to_float(tf.range(length))  # len == 14
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
            inv_timescales, 0)  # pos = [14, 1], inv = [1, 128], scaled_time = [14, 128]
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)  # [14, 256]
        prepad = dim * 2 * num_timescales  # 0; 256
        postpad = channels - (dim + 1) * 2 * num_timescales  # 512-(1;2)*2*128 = 256; 0
        signal = tf.pad(signal, [[0, 0], [prepad, postpad]])  # [14, 512]
        for _ in range(1 + dim):  # 1; 2
            signal = tf.expand_dims(signal, 0)
        for _ in range(num_dims - 1 - dim):  # 1, 0
            signal = tf.expand_dims(signal, -2)
        x += signal  # [1, 14, 1, 512]; [1, 1, 14, 512]
    return x

import tensorflow as tf
import tensorflow_datasets as tfds
# import numpy as np
from tensorflow.data.experimental import AUTOTUNE
import jax
from jax import jit, vmap  # , pmap
import jax.numpy as jnp
import util as u
import logging
from functools import partial

# see notebook on derivation of these from training sample

CHANNEL_P999S = jnp.array([2612., 3492., 3582., 3970., 3709., 4407., 5678.,
                           5604., 2264., 52., 4848., 3876., 6124.])

CHANNEL_MEANS = jnp.array([1354.7904, 1117.1971, 1041.1869, 946.0516,
                           1198.119, 2003.2725, 2375.2615, 2302.0972,
                           730.6099, 12.077799, 1821.995, 1119.2013,
                           2602.028])

CHANNEL_STDS = jnp.array([242.14961, 324.46646, 386.99976, 587.74664,
                          565.23846, 859.7307, 1086.1215, 1116.8077,
                          404.7259, 4.397278, 998.8627, 756.0413,
                          1231.3727])


@partial(vmap, in_axes=(1, 0), out_axes=1)
def _clip_per_channel(x, a_max):
    return jnp.clip(x, a_min=0, a_max=a_max)


@partial(vmap, in_axes=(1, 0, 0), out_axes=1)
def _standardise_per_channel(x, mean, std):
    return (x - mean) / std


def clip_and_standardise(x):
    orig_shape = x.shape
    x = x.reshape(-1, 13)
    x = _clip_per_channel(x, CHANNEL_P999S)
    x = _standardise_per_channel(x, CHANNEL_MEANS, CHANNEL_STDS)
    return x.reshape(orig_shape)


@tf.autograph.experimental.do_not_convert
def _augment(x, y):
    # rotate 0, 90, 180 or 270 deg
    k = tf.random.uniform([], 0, 3, dtype=tf.int32)
    x = tf.image.rot90(x, k)
    # flip L/R 50% time
    x = tf.image.random_flip_left_right(x)
    return x, y


def dataset(split, batch_size, channel_mask=None, input_size=64,
            dropout_key=None):
    """Generates pairs of x, y data.

    Args:
      split: which split to use
      batch_size: batch size to iterate with
      channel_mask: channel mask to apply to x; arrays of 0s & 1s
      input_size: spatial size to resize H, W to. 64 is a noop.
      dropout_key: key for dropout_channels; None => no dropout

    Returns:
      yields (x, y) pairs in batch_size

    Raises:
      Exception if both dropout_key and channel_mask set.
    """

    if (dropout_key is not None) and (channel_mask is not None):
        raise Exception("dropout_key", dropout_key, "and channel_mask",
                        channel_mask, "can't both be set")

    is_training = split in ['sample', 'train']

    # note: train is 27,000 total
    split = {'sample': 'train[:1%]',
             'train': 'train[:70%]',               # 18900
             'validate': 'train[70%:80%]',         # 2700
             'ga_train': 'train[80%:90%]',         # 2700
             'ga_validate': 'train[90%:]'}[split]  # 2700

    dataset = tfds.load('eurosat/all', split=split, as_supervised=True)

    if is_training:
        dataset = (dataset.map(_augment, num_parallel_calls=AUTOTUNE)
                          .shuffle(1024))

    dataset = dataset.batch(batch_size)

    def preprocess(x, dropout_key):
        if input_size != 64:
            x = jax.image.resize(x, shape=(input_size, input_size, 13),
                                 method='linear', antialias=True)

        x = clip_and_standardise(x)

        # TODO: masking of channels was put here in the data pipeline when
        #   --model-type single was first written but these would make more
        #   sense to be done in the logits calc as it's done for
        #   --model-type multi-res

        if channel_mask is not None:
            x *= channel_mask

        if dropout_key is not None:
            # sample sequence of 13 0s and 1s
            random_channel_mask = jax.random.randint(
                dropout_key, minval=0, maxval=2, shape=(13,))
            # tile that out to be a mask across the channels of x
            mask = jnp.tile(random_channel_mask, (input_size, input_size, 1))
            # apply that mask
            x *= mask

        return x

    # vectorise over x, but not the channel dropout
    preprocess = jit(vmap(preprocess, in_axes=(0, None)))

    for x, labels in dataset:
        x, labels = jnp.array(x), jnp.array(labels)
        key = None
        if dropout_key is not None:
            dropout_key, key = jax.random.split(dropout_key)
        x = preprocess(x, key)
        yield x, labels

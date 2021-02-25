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


@jit
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


def dataset(split, batch_size, shuffle_seed=123):

    # # choose split that divides evenly across 4 hosts. when force_small_data
    # # is set (e.g. local dev smoke test) just use 10 examples for everything.
    # h = jax.host_id()
    # if training:  # train[:80%]
    #     if force_small_data:
    #         per_host_egs = 10
    #     else:
    #         per_host_egs = 5400
    #     split = f"train[{h*per_host_egs}:{(h+1)*per_host_egs}]"
    # else:  # validation; # train[80%:90%]
    #     if force_small_data:
    #         per_host_egs = 10
    #     else:
    #         per_host_egs = 675
    #     split = f"train[{21600+(h*per_host_egs)}:{21600+((h+1)*per_host_egs)}]"
    # logging.info("for host %s (training=%s) split is %s",
    #              jax.host_id(), training, split)

    is_training = split in ['sample', 'train']

    split = {'sample': 'train[:1%]',
             'train': 'train[:70%]',
             'tune_1': 'train[70%:80%]',
             'tune_2': 'train[80%:90%]',
             'test': 'train[90%:]'}[split]

    dataset = tfds.load('eurosat/all', split=split, as_supervised=True)

    if is_training:
        dataset = (dataset.map(_augment, num_parallel_calls=AUTOTUNE)
                          .shuffle(1024, seed=shuffle_seed))

    dataset = dataset.batch(batch_size)

    for imgs, labels in dataset:
        imgs, labels = jnp.array(imgs), jnp.array(labels)
        imgs = clip_and_standardise(imgs)
        yield imgs, labels


# def shard_dataset(imgs, labels):
#     # clip lens to ensure can be reshaped to leading 8
#     n = (len(imgs) // 8) * 8
#     imgs, labels = imgs[:n], labels[:n]

#     # resize to leading dim of 8
#     imgs = imgs.reshape(8, n // 8, 64, 64, 13)
#     labels = labels.reshape(8, n // 8)

#     # return sharded
#     return u.shard(imgs), u.shard(labels)

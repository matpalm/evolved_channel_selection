import tensorflow_datasets as tfds
import numpy as np
import jax
from jax import jit, vmap, pmap
import jax.numpy as jnp
import util as u
import logging

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


def dataset(training, force_small_data=False):

    # choose split that divides evenly across 4 hosts
    # when force_small_data is set (e.g. local dev smoke test)
    # just use 10 examples for everything
    h = jax.host_id()
    if training:  # train[:80%]
        if force_small_data:
            per_host_egs = 10
        else:
            per_host_egs = 5400
        split = f"train[{h*per_host_egs}:{(h+1)*per_host_egs}]"
    else:  # validation; # train[80%:90%]
        if force_small_data:
            per_host_egs = 10
        else:
            per_host_egs = 675
        split = f"train[{21600+(h*per_host_egs)}:{21600+((h+1)*per_host_egs)}]"
    logging.info("for host %s (training=%s) split is %s",
                 jax.host_id(), training, split)

    # load images and labels as single batch
    dataset = tfds.load('eurosat/all', split=split, as_supervised=True)
    for tfe_imgs, tfe_labels in dataset.batch(100_000):
        break

    return tfe_imgs, tfe_labels


@partial(vmap, in_axes=(1, 0), out_axes=1)
def clip_per_channel(x, a_max):
    return jnp.clip(x, a_min=0, a_max=a_max)


@partial(vmap, in_axes=(1, 0, 0), out_axes=1)
def standardise_per_channel(x, mean, std):
    return (x - mean) / std


@jit
def clip_and_standardise(x):
    orig_shape = x.shape
    x = x.reshape(-1, 13)
    x = clip_per_channel(x, CHANNEL_P999S)
    x = standardise_per_channel(x, CHANNEL_MEANS, CHANNEL_STDS)
    return x.reshape(orig_shape)


def preprocess_and_shard_dataset(tfe_imgs, tfe_labels):
    # clip size to ensure can be reshaped to leading 8
    imgs, labels = np.array(tfe_imgs), np.array(tfe_labels)
    del tfe_imgs  # transistent OOM ??
    del tfe_labels
    n = (len(imgs) // 8) * 8
    imgs, labels = imgs[:n], labels[:n]

    # clip and standardise
    imgs = clip_and_standardise(imgs)

    # resize to leading dim of 8
    imgs = imgs.reshape(8, n // 8, 64, 64, 13)
    labels = labels.reshape(8, n // 8)

    # return sharded
    return u.shard(imgs), u.shard(labels)


def rot90(m, k=1):
    # jax.numpy.rot90 uses conditionals on k which means we
    # can't vmap over them. but if we swap them to jnp.wheres
    # we can!
    # https://jax.readthedocs.io/en/latest/_modules/jax/_src/numpy/lax_numpy.html#rot90
    ax1, ax2 = 0, 1
    k = k % 4

    def not_0():
        return jnp.where(k == 2,
                         jnp.flip(jnp.flip(m, ax1), ax2),
                         not_0_or_2())

    def not_0_or_2():
        perm = list(range(m.ndim))
        perm[ax1], perm[ax2] = perm[ax2], perm[ax1]
        return jnp.where(k == 1,
                         jnp.transpose(jnp.flip(m, ax2), perm),
                         jnp.flip(jnp.transpose(m, perm), ax2))

    return jnp.where(k == 0, m, not_0())


def random_flip(m, k):
    return jnp.where(k == 0, m, jnp.fliplr(m))


def augment(img, rot, flip):  # (H,W,C) -> (H,W,C)
    img = rot90(img, rot)
    img = random_flip(img, flip)
    return img


def all_combos_augment(img):  # (H,W,C) -> (8,H,W,C)
    rots = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    flips = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    v_augment = vmap(augment, in_axes=(None, 0, 0))
    return v_augment(img, rots, flips)


@pmap
def batched_all_combos_augment(imgs):
    v_all_combos_augment = vmap(all_combos_augment)
    imgs = v_all_combos_augment(imgs)    # (B,H,W,C) -> (B,8,H,W,C)
    return imgs.reshape(-1, 64, 64, 13)  # (8B,H,W,C)

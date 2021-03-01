import jax
from jax import vmap
import haiku as hk
from jax.nn import gelu
import jax.numpy as jnp
from functools import partial


def conv(c):
    return hk.Conv2D(output_channels=c, kernel_shape=3, stride=2)


def global_spatial_mean_pooling(x):
    return jnp.mean(x, axis=(1, 2))


def _single_trunk_model(x):
    # input (64, 64, 13)
    num_classes = 10
    return hk.Sequential(
        [conv(32), gelu,                      # (32, 32, 32)
         conv(64), gelu,                      # (16, 16, 64)
         conv(128), gelu,                     # (8, 8, 128)
         global_spatial_mean_pooling,         # (128)
         hk.Linear(32), gelu,                 # (32)
         hk.Linear(num_classes)])(x)          # (10)


def construct_single_trunk_model():
    return hk.without_apply_rng(hk.transform(_single_trunk_model))


def downsample(x, factor):
    return hk.max_pool(x,
                       window_shape=(1, factor, factor, 1),
                       strides=(1, factor, factor, 1),
                       padding='VALID')


def _multires_model(x64, channel_selection):
    """Builds multiresolution shared trunk model.

    Args:
      x64: full res input; (B, 64, 64, 13)
      channel_selection: array of length 13 representing which resolution
        to select per channel. values are 0 to 4 representing...
         0-> x64, 1 -> x32, 2 -> x16, 3 -> x8, 4 -> zero.

    Returns:
      untransformed haiku model.
    """

    # make three downscaled variants of the input; down to a HW of 32, 16
    # and 8. note: we don't bother with the zero case (channel_selection=4)
    # since it's implied by being masked out in all resolutions.
    x32 = downsample(x64, 2)  # (B, 32, 32, 13)
    x16 = downsample(x64, 4)  # (B, 16, 16, 13)
    x8 = downsample(x64, 8)   # (B, 8, 8, 13)

    # mask out different channels for the different resolutions depending
    # on channel_selection. each channel will be represented by either the
    # zero case or one, and only one, resolution.
    def mask_for(x, channel_idx):
        mask_shape = list(x.shape)
        mask_shape[-1] = 1
        ch_mask = jnp.equal(channel_selection, channel_idx)
        ch_mask = jnp.tile(ch_mask.astype(jnp.int32), mask_shape)
        return ch_mask
    x64 *= mask_for(x64, 0)
    x32 *= mask_for(x32, 1)
    x16 *= mask_for(x16, 2)
    x8 *= mask_for(x8, 3)
    # recall: channel_selection = 4 implies "select zero"; i.e. mask
    # everything else.

    # build a common trunk and apply to each of the four resolutions
    common_trunk = hk.Sequential(
        [conv(32), gelu,                      # (B, HW, HW, 32)
         conv(64), gelu,                      # (B, HW/2, HW/2, 64)
         conv(128), gelu,                     # (B, HW/4, HW/4, 128)
         global_spatial_mean_pooling,         # (B, 128)
         hk.Linear(32), gelu])                # (B, 32)
    t64 = common_trunk(x64)  # (B, 32)
    t32 = common_trunk(x32)  # (B, 32)
    t16 = common_trunk(x16)  # (B, 32)
    t8 = common_trunk(x8)    # (B, 32)

    # concatenate the 4 trunks into one and do one extra mixing
    head = jnp.concatenate([t64, t32, t16, t8], axis=-1)  # (B, 128)
    head = gelu(hk.Linear(32)(head))                      # (B, 32)

    # calculate logits
    num_classes = 10
    logits = hk.Linear(num_classes)(head)                 # (B, 10)
    return logits


def construct_multires_model():
    return hk.without_apply_rng(hk.transform(_multires_model))

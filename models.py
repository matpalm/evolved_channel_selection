import haiku as hk
from jax.nn import gelu
import jax.numpy as jnp


def conv(c):
    return hk.Conv2D(output_channels=c, kernel_shape=3, stride=2)


def global_spatial_mean_pooling(x):
    return jnp.mean(x, axis=(1, 2))


def single_trunk_model(x, dense_kernel_size=32, num_classes=10):
    # input (64, 64, 13)
    return hk.Sequential(
        [conv(32), gelu,                      # (32, 32, 32)
         conv(64), gelu,                      # (16, 16, 64)
         conv(128), gelu,                     # (8, 8, 128)
         global_spatial_mean_pooling,         # (128)
         hk.Linear(dense_kernel_size), gelu,  # (32)
         hk.Linear(num_classes)])(x)          # (10)

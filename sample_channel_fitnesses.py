import data
import models
import jax
from jax import jit, vmap
import jax.numpy as jnp
import util as u
import argparse
import sys
import simple_ga
import numpy as np

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--params', type=str, required=True)
parser.add_argument('--split', type=str, required=True)
parser.add_argument('--num-examples', type=int, required=True)
parser.add_argument('--model-type', type=str, default='single',
                    help="model type; 'single' or 'multi-res'")

opts = parser.parse_args()
print(opts, file=sys.stderr)

assert opts.model_type in ['single', 'multi-res']

dataset = data.dataset(split=opts.split,
                       batch_size=opts.num_examples)
for x, y_true in dataset:
    break


params = u.load_params(opts.params)


@jit
def mean_loss(member):
    if opts.model_type == 'single':
        # member denotes a channel mask we want to apply to
        # entire x batch
        model = models.construct_single_trunk_model()
        mask_tile_shape = list(x.shape)
        mask_tile_shape[-1] = 1
        mask = jnp.tile(member, mask_tile_shape)
        logits = model.apply(params, x * mask)
    else:  # multi-res
        # member denotes channel selection handled in model
        model = models.construct_multires_model()
        logits = model.apply(params, x, member)
    return u.softmax_cross_entropy(logits, y_true).mean()


@jit
def channel_penalty(member):
    if opts.model_type == 'single':
        return 0
    penalty = jnp.sum(jnp.equal(member, 0)) * 0.8   # x64
    penalty += jnp.sum(jnp.equal(member, 1)) * 0.4  # x32
    penalty += jnp.sum(jnp.equal(member, 2)) * 0.2  # x16
    penalty += jnp.sum(jnp.equal(member, 3)) * 0.1  # x8
    # channel 4, x0, is free to use
    return penalty


def new_member():
    if opts.model_type == 'single':
        return np.random.randint(0, 2, size=(13,))
    else:  # multi-res
        return np.random.randint(0, 5, size=(13,))


print("member\tloss\tpenalty")
for _ in range(1000):
    random_member = new_member()
    m = list(random_member)
    l = float(mean_loss(random_member))
    p = float(channel_penalty(random_member))
    print("\t".join(map(str, [m, l, p])))
    sys.stdout.flush()

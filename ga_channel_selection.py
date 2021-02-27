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
parser.add_argument('--popn-size', type=int, required=True)
parser.add_argument('--num-epochs', type=int, default=100)
parser.add_argument('--proportion-new-members', type=float, default=0.1)
parser.add_argument('--proportion-elite', type=float, default=0.0)
opts = parser.parse_args()
print(opts, file=sys.stderr)

dataset = data.dataset(split=opts.split,
                       batch_size=opts.num_examples)
for x, y_true in dataset:
    break

mask_tile_shape = list(x.shape)
mask_tile_shape[-1] = 1

model = models.construct_single_trunk_model()
params = u.load_params(opts.params)


def inv_mean_loss(channel_mask):
    mask = jnp.tile(channel_mask, mask_tile_shape)
    logits = model.apply(params, x * mask)
    mean_loss = u.softmax_cross_entropy(logits, y_true).mean()
    return 1.0 / mean_loss


inv_mean_loss = jit(vmap(inv_mean_loss))

baseline = jnp.stack([
    jnp.zeros(13,),  # no channels
    jnp.ones(13,)])  # all channels
print("baseline no / all channels fitness", inv_mean_loss(baseline))


def new_member():
    return np.random.randint(0, 2, size=(13,))


ga = simple_ga.SimpleGA(popn_size=opts.popn_size,
                        new_member_fn=new_member,
                        fitness_fn=inv_mean_loss,
                        cross_over_fn=simple_ga.np_array_crossover,
                        proportion_new_members=0.2,
                        proportion_elite=0.0)

for _ in range(opts.num_epochs):
    print("ga.members", ga.members)
    ga.calc_fitnesses()
    print("fitness", ga.raw_fitness_values)
    print("selection", ga.selection_array)
    print("popn mean fitness", np.mean(ga.raw_fitness_values))
    print("elite", ga.get_elite_member())
    ga.breed_next_gen()

ga.calc_fitnesses()
print("elite", ga.get_elite_member())

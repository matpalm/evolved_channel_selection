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
parser.add_argument('--model-type', type=str, default='single',
                    help="model type; 'single' or 'multi-res'")

opts = parser.parse_args()
print(opts, file=sys.stderr)

assert opts.model_type in ['single', 'multi-res']

FITNESS_EVAL_BATCHES = 10
if opts.popn_size % FITNESS_EVAL_BATCHES != 0:
    raise Exception("only support population size that's"
                    " a multiple of %d" % FITNESS_EVAL_BATCHES)

dataset = data.dataset(split=opts.split,
                       batch_size=opts.num_examples)
for x, y_true in dataset:
    break


params = u.load_params(opts.params)


@jit
def inv_mean_loss(member):
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
    mean_loss = u.softmax_cross_entropy(logits, y_true).mean()
    return 1.0 / mean_loss


@jit
def channel_penalty(member):
    penalty = jnp.sum(jnp.equal(member, 0)) * 0.8   # x64
    penalty += jnp.sum(jnp.equal(member, 1)) * 0.4  # x32
    penalty += jnp.sum(jnp.equal(member, 2)) * 0.2  # x16
    penalty += jnp.sum(jnp.equal(member, 3)) * 0.1  # x8
    # channel 4, x0, is free to use
    return penalty


def fitness(member):
    return inv_mean_loss(member) - channel_penalty(member)


fitness = jit(vmap(fitness))


def eval_fitness_in_batches(members):
    # have to manually do vectorisation here since entire lot
    # vectorised with vmap will OOM o_O
    fitnesses = []
    members = members.reshape(FITNESS_EVAL_BATCHES,
                              opts.popn_size // FITNESS_EVAL_BATCHES, 13)
    for b in range(FITNESS_EVAL_BATCHES):
        fitnesses.append(fitness(members[b]))
    return jnp.concatenate(fitnesses)


def new_member():
    if opts.model_type == 'single':
        return np.random.randint(0, 2, size=(13,))
    else:  # multi-res
        return np.random.randint(0, 5, size=(13,))


ga = simple_ga.SimpleGA(popn_size=opts.popn_size,
                        new_member_fn=new_member,
                        fitness_fn=eval_fitness_in_batches,
                        cross_over_fn=simple_ga.np_array_crossover,
                        proportion_new_members=0.2,
                        proportion_elite=0.0)

for _ in range(opts.num_epochs):
    #print("ga.members", ga.members)
    ga.calc_fitnesses()
    print("fitness", ga.raw_fitness_values)
    print("selection", ga.selection_array)
    print("popn mean fitness", np.mean(ga.raw_fitness_values))
    e = ga.get_elite_member()
    print("elite %s inv_mean_loss %f channel_penalty %f" %
          (e, inv_mean_loss(e), channel_penalty(e)))
    ga.breed_next_gen()

ga.calc_fitnesses()
print("elite", ga.get_elite_member())

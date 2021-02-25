import jax.numpy as jnp
from jax import pmap, host_id, jit
from jax.tree_util import tree_map
import datetime


def shard(x):
    # pmap x across first axis
    return pmap(lambda v: v)(x)


def replicate(x, replicas=8):
    # replicate leafs of x and then shard
    replicated = tree_map(lambda v: jnp.stack([v] * replicas), x)
    return shard(replicated)


def shapes_of(pytree):
    # rebuild a pytree swapping actual params for just shape and type
    return tree_map(lambda v: (v.shape, type(v), v.dtype), pytree)


def DTS():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def primary_host():
    return host_id() == 0


def accuracy(model, params, dataset):
    num_correct = 0
    num_total = 0

    @jit
    def predict(x):
        logits = model.apply(params, x)
        return jnp.argmax(logits, axis=-1)

    for x, y_true in dataset:
        y_pred = predict(x)
        num_correct += jnp.sum(y_true == y_pred)
        num_total += len(y_true)

    accuracy = float(num_correct / num_total)
    return accuracy
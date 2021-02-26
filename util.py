import jax.numpy as jnp
from jax import pmap, host_id, jit
from jax.tree_util import tree_map
from jax.nn import one_hot, log_softmax
import datetime
import os
import pickle


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


def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_dir_exists_for_file(fname):
    ensure_dir_exists(os.path.dirname(fname))


def primary_host():
    return host_id() == 0


def softmax_cross_entropy(logits, labels):
    one_hot_labels = one_hot(labels, logits.shape[-1])
    return -jnp.sum(log_softmax(logits) * one_hot_labels, axis=-1)


def accuracy_mean_loss(model, params, dataset):
    num_correct = 0
    total_loss = 0
    num_total = 0

    @jit
    def predict_with_losses(x, y_true):
        logits = model.apply(params, x)
        losses = softmax_cross_entropy(logits, y_true)
        return jnp.argmax(logits, axis=-1), losses

    for x, y_true in dataset:
        y_pred, losses = predict_with_losses(x, y_true)
        num_correct += jnp.sum(y_true == y_pred)
        total_loss += jnp.sum(losses)
        num_total += len(y_true)

    accuracy = float(num_correct / num_total)
    mean_loss = float(total_loss / num_total)

    return accuracy, mean_loss


def save_params(run, epoch, params):
    fname = f"params/{run}/{epoch}.pkl"
    ensure_dir_exists_for_file(fname)
    with open(fname, 'wb') as f:
        pickle.dump(params, f)


def load_params(fname):
    with open(fname, 'rb') as f:
        params = pickle.load(f)
    return params

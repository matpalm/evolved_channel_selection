
import logging
logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO)

import jax
from jax import jit, grad
import jax.numpy as jnp
import models
import data as d
import optax
from functools import partial
import util as u


def train(opts):

    run = u.DTS()
    logging.info("run %s", run)

    host_rng = jax.random.PRNGKey(opts.seed ^ jax.host_id())
    pod_rng = jax.random.PRNGKey(opts.seed - 1)

    model = models.construct_single_trunk_model()

    pod_rng, init_key = jax.random.split(pod_rng)
    representative_input = jnp.zeros((1, 64, 64, 13))
    params = model.init(init_key, representative_input)

    opt = optax.adam(opts.learning_rate)
    opt_state = opt.init(params)

    def softmax_cross_entropy(logits, labels):
        one_hot = jax.nn.one_hot(labels, logits.shape[-1])
        return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)

    @jit
    def mean_cross_entropy(params, x, y_true):
        logits = model.apply(params, x)
        return jnp.mean(softmax_cross_entropy(logits, y_true))

    @jit
    def update(params, opt_state, x, y_true):
        grads = grad(mean_cross_entropy)(params, x, y_true)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    best_validation_accuracy = 0
    best_validation_epoch = None
    for epoch in range(opts.epochs):

        # make one pass through training set
        train_dataset = d.dataset(split='train', batch_size=opts.batch_size)
        for x, y_true in train_dataset:
            params, opt_state = update(params, opt_state, x, y_true)

        # just report loss for final batch (note: this is _post_ the grad update)
        mean_last_batch_loss = mean_cross_entropy(params, x, y_true).mean()

        # calculate validation loss
        validate_dataset = d.dataset(split='tune_1', batch_size=opts.batch_size)
        accuracy = u.accuracy(model, params, validate_dataset)
        if accuracy > best_validation_accuracy:
            best_validation_accuracy = accuracy
            best_validation_epoch = epoch
            u.save_params(run, epoch, params)

        logging.info("epoch %d mean_last_batch_loss %0.4f"
                     " validate accuracy %0.3f", epoch, mean_last_batch_loss,
                     accuracy)

    logging.info("best_validation_accuracy %0.3f best_validation_epoch %d",
                 best_validation_accuracy, best_validation_epoch)


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=2)
    opts = parser.parse_args()
    print(opts, file=sys.stderr)

    train(opts)

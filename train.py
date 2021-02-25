
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
import wandb


def train(opts):

    run = u.DTS()
    logging.info("run %s", run)

    # only run wandb stuff if it's configured, and only on primary host
    wandb_enabled = (opts.group is not None) and u.primary_host()
    if wandb_enabled:
        wandb.init(project='evolved_channel_selection', group=opts.group,
                   name=run, reinit=True)
        # save group again explicitly to work around sync bug that drops
        # group when 'wandb off'
        wandb.config.group = opts.group
        wandb.config.seed = opts.seed
        wandb.config.learning_rate = opts.learning_rate
        wandb.config.batch_size = opts.batch_size
        wandb.config.input_size = opts.input_size
    else:
        logging.info("not using wandb and/or not primary host")

    host_rng = jax.random.PRNGKey(opts.seed ^ jax.host_id())
    pod_rng = jax.random.PRNGKey(opts.seed - 1)

    model = models.construct_single_trunk_model()

    pod_rng, init_key = jax.random.split(pod_rng)
    representative_input = jnp.zeros((1, opts.input_size, opts.input_size, 13))
    params = model.init(init_key, representative_input)
    # logging.debug("params %s", u.shapes_of(params))

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
        train_dataset = d.dataset(split='train',
                                  batch_size=opts.batch_size,
                                  input_size=opts.input_size)
        for x, y_true in train_dataset:
            params, opt_state = update(params, opt_state, x, y_true)

        # just report loss for final batch (note: this is _post_ the grad update)
        mean_last_batch_loss = mean_cross_entropy(params, x, y_true).mean()

        # calculate validation loss
        validate_dataset = d.dataset(split='tune_1',
                                     batch_size=opts.batch_size,
                                     input_size=opts.input_size)
        accuracy = u.accuracy(model, params, validate_dataset)
        if accuracy > best_validation_accuracy:
            best_validation_accuracy = accuracy
            best_validation_epoch = epoch
            u.save_params(run, epoch, params)

        stats = {'loss': float(mean_last_batch_loss),
                 'validate_accuracy': accuracy}
        logging.info("epoch %d stats %s", epoch, stats)
        if wandb_enabled:
            wandb.log(stats, step=epoch)

    final_stats = {'best_validation_accuracy': best_validation_accuracy,
                   'best_validation_epoch': best_validation_epoch}
    logging.info("final_stats %s", final_stats)
    if wandb_enabled:
        wandb.log(final_stats, step=epoch)
        wandb.join()


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--group', type=str,
                        help='w&b init group', default=None)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--input-size', type=int, default=64)
    opts = parser.parse_args()
    print(opts, file=sys.stderr)

    assert opts.input_size in [64, 32, 16, 8]

    train(opts)

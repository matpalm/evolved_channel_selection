
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
        wandb.config.dropout_channels = opts.dropout_channels
    else:
        logging.info("not using wandb and/or not primary host")

    host_rng = jax.random.PRNGKey(opts.seed ^ jax.host_id())
    pod_rng = jax.random.PRNGKey(opts.seed - 1)

    pod_rng, init_key = jax.random.split(pod_rng)
    representative_input = jnp.zeros((1, opts.input_size, opts.input_size, 13))

    if opts.model_type == 'single':
        model = models.construct_single_trunk_model()
        params = model.init(init_key, representative_input)
    elif opts.model_type == 'multi-res':
        model = models.construct_multires_model()
        representative_channel_selection = jnp.zeros(13,)
        params = model.init(init_key, representative_input,
                            representative_channel_selection)
    else:
        raise Exception(opts.model_type)

    # logging.debug("params %s", u.shapes_of(params))

    def calc_logits(params, x, dropout_key):
        if opts.model_type == 'single':
            # TODO: move channel masking from data pipeline to here to
            #   more consistent compared to multi-res
            return model.apply(params, x)
        else:  # multi-res
            # TODO: handle --fixed-channel-selection &
            #   --random-channel-selection here
            channel_selection = jax.random.randint(
                dropout_key, minval=0, maxval=5, shape=(13,))
            return model.apply(params, x, channel_selection)

    @jit
    def mean_cross_entropy(params, x, y_true, dropout_key):
        logits = calc_logits(params, x, dropout_key)
        return jnp.mean(u.softmax_cross_entropy(logits, y_true))

    opt = optax.adam(opts.learning_rate)
    opt_state = opt.init(params)

    @jit
    def update(params, opt_state, x, y_true, dropout_key):
        grads = grad(mean_cross_entropy)(params, x, y_true, dropout_key)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    best_validation_accuracy = 0
    best_validation_epoch = None
    for epoch in range(opts.epochs):

        # make one pass through training set
        dropout_key = None
        if opts.dropout_channels:
            host_rng, dropout_key = jax.random.split(host_rng)
        train_dataset = d.dataset(split='train',
                                  batch_size=opts.batch_size,
                                  input_size=opts.input_size,
                                  dropout_key=dropout_key)
        for x, y_true in train_dataset:
            host_rng, dropout_key = jax.random.split(host_rng)
            params, opt_state = update(
                params, opt_state, x, y_true, dropout_key)

        # just report loss for final batch (note: this is _post_ the grad update)
        mean_last_batch_loss = mean_cross_entropy(
            params, x, y_true, dropout_key)

        # calculate validation loss
        validate_dataset = d.dataset(split='validate',
                                     batch_size=opts.batch_size,
                                     input_size=opts.input_size)

        def calc_logits_for_validation(x):
            if opts.model_type == 'single':
                return model.apply(params, x)
            else:  # multi-res
                just_select_x64 = jnp.array([0] * 13)
                return model.apply(params, x, just_select_x64)

        accuracy, _mean_loss = u.accuracy_mean_loss(calc_logits_for_validation,
                                                    validate_dataset)
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

    logging.info("best params params/%s/%s.pkl", run, best_validation_epoch)


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
    parser.add_argument('--model-type', type=str, default='single',
                        help="model type; 'single' or 'multi-res'")
    parser.add_argument('--input-size', type=int, default=64,
                        help="input size for force, only applicable to"
                             " --model-type=single")
    parser.add_argument('--dropout-channels', action='store_true',
                        help="only applicable to"
                             " --model-type=single")
    parser.add_argument('--fixed-channel-selection', type=str, default=None,
                        help="fixed channel selection to use. evaled as an"
                             " int array with values 0 thru 4. only applicable"
                             " to --model-type=multi-res")
    parser.add_argument('--random-channel-selection', action='store_true',
                        help="only applicable to"
                             " --model-type=multi-res")

    opts = parser.parse_args()
    print(opts, file=sys.stderr)

    assert opts.input_size in [64, 32, 16, 8]
    assert opts.model_type in ['single', 'multi-res']

    if opts.model_type != 'single':
        if opts.input_size != 64 or opts.dropout_channels:
            raise Exception("--dropout-channels or input_size != 64 only"
                            " applicable to --model-type=single")
    if opts.model_type != 'multi-res':
        if opts.random_channel_selection or opts.fixed_channel_selection is not None:
            raise Exception("--random-channel-selection and"
                            " --fixed-channel-selection only applicable to"
                            " --model-type=multi-res")
        if opts.random_channel_selection and opts.fixed_channel_selection is not None:
            raise Exception("can't set both --random-channel-selection and"
                            " --fixed-channel-selection")

    train(opts)

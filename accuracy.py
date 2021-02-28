import argparse
import sys
import util as u
import data
import models
import numpy as np
from jax import jit
import jax.numpy as jnp
import json

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--params', type=str, required=True)
parser.add_argument('--split', type=str, required=True)
parser.add_argument('--channel-sweep', action='store_true')
parser.add_argument('--channel-mask', type=str, default=None, required=False)
parser.add_argument('--model-type', type=str, default='single',
                    help="model type; 'single' or 'multi-res'")
parser.add_argument('--fixed-channel-selection', type=str, default=None,
                    help="fixed channel selection to use. evaled as an"
                         " int array with values 0 thru 4. only applicable"
                         " to --model-type=multi-res")
opts = parser.parse_args()
print(opts, file=sys.stderr)

assert opts.model_type in ['single', 'multi-res']

channel_selection = None
if opts.model_type == 'single':
    if opts.fixed_channel_selection is not None:
        raise Exception("--fixed-channel-selection"
                        " applicable to --model-type=multi-res")
elif opts.model_type == 'multi-res':
    if opts.channel_sweep or opts.channel_mask is not None:
        raise Exception("--channel-sweep or --channel-mask only"
                        " applicable to --model-type=single")
    if opts.fixed_channel_selection is None:
        raise Exception("--fixed-channel-selection required for"
                        " --model-type=multi-res")
    channel_selection = eval(opts.fixed_channel_selection)
    assert len(channel_selection) == 13
    channel_selection = jnp.array(channel_selection)

params = u.load_params(opts.params)


@jit
def calc_logits_fn(x):
    if opts.model_type == 'single':
        model = models.construct_single_trunk_model()
        return model.apply(params, x)
    else:  # multi-res'
        model = models.construct_multires_model()
        return model.apply(params, x, channel_selection)


if opts.channel_sweep:
    results = []
    for ch1 in range(0, 13):
        channel_mask = np.ones(13)
        channel_mask[ch1] = 0.0
        # for ch2 in range(ch1 + 1, 13):
        dataset = data.dataset(split=opts.split, batch_size=32,
                               channel_mask=channel_mask)
        accuracy, mean_loss = u.accuracy_mean_loss(calc_logits_fn, dataset)
        results.append((ch1, accuracy, mean_loss))
    for ch1, accuracy, mean_loss in sorted(results, key=lambda v: v[-1]):
        print("ch %02d accuracy %0.3f mean_loss %0.3f" %
              (ch1, accuracy, mean_loss))
    exit()

if opts.channel_mask is not None:
    channel_mask = np.array(eval(opts.channel_mask))
    dataset = data.dataset(split=opts.split, batch_size=32,
                           channel_mask=channel_mask)
else:
    dataset = data.dataset(split=opts.split, batch_size=32)

accuracy, mean_loss = u.accuracy_mean_loss(calc_logits_fn, dataset)

stats = {"channel_selection": "%s" % channel_selection,
         "accuracy": float(accuracy),
         "mean_loss": float(mean_loss)}
print(json.dumps(stats))
sys.stdout.flush()

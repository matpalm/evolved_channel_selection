import argparse
import sys
import util as u
import data
import models
import numpy as np

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--params', type=str, required=True)
parser.add_argument('--split', type=str, required=True)
parser.add_argument('--channel-sweep', action='store_true')
parser.add_argument('--channel-mask', type=str, required=False)
opts = parser.parse_args()
print(opts, file=sys.stderr)

model = models.construct_single_trunk_model()
params = u.load_params(opts.params)

if opts.channel_sweep:
    results = []
    for ch1 in range(0, 13):
        for ch2 in range(ch1 + 1, 13):
            dataset = data.dataset(split=opts.split, batch_size=32,
                                   channels_to_zero_out=[ch1, ch2])
            accuracy, mean_loss = u.accuracy_mean_loss(model, params, dataset)
            results.append((ch1, ch2, accuracy, mean_loss))
    for ch1, ch2, accuracy, mean_loss in sorted(results, key=lambda v: v[-1]):
        print("ch %02d %02d accuracy %0.3f mean_loss %0.3f" %
              (ch1, ch2, accuracy, mean_loss))
    exit()

if opts.channel_mask is not None:
    channel_mask = np.array(eval(opts.channel_mask))
    channels_to_zero_out = np.where(channel_mask != 1)
    dataset = data.dataset(split=opts.split, batch_size=32,
                           channels_to_zero_out=channels_to_zero_out)
else:
    dataset = data.dataset(split=opts.split, batch_size=32)

accuracy, mean_loss = u.accuracy_mean_loss(model, params, dataset)
print("accuracy %0.3f mean_loss %0.3f" % (accuracy, mean_loss))

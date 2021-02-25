import argparse
import sys
import util as u
import data
import models

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--params', type=str, required=True)
parser.add_argument('--split', type=str, required=True)
parser.add_argument('--channel-sweep', action='store_true')
opts = parser.parse_args()
print(opts, file=sys.stderr)

model = models.construct_single_trunk_model()
params = u.load_params(opts.params)

if opts.channel_sweep:
    for ch in range(13):
        dataset = data.dataset(split=opts.split, batch_size=32,
                               channels_to_zero_out=ch)
        print("ch", ch, "accuracy %0.3f" % u.accuracy(model, params, dataset))
else:
    dataset = data.dataset(split=opts.split, batch_size=32)
    print("accuracy %0.3f" % u.accuracy(model, params, dataset))

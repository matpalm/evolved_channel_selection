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
    results = []
    for ch in range(13):
        dataset = data.dataset(split=opts.split, batch_size=32,
                               channels_to_zero_out=ch)
        accuracy, mean_loss = u.accuracy_mean_loss(model, params, dataset)
        results.append((ch, accuracy, mean_loss))
    for ch, accuracy, mean_loss in sorted(results, key=lambda v: v[2]):
        print("ch %02d accuracy %0.3f mean_loss %0.3f" %
              (ch, accuracy, mean_loss))
else:
    dataset = data.dataset(split=opts.split, batch_size=32)
    accuracy, mean_loss = u.accuracy_mean_loss(model, params, dataset)
    print("accuracy %0.3f mean_loss %0.3f" % (accuracy, mean_loss))

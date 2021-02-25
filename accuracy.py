import argparse
import sys
import util as u
import data
import models

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--params', type=str, required=True)
parser.add_argument('--split', type=str, required=True)
opts = parser.parse_args()
print(opts, file=sys.stderr)

model = models.construct_single_trunk_model()
params = u.load_params(opts.params)
dataset = data.dataset(split=opts.split, batch_size=32)

print(u.accuracy(model, params, dataset))

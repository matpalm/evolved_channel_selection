import tensorflow_datasets as tfds
import numpy as np
from PIL import Image

dataset = tfds.load('eurosat/rgb', split='train', as_supervised=True)
for tfe_imgs, tfe_labels in dataset.batch(32):
    imgs = np.array(tfe_imgs)
    break

for b in range(32):
    i = Image.fromarray(imgs[b])
    for r in [64, 32, 16, 8]:
        downsample = i.resize((r, r), Image.NEAREST)
        explicit_upsample = downsample.resize((128, 128), Image.NEAREST)
        explicit_upsample.save("i%02d_x%02d.png" % (b, r))

import jax
from jax import jit
import jax.numpy as jnp
import models
import haiku as hk
import data as d


class Options:
    seed = 123


opts = Options()
host_rng = jax.random.PRNGKey(opts.seed ^ jax.host_id())
pod_rng = jax.random.PRNGKey(opts.seed - 1)

model = hk.without_apply_rng(hk.transform(models.single_trunk_model))

pod_rng, init_key = jax.random.split(pod_rng)
representative_input = jnp.zeros((1, 64, 64, 13))
params = model.init(init_key, representative_input)


@jit
def predict(params, x):
    logits = model.apply(params, x)
    return jnp.argmax(logits, axis=-1)


for x, labels in d.dataset(split='sample', batch_size=16):
    print(predict(params, x))
    break

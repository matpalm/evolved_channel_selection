import jax
from jax import jit, value_and_grad
import jax.numpy as jnp
import models
import haiku as hk
import data as d
import optax


class Options:
    seed = 123
    learning_rate = 1e-4
    batch_size = 32


opts = Options()
host_rng = jax.random.PRNGKey(opts.seed ^ jax.host_id())
pod_rng = jax.random.PRNGKey(opts.seed - 1)

model = hk.without_apply_rng(hk.transform(models.single_trunk_model))

pod_rng, init_key = jax.random.split(pod_rng)
representative_input = jnp.zeros((1, 64, 64, 13))
params = model.init(init_key, representative_input)

opt = optax.adam(opts.learning_rate)
opt_state = opt.init(params)


@jit
def predict(params, x):
    logits = model.apply(params, x)
    return jnp.argmax(logits, axis=-1)


def softmax_cross_entropy(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


def mean_cross_entropy(params, x, y_true):
    logits = model.apply(params, x)
    return jnp.mean(softmax_cross_entropy(logits, y_true))


@jit
def update(params, opt_state, x, y_true):
    loss, grads = value_and_grad(mean_cross_entropy)(params, x, y_true)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss.mean()


for i, (x, y_true) in enumerate(d.dataset(split='train',
                                          batch_size=opts.batch_size)):
    params, opt_state, loss = update(params, opt_state, x, y_true)
    if i % 100 == 0:
        y_pred = predict(params, x)
        num_correct = jnp.sum(jnp.equal(y_pred, y_true))
        print(i, loss, num_correct, y_true, y_pred)
    if i > 1000:
        break

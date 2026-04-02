"""Profile per-epoch computation costs."""
import time
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np
import optax
from jaxopt import OptaxSolver

from common.setup import get_one_tree_and_seq
from common.tree_func import *

# Simulate 16-leaf, 256 seq, 20 letters, 10 inits
n_leaves = 16
n_ancestors = 15
n_all = 31
seq_length = 256
n_letters = 20
init_count = 10

key = jax.random.PRNGKey(42)
initializer = jax.nn.initializers.kaiming_normal()

# Fake metadata (minimal for JIT)
metadata = {
    'n_all': n_all, 'n_leaves': n_leaves, 'n_ancestors': n_ancestors,
    'seq_length': seq_length, 'n_letters': n_letters, 'n_mutations': 50,
    'args': {'fix_seqs': False, 'fix_tree': False},
    'seq_temp': 0.5, 'lr': 0.1, 'lr_seq': 0.01, 'epochs': 100,
    'tLs': [0, 0.005, 10, 50], 'init_count': init_count,
    'exp_name': None, 'notes': None, 'tags': None, 'project': None,
}
metadata['args']['notes'] = None
metadata['args']['tags'] = None
metadata['args']['project'] = None

# Params
tree_params = {'t': initializer(key + 10, (init_count, n_all - 1, n_ancestors), jnp.float64)}
seq_slices = [initializer(key+i+10, (init_count, seq_length, n_letters), jnp.float64)
              for i in range(n_ancestors)]
seq_params = {'s': jnp.stack(seq_slices, axis=1)}

seqs = jax.random.normal(key, (n_all, seq_length, n_letters), dtype=jnp.float64)
sm = jnp.ones((n_letters, n_letters)) - jnp.identity(n_letters).astype(jnp.float64)

# Objectives
def inner_objective(seq_params, tree_params, data):
    seqs, temp, epoch = data
    return compute_loss_optimized(tree_params, seq_params, seqs, temp, epoch)

def outer_objective(tree_params, seq_params, data):
    seqs, temp, epoch = data
    return compute_loss_optimized(tree_params, seq_params, seqs, temp, epoch)

# Optimizers
seq_optimizer = OptaxSolver(opt=optax.adam(0.01), fun=inner_objective, maxiter=1)
vmap_seq_init = vmap(seq_optimizer.init_state, ({'s': 0}, {'t': 0}, None), 0)
seq_opt_state = vmap_seq_init(seq_params, tree_params, [seqs, metadata['tLs'][0], 0])
jitted_seq_update = jit(vmap(seq_optimizer.update, ({'s': 0}, 0, {'t': 0}, None), 0))

tree_optimizer = OptaxSolver(opt=optax.adam(0.1), fun=outer_objective)
vmap_tree_init = vmap(tree_optimizer.init_state, ({'t': 0}, {'s': 0}, None), 0)
tree_opt_state = vmap_tree_init(tree_params, seq_params, [seqs, metadata['tLs'][0], 0])
jitted_tree_update = jit(vmap(tree_optimizer.update, ({'t': 0}, 0, {'s': 0}, None), 0))

vmap_compute_detailed = jit(vmap(compute_detailed_loss_optimized, ({'t':0}, {'s': 0}, None, None, None, None), 0))

def bench(name, fn, n=20):
    # warmup
    for _ in range(3):
        fn()
    jax.block_until_ready(fn())  # ensure GPU sync
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)
    avg = np.mean(times) * 1000
    std = np.std(times) * 1000
    print(f"{name:45s}  {avg:8.2f} ms  (+/- {std:.2f} ms)")

print(f"\n--- Per-epoch compute costs (l={n_leaves}, sl={seq_length}, ic={init_count}) ---\n")
print(f"Device: {jax.devices()[0]}\n")

# Tree update
bench("Tree optimizer update (vmap+jit)",
      lambda: jitted_tree_update(tree_params, tree_opt_state, seq_params, [seqs, metadata['tLs'][0], 0]))

# Seq update
bench("Seq optimizer update (vmap+jit)",
      lambda: jitted_seq_update(seq_params, seq_opt_state, tree_params, [seqs, metadata['tLs'][0], 0]))

# Detailed loss (for logging)
bench("Detailed loss computation (vmap+jit)",
      lambda: vmap_compute_detailed(tree_params, seq_params, seqs, metadata['tLs'][0], sm, 0))

# argmin + get_one_tree_and_seq
cost, _, _, _ = vmap_compute_detailed(tree_params, seq_params, seqs, metadata['tLs'][0], sm, 0)
bench("argmin + get_one_tree_and_seq",
      lambda: get_one_tree_and_seq(tree_params, seq_params, jnp.argmin(cost)))

# Full epoch (tree + seq + cost)
def full_epoch():
    tp, ts_state = jitted_tree_update(tree_params, tree_opt_state, seq_params, [seqs, metadata['tLs'][0], 0])
    sp, ss_state = jitted_seq_update(seq_params, seq_opt_state, tp, [seqs, metadata['tLs'][0], 0])
    c, cs, tf, l = vmap_compute_detailed(tp, sp, seqs, metadata['tLs'][0], sm, 0)
    return c

bench("Full epoch (tree + seq + cost)", full_epoch)

# Scale test: ic=100
print(f"\n--- Same but with ic=100 ---\n")
init_count_100 = 100
tree_params_100 = {'t': initializer(key + 10, (init_count_100, n_all - 1, n_ancestors), jnp.float64)}
seq_slices_100 = [initializer(key+i+10, (init_count_100, seq_length, n_letters), jnp.float64)
                  for i in range(n_ancestors)]
seq_params_100 = {'s': jnp.stack(seq_slices_100, axis=1)}

seq_opt_state_100 = vmap_seq_init(seq_params_100, tree_params_100, [seqs, metadata['tLs'][0], 0])
tree_opt_state_100 = vmap_tree_init(tree_params_100, seq_params_100, [seqs, metadata['tLs'][0], 0])

def full_epoch_100():
    tp, ts_state = jitted_tree_update(tree_params_100, tree_opt_state_100, seq_params_100, [seqs, metadata['tLs'][0], 0])
    sp, ss_state = jitted_seq_update(seq_params_100, seq_opt_state_100, tp, [seqs, metadata['tLs'][0], 0])
    c, cs, tf, l = vmap_compute_detailed(tp, sp, seqs, metadata['tLs'][0], sm, 0)
    return c

bench("Full epoch ic=100 (tree + seq + cost)", full_epoch_100)
print()

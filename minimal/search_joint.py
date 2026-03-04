"""Joint optimization: tree + seq params optimized together with a single optimizer."""

import optax
from jaxopt import OptaxSolver
from jax import jit, vmap

from common import *

# ── Objective ──

def objective(params, data):
    seqs, metadata, temp, epoch = data
    tree_params = {'t': params['t']}
    seq_params = {k: v for k, v in params.items() if k != 't'}
    return compute_loss_optimized(tree_params, seq_params, seqs, metadata, temp, epoch)

# ── Setup ──

args = vars(parse_args())
args = sanity_check(args)
metadata = build_metadata(args)
setup_device(args)

print(pretty_print_dict(metadata))

seqs, gt_seqs, tree, base_tree, sm, sankoff_cost, gt_cost = generate_data(metadata)
clear_metadata_for_jit(metadata, args)
tree_params, seq_params = init_params(metadata, seqs, metadata['n_leaves'], metadata['n_ancestors'], metadata['init_count'])

if args['initialize_tree']:
    tree_params['t'] = tree[0:-1, metadata['n_leaves']:] * 100

# Merge into single param dict
params = {'t': tree_params['t'], **seq_params}

# ── Optimizer ──

optimizer = OptaxSolver(opt=optax.adam(metadata['lr']), fun=objective)
vmap_init = vmap(optimizer.init_state, (0, None), 0)

# vmap over init_count dimension — need vmap spec for merged params
vmap_keys = {k: 0 for k in params.keys()}
vmap_init = vmap(optimizer.init_state, (vmap_keys, None), 0)
opt_state = vmap_init(params, [seqs, metadata, metadata['tLs'][0], 0])
jitted_update = jit(vmap(optimizer.update, (vmap_keys, 0, None), 0))

# ── Update step ──

def update_step(tree_params, seq_params, seqs, metadata, epoch):
    nonlocal_state = update_step.state

    # Merge
    merged = {'t': tree_params['t'], **seq_params}

    merged, nonlocal_state['opt'] = jitted_update(
        merged, nonlocal_state['opt'], [seqs, metadata, metadata['tLs'][0], epoch]
    )

    update_step.state = nonlocal_state

    # Split back
    new_tree_params = {'t': merged['t']}
    new_seq_params = {k: v for k, v in merged.items() if k != 't'}
    return new_tree_params, new_seq_params

update_step.state = {'opt': opt_state}

# ── Run ──

best_cost, best_tree, best_seq = run_search(
    update_step, tree_params, seq_params, seqs, metadata, sm,
    base_tree=base_tree, tree=tree, sankoff_cost=sankoff_cost, gt_cost=gt_cost
)

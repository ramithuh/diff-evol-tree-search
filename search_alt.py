"""Alternating optimization: tree update -> seq update loop (no implicit diff)."""

import optax
from jaxopt import OptaxSolver
from jax import jit, vmap

from common import *

# ── Objectives ──

def inner_objective(seq_params, tree_params, data):
    seqs, temp, epoch = data
    return compute_loss_optimized(tree_params, seq_params, seqs, temp, epoch)

def outer_objective(tree_params, seq_params, data):
    seqs, temp, epoch = data
    return compute_loss_optimized(tree_params, seq_params, seqs, temp, epoch)

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

# ── Optimizers ──

alt_interval = args['alternate_interval'] if args['alternate_interval'] is not None else 5

seq_optimizer = OptaxSolver(opt=optax.adam(metadata['lr_seq']), fun=inner_objective, maxiter=alt_interval)
vmap_seq_init = vmap(seq_optimizer.init_state, (0, 0, None), 0)
seq_opt_state = vmap_seq_init(seq_params, tree_params, [seqs, metadata['tLs'][0], 0])
jitted_seq_update = jit(vmap(seq_optimizer.update, (0, 0, 0, None), 0))

tree_optimizer = OptaxSolver(opt=optax.adam(metadata['lr']), fun=outer_objective)
vmap_tree_init = vmap(tree_optimizer.init_state, (0, 0, None), 0)
tree_opt_state = vmap_tree_init(tree_params, seq_params, [seqs, metadata['tLs'][0], 0])
jitted_tree_update = jit(vmap(tree_optimizer.update, (0, 0, 0, None), 0))

# ── Update step ──

def update_step(tree_params, seq_params, seqs, metadata, epoch):
    nonlocal_state = update_step.state

    tree_params, nonlocal_state['tree'] = jitted_tree_update(
        tree_params, nonlocal_state['tree'], seq_params, [seqs, metadata['tLs'][0], epoch]
    )

    for _ in range(alt_interval):
        seq_params, nonlocal_state['seq'] = jitted_seq_update(
            seq_params, nonlocal_state['seq'], tree_params, [seqs, metadata['tLs'][0], epoch]
        )

    update_step.state = nonlocal_state
    return tree_params, seq_params

update_step.state = {'tree': tree_opt_state, 'seq': seq_opt_state}

# ── Run ──

best_cost, best_tree, best_seq = run_search(
    update_step, tree_params, seq_params, seqs, metadata, sm,
    base_tree=base_tree, tree=tree, sankoff_cost=sankoff_cost, gt_cost=gt_cost
)

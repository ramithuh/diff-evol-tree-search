import jax
import jax.numpy as jnp
from jax import jit, vmap
import plotly.express as px
import os

from .tree_func import *
from .vis_utils import *
from .setup import (
    print_critical_info, print_success_info, print_bold_info,
    generate_vmap_keys, get_one_tree_and_seq,
)


def run_search(update_step_fn, tree_params, seq_params, seqs, metadata, sm,
               base_tree=None, tree=None, sankoff_cost=None, gt_cost=None,
               save_dir="figures"):
    """
    Main search loop shared by alternating and bilevel modes.

    Args:
        update_step_fn: callable(tree_params, seq_params, seqs, metadata, epoch) -> (tree_params, seq_params)
        tree_params, seq_params: initial parameters
        seqs: one-hot encoded leaf+ancestor sequences
        metadata: experiment config dict
        sm: substitution matrix
        base_tree: base tree (zeros or GT if fix_tree)
        tree: groundtruth tree (for visualization)
        sankoff_cost: Sankoff baseline cost (if computed)
        gt_cost: groundtruth traversal cost
        save_dir: directory to save figure outputs
    """
    args = metadata['args']
    n_leaves = metadata['n_leaves']
    n_all = metadata['n_all']
    vis_interval = args.get('vis_interval', 200)
    log_interval = args.get('log_interval', 200)

    os.makedirs(save_dir, exist_ok=True)

    vmap_keys = generate_vmap_keys(seq_params)
    vmap_compute_detailed = jit(vmap(
        compute_detailed_loss_optimized,
        ({'t':0}, vmap_keys, None, None, None, None, None), 0
    ))

    best_ans = 1e9
    best_seq = None
    best_tree = None
    pos = 0

    # Initial params extraction for visualization
    params = get_one_tree_and_seq(tree_params, seq_params, 0)

    for epoch in range(metadata['epochs']):

        # ── Visualization ──
        if epoch % vis_interval == 0:
            if args['fix_tree'] and base_tree is not None:
                t_ = base_tree
            else:
                t_ = update_tree(params, epoch, metadata['tLs'][0])
            t_d = discretize_tree_topology(t_, n_all)
            tree_fig = show_graph_with_labels(t_d, n_leaves, True)
            tree_fig.savefig(os.path.join(save_dir, f"tree_epoch_{epoch}.png"), dpi=100, bbox_inches='tight')

            if args['fix_seqs']:
                seqs_ = seqs
            else:
                seqs_ = update_seq(params, seqs, metadata['seq_temp'])
            seq_fig = px.imshow(jnp.argmax(seqs_, axis=2), text_auto=True)
            seq_fig.write_html(os.path.join(save_dir, f"seq_epoch_{epoch}.html"))

        # ── Parameter update ──
        tree_params, seq_params = update_step_fn(tree_params, seq_params, seqs, metadata, epoch)

        # ── Cost computation ──
        cost, cost_surrogate, tree_force_loss, loss = vmap_compute_detailed(
            tree_params, seq_params, seqs, metadata, metadata['tLs'][0], sm, epoch
        )
        pos = jnp.argmin(cost)

        params = get_one_tree_and_seq(tree_params, seq_params, pos)

        if cost.min() < best_ans:
            if epoch % 20 == 0:
                print_success_info(
                    "Found better solution at epoch %d with cost %f from init %d (delta = %d)\n"
                    % (epoch, cost.min(), pos, cost.max()-cost.min())
                )
            best_ans = cost.min()

            t_ = update_tree(params, epoch, metadata['tLs'][0])
            best_tree = discretize_tree_topology(t_, n_all)
            best_seq = update_seq(params, seqs, metadata['seq_temp'])

        # ── Logging ──
        if epoch % log_interval == 0:
            print_bold_info(f"epoch {epoch}")
            print("  tLs={:.3f}  surrogate={:.3f}  hard_cost={:.3f}  loss={:.3f}".format(
                metadata['tLs'][0],
                cost_surrogate[pos].item(),
                cost[pos].item(),
                loss[pos].item()
            ))

        # ── Tree loss schedule update ──
        if epoch % metadata['tLs'][3] == 0:
            metadata['tLs'][0] = min(metadata['tLs'][2], metadata['tLs'][0] + metadata['tLs'][1])

    # ── Final summary ──
    print_success_info("Optimization done!\n")
    print_success_info("Final cost: {:.5f}\n".format(cost[pos]))
    print_success_info("Best cost encountered: {:.5f}\n".format(best_ans))

    if args['fix_tree'] and sankoff_cost is not None:
        print_success_info("Sankoff cost for groundtruth tree: {:.5f}\n".format(float(sankoff_cost)))
        target_cost = sankoff_cost
    elif args['fix_seqs'] and gt_cost is not None:
        print_success_info("Groundtruth tree cost: {:.5f}\n".format(float(gt_cost)))
        target_cost = gt_cost
    else:
        target_cost = 0

    if target_cost > 0 and abs(target_cost - best_ans) == 0:
        print_success_info("Optimization succeeded! Reached groundtruth!\n")

    # Save best tree
    if best_tree is not None:
        best_tree_fig = show_graph_with_labels(best_tree, n_leaves, True)
        best_tree_fig.savefig(os.path.join(save_dir, "best_tree.png"), dpi=100, bbox_inches='tight')

    if best_seq is not None:
        best_seq_fig = px.imshow(jnp.argmax(best_seq, axis=2), text_auto=True)
        best_seq_fig.write_html(os.path.join(save_dir, "best_seq.html"))

    return best_ans, best_tree, best_seq

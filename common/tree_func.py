import jax
import jax.nn as nn
import jax.numpy as jnp

from jax import jit
from functools import partial
from typing import Dict, List
from jaxtyping import Array, Float

@partial(jit, static_argnums=1)
def discretize_tree_topology(tree : Float[Array, "nodes nodes"], n_nodes : int) -> Float[Array, "nodes nodes"]:
    t_argmax = jnp.argmax(tree,axis = 1)
    t_q = nn.one_hot(t_argmax, n_nodes)
    return t_q

@jit
def update_tree(params : Dict[str, Array], epoch : int = 0 ,  temp : Float = 1 ) -> Float[Array, "nodes nodes"]:
    tree = params['t']

    n_all = tree.shape[0] + 1
    n_ancestors = tree.shape[1]
    n_leaves = n_all - n_ancestors

    key = jax.random.PRNGKey(epoch)
    gumbel_noise = jax.random.gumbel(key, (tree.shape[0], tree.shape[1]))

    perturbed_tree_params = tree + gumbel_noise*0.0

    inf_matrix = -jnp.ones((n_all, n_all))*jnp.inf

    lower_indices = jnp.triu_indices(n_ancestors)
    shifted_lower_indices = (lower_indices[0] + n_ancestors, lower_indices[1] + n_leaves )

    inf_matrix = inf_matrix.at[:n_ancestors, n_leaves:].set(perturbed_tree_params[:n_ancestors])
    inf_matrix = inf_matrix.at[shifted_lower_indices].set(perturbed_tree_params[n_ancestors:, :][lower_indices])
    inf_matrix = inf_matrix.at[-1,-1].set(1)

    return jax.nn.softmax(inf_matrix, axis = 1)

@jit
def update_seq(params : Dict[str, Array], seqs : Float[Array, "nodes letters"], temperature : Float = 1 ) -> Float[Array, "nodes letters"]:
    n_all = seqs.shape[0]
    n_leaves = (n_all + 1)//2

    for i in range(0, n_all - n_leaves):
        key = str(i)
        seq      = nn.softmax(params[key]*temperature)
        seqs = seqs.at[- n_leaves + i + 1].set(seq)

    return seqs

@jit
def enforce_graph(t_ : Float[Array, "nodes nodes"], s : Float, metadata = None, verbose = False) -> List[Float]:
    n_all = t_.shape[0]
    n_leaves = (n_all + 1)//2
    n_ancestors = n_all - n_leaves

    ancestor_columns = t_[:-1, n_all - n_ancestors: n_all]

    tree_force_loss = jnp.sum(jnp.power(s*jnp.abs(jnp.sum(ancestor_columns, axis = 0) - 2),2))

    if(verbose):
        print("bifurcating tree_forcing_loss = ", tree_force_loss)

    return tree_force_loss

@jit
def compute_surrogate_cost(sequences : Float[Array, "nodes seq_length letters"], tree : Float[Array, "nodes nodes"]) -> Float:
    seq_onehot = jnp.transpose(sequences, (2,0,1))
    sel        = jnp.expand_dims(tree, axis = 0)
    selection  = jnp.matmul(sel, seq_onehot)
    out = jnp.transpose(selection, (1,2,0))
    ans = (jnp.sum(abs((out - sequences)), axis = [1,2])/(2)).sum()
    return ans

@jit
def compute_cost(sequences : Float[Array, "nodes seq_length letters"], tree : Float[Array, "nodes nodes"], sm) -> Float:
    selection_ = jnp.matmul(discretize_tree_topology(tree, tree.shape[0]), jnp.argmax(sequences, axis = 2))
    sequences_ = jnp.argmax(sequences, axis = 2)
    ans = sm[jnp.round(selection_).astype(jnp.int64),jnp.round(sequences_).astype(jnp.int64)].sum()
    return ans

def compute_loss(params : Dict[str, Array], seqs : Float[Array, "nodes length letters"], base_tree : Float[Array, "nodes nodes"], metadata : Dict, temp : Float, epoch : int = 0, verbose = False) -> Float:
    n_leaves = metadata['n_leaves']
    n_all    = metadata['n_all']

    if(metadata['args']['fix_seqs']):
        seqs_ = seqs
    else:
        seqs_ = update_seq(params, seqs, temp )

    if(metadata['args']['fix_tree']):
        t_ = base_tree
    else:
        t_ = update_tree(params, epoch)

    cost_surrogate = compute_surrogate_cost(seqs_,t_)
    tree_force_loss = enforce_graph(t_,10,metadata)
    loss = cost_surrogate + temp*(tree_force_loss)

    if(verbose):
        sm   = jnp.ones((metadata['n_letters'],metadata['n_letters'])) - jnp.identity(metadata['n_letters']).astype(jnp.bfloat16)
        cost = compute_cost(seqs_,t_, sm)
        return cost, cost_surrogate, tree_force_loss, loss

    return loss

@jit
def compute_loss_optimized(tree_params : Dict[str, Array], seq_params : Dict[str, Array], seqs : Float[Array, "nodes length letters"], metadata : Dict, temp : Float, epoch : int) -> Float:
    n_leaves = metadata['n_leaves']
    n_all    = metadata['n_all']

    seqs_ = update_seq(seq_params, seqs, temp )
    t_ = update_tree(tree_params, epoch, temp)

    cost_surrogate = compute_surrogate_cost(seqs_,t_)
    tree_force_loss = enforce_graph(t_,10,metadata)
    loss = cost_surrogate + temp*(tree_force_loss)

    return loss

@jit
def compute_detailed_loss_optimized(tree_params : Dict[str, Array], seq_params : Dict[str, Array], seqs : Float[Array, "nodes length letters"], metadata : Dict, temp : Float, sm : Float[Array, "letters letters"], epoch : int = 0) -> Float:
    n_leaves = metadata['n_leaves']
    n_all    = metadata['n_all']

    seqs_ = update_seq(seq_params, seqs, temp )
    t_ = update_tree(tree_params, epoch, temp)

    cost_surrogate = compute_surrogate_cost(seqs_,t_)
    tree_force_loss = enforce_graph(t_,10,metadata)
    loss = cost_surrogate + temp*(tree_force_loss)

    cost = compute_cost(seqs_,t_, sm)

    return cost, cost_surrogate, tree_force_loss, loss

import jax
import numpy as np
import jax.nn as nn
import jax.numpy as jnp

from typing import Dict, List
from jaxtyping import Array, Float
from functools import partial

def softmin(x, epsilon = 1, axis = 0):
    c = jnp.max(-x*epsilon, axis = axis, keepdims=True)
    return -(jnp.log(jnp.mean(jnp.exp(-x*epsilon - c), axis = axis, keepdims = True ) - 1e-6) + c/epsilon)[...,-1]

def run_dp(adj : Float[Array, "nodes nodes"], dp, dp_nodes, seq, cost_mat, n_letters, verbose = False):
    n_all    = adj.shape[0]
    n_leaves = (n_all + 1)//2

    for i in range(0, n_leaves):
        dp =  dp.at[i,seq[i].astype(int)].set(0)

    for node in range(n_leaves, n_all):
        children = jnp.where(adj[:,node] == 1)[0]

        if(verbose):
            print(f"at node {node+1} children are : {children}")

        total_cost = 0
        nodes = []
        for child in children:
            cur_node = int(child)

            cost_array = cost_mat[::][::] + dp[cur_node][::]
            cost = jnp.min(cost_array, axis = 1)
            char = jnp.argmin(cost_array, axis = 1)

            nodes.append([cur_node, char])
            total_cost += cost

        dp = dp.at[node,::].set(total_cost)

    return dp, dp_nodes

vectorized_dp = jax.vmap(run_dp, (None, 0, 0, 1, None, None, None), 0)

def run_sankoff(adj, cost_mat, seq, metadata, return_path = False):
    adj = adj.at[-1,-1].set(0)

    adj = adj.astype(jnp.float64)
    seq = seq.astype(jnp.float64)
    cost_mat = cost_mat.astype(jnp.float64)

    n_letters = metadata['n_letters']
    n_leaves  = metadata['n_leaves']

    dp_nodes = jnp.zeros((seq.shape[1], metadata['n_all'], n_letters, 4)).astype(jnp.float64)
    dp       = jnp.ones((seq.shape[1], metadata['n_all'], n_letters)).astype(jnp.float64)*1e5

    dp, connections = vectorized_dp(adj, dp, dp_nodes, seq, cost_mat, n_letters, False)

    seq_chars = jnp.zeros((seq.shape[1], metadata['n_all'],1)).astype(jnp.float64)
    found_seq = seq.copy().astype(jnp.float64)

    if(return_path):
        node = adj.shape[0] - 1

        if(seq.shape[0] <=  1024):
            for i in range(0,seq.shape[1]):
                letter = jnp.argmin(dp[i,node,:]).astype(int)

                chars = backtrack_dp(node, letter, seq_chars[i], connections[i], n_leaves).reshape(metadata['n_all'],)

                found_seq = found_seq.at[metadata['n_leaves']:,i].set(chars[metadata['n_leaves']:])

    return found_seq, dp, dp[:, -1].min(axis = 1).sum()

def backtrack_dp(node, letter, seq_chars, connections, n_leaves):
    if(node < n_leaves):
        return seq_chars

    seq_chars = seq_chars.at[node].set(letter)

    child = connections[node,letter]

    seq_chars = backtrack_dp(child[0].astype(int), child[1].astype(int), seq_chars, connections, n_leaves)
    seq_chars = backtrack_dp(child[2].astype(int), child[3].astype(int), seq_chars, connections, n_leaves)

    return seq_chars

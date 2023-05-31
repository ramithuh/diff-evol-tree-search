import jax
import numpy as np
import jax.nn as nn
import jax.numpy as jnp

from typing import Dict, List
from jaxtyping import Array, Float
from functools import partial

from ott.tools import soft_sort
# softranks = soft_sort.ranks#jax.jit(soft_sort.ranks)
softranks = jax.jit(soft_sort.ranks, static_argnums=(2)) 

def softmin(x, epsilon = 1, axis = 0):
    c = jnp.max(-x*epsilon, axis = axis, keepdims=True)
    return -(jnp.log(jnp.mean(jnp.exp(-x*epsilon - c), axis = axis, keepdims = True ) - 1e-6) + c/epsilon)[...,-1]

# @partial(jax.jit, static_argnums=[5,6]) 
def run_dp(adj : Float[Array, "nodes nodes"], dp, dp_nodes, seq, cost_mat, n_letters, verbose = False):
    '''
        Run sankoff algorithm given the tree topology and the leaf sequences
        
        Args:
            adj: adjacency matrix of the tree
            dp : blank dp table
            dp_nodes : blank dp table for storing the nodes
            seq : leaf sequences
            args : metadata
            verbose : print the progress of the algorit hm
            
        Returns:
            dp : filled dp table
            dp_nodes : node information for backtracking
    '''
    
    n_all    = adj.shape[0]
    n_leaves = (n_all + 1)//2

    for i in range(0, n_leaves): # initailize the dp table
        dp =  dp.at[i,seq[i].astype(int)].set(0)


    for node in range(n_leaves, n_all): ## loop all the ancestors in order
        children = jnp.where(adj[:,node] == 1)[0] ## find the children of the current node

        if(verbose):
            print(f"at node {node+1} children are : {children}")

        #for i in range(n_letters): # consider all the possible letters
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

        #below parts should be generalized for trees other than bifurcating ones
        # dp_nodes = dp_nodes.at[node, ::, 0].set(nodes[0][0])
        # dp_nodes = dp_nodes.at[node, ::, 1].set(nodes[0][1])

        # dp_nodes = dp_nodes.at[node, ::, 2].set(nodes[1][0])
        # dp_nodes = dp_nodes.at[node, ::, 3].set(nodes[1][1])

    return dp, dp_nodes

vectorized_dp      = jax.vmap(run_dp, (None, 0, 0, 1, None, None, None), 0)

def run_sankoff(adj, cost_mat, seq, metadata, return_path = False):
    adj = adj.at[-1,-1].set(0) ##manually remove self connection (convention used in the train script)
    
    ## ensure types are float64
    adj = adj.astype(jnp.float64)
    seq = seq.astype(jnp.float64)
    cost_mat = cost_mat.astype(jnp.float64)
    
    
    n_letters = metadata['n_letters']
    n_leaves  = metadata['n_leaves']
    
    ##empty dp tables
    dp_nodes = jnp.zeros((seq.shape[1], metadata['n_all'], n_letters, 4)).astype(jnp.float64)
    dp       = jnp.ones((seq.shape[1], metadata['n_all'], n_letters)).astype(jnp.float64)*1e5
    
    # connections = dp_nodes.copy()
        
    dp, connections = vectorized_dp(adj, dp, dp_nodes, seq, cost_mat, n_letters, False)
    # print("ran dp for all positions through vmap")


    # print(dp)
    # for i in range(0,seq.shape[1]):
    #     print("ran dp for position", i)
    #     ans = run_dp(adj, dp[i], dp_nodes[i], seq[:,i], cost_mat, n_letters, False)
        
    #     dp       = dp.at[i].set(ans[0])
    #     connections = connections.at[i].set(ans[1])
        
    
    seq_chars = jnp.zeros((seq.shape[1], metadata['n_all'],1)).astype(jnp.float64)
    found_seq = seq.copy().astype(jnp.float64)
    
    if(return_path):
        node = adj.shape[0] - 1

        if(seq.shape[0] <=  1024): # if too many sequences don't backtrack and find best ancestors, cost is enough
            for i in range(0,seq.shape[1]):
                letter = jnp.argmin(dp[i,node,:]).astype(int)

                chars = backtrack_dp(node, letter, seq_chars[i], connections[i], n_leaves).reshape(metadata['n_all'],)
                
                found_seq = found_seq.at[metadata['n_leaves']:,i].set(chars[metadata['n_leaves']:])
                # print(".", end = "")

    
    return found_seq, dp, dp[:, -1].min(axis = 1).sum()

def backtrack_dp(node, letter, seq_chars, connections, n_leaves):
    '''
        Derive the ancestor sequences from the dp table and the node information
        
        Args:
            node : current node
            letter : letter used for current node
            seq_chars : blank sequence table (leaf sequences are already filled)
            connections : node information
            args : metadata
        
        Return : 
            seq_chars : filled sequence table (computed ancestor sequences)
    '''

    if(node < n_leaves): #break if it's a leaf
        return seq_chars


    seq_chars = seq_chars.at[node].set(letter)

    child = connections[node,letter]

    seq_chars = backtrack_dp(child[0].astype(int), child[1].astype(int), seq_chars, connections, n_leaves)
    seq_chars = backtrack_dp(child[2].astype(int), child[3].astype(int), seq_chars, connections, n_leaves)

    return seq_chars

'''
    Functions below are work in progress
'''

def run_diff_dp(adj : Float[Array, "nodes nodes"], dp, seq, cost_mat, n_letters, epsilon = 1e-5, verbose = True):
    '''
        Run sankoff algorithm given the tree topology and the leaf sequences (work in progress)
        
        Args:
            adj: adjacency matrix of the tree
            dp : blank dp table
            seq : leaf sequences
            args : metadata
            verbose : print the progress of the algorit hm
            
        Returns:
            dp : filled dp table
            dp_nodes : <TODO>
    '''
    
    n_all    = adj.shape[0]
    n_leaves = (n_all + 1)//2

    for i in range(0, n_leaves): # initailize the dp table
        dp =  dp.at[i,seq[i].astype(int)].set(0) ## sequences are fixed!! so no need to propogate the gradients till then


    for node in range(n_leaves, n_all): ## loop all the ancestors in order
        
        ## consider topologically feasible candidates (slicing -> :node)
        prob_children    = jax.nn.softmax(adj[:,node][:node]*5)*2 #softranks(adj[:,node][:node], epsilon = epsilon)/epsilon)*2 
        children_onehot  = jnp.eye(node,node).astype(jnp.float64)*prob_children
        
        #print(prob_children)
        delta = 1e1*(adj[:,node][:node].sum() - 2)**2
        
        mask = jnp.matmul(children_onehot, jnp.ones((node,dp.shape[1])).astype(jnp.float64)) #mask to supress rows from the dp table

        if(verbose):
            print(f"at node {node+1} children are : {adj[:,node][:node]}")
            print(f" probable children are -> {prob_children}")
            print("_____")
        
        dp_sel = jnp.matmul(children_onehot,dp[:node]) ##lookup dp table's first node-1 rows (topologically feasible candidates)
        node_mix = cost_mat[::,::] + dp_sel[::, None]
        
        ans = node_mix * mask[::, None]
        ans = softmin(ans, epsilon, axis = 2) #-nn.logsumexp(-ans/epsilon, axis = 2)*epsilon
        ans = jnp.sum(ans, axis = 0) + delta #penalize more if connections are not rigid

        dp = dp.at[node,::].set(ans)

    return dp

vectorized_diff_dp = jax.vmap(run_diff_dp, (None, 0, 1, None, None, None, None), 0)

def run_diff_sankoff(adj, cost_mat, seq, metadata, epsilon = 1e-5):
    '''
        (Work in progress -> making sankoff algorithm differentiable)
    '''
    adj = adj.at[-1,-1].set(0) ##manually remove self connection (convention used in the train script)
    
    ## ensure types are float64
    adj = adj.astype(jnp.float64)
    seq = seq.astype(jnp.float64)
    cost_mat = cost_mat.astype(jnp.float64)
    
    n_letters = metadata['n_letters']
    n_leaves  = metadata['n_leaves']
    
    ##empty dp table
    dp       = jnp.ones((seq.shape[1], metadata['n_all'], n_letters)).astype(jnp.float64)*1e5
    
    # connections = dp_nodes.copy()
        
    dp = vectorized_diff_dp(adj, dp, seq, cost_mat, n_letters, epsilon, False)

    return dp, softmin(dp[:, -1], epsilon, axis = 1).sum() #dp[:, -1].min(axis = 1).sum()

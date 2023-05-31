import jax
import jax.nn as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt

import jax.scipy.linalg as slin
import jax.numpy.linalg as la

from jax import jit
from functools import partial
from typing import Dict, List
from jaxtyping import Array, Float

from modules.upgma import differentiable_upgma
from modules.sankoff import run_diff_sankoff
from modules.sankoff import run_sankoff

@partial(jit, static_argnums=1)
def discretize_tree_topology(tree : Float[Array, "nodes nodes"], n_nodes : int) -> Float[Array, "nodes nodes"]:
    '''
        Returns the one-hot encoding of the given soft tree topology

        Args:
            tree : soft tree topology (softmax applied over the rows)
            n_nodes : number of total nodes in the tree
        Returns:
            t_q : one-hot encoding of the tree topology 
    '''

    t_argmax = jnp.argmax(tree,axis = 1)
    t_q = nn.one_hot(t_argmax, n_nodes)
    
    return t_q

@jit
def update_tree(params : Dict[str, Array], epoch : int = 0 ,  temp : Float = 1 ) -> Float[Array, "nodes nodes"]:
    '''
        Uses base_tree and returns the updated soft tree topology
        Particularly updates the ancestor probabilities in the trees

        Args:
            params : params['t'] denotes the trainable params; (ancestors)
                     params['t'] -> (n_nodes-1, n_ancestors)

            epoch : epoch number
            temp : temperature for softmax
            
        Returns:
            t : updated soft tree topology
                t -> (n_nodes, n_nodes)
    '''
    tree = params['t']#.sum(axis = 0)
    
    n_all = tree.shape[0] + 1 #(adding +1 because params doesn't have the root node)
    n_ancestors = tree.shape[1]
    n_leaves = n_all - n_ancestors
    
    key = jax.random.PRNGKey(epoch) # generate a random key
    gumbel_noise = jax.random.gumbel(key, (tree.shape[0], tree.shape[1])) # generate gumbel noise

    perturbed_tree_params = tree + gumbel_noise*0.0 # add gumbel noise to the tree params

    inf_matrix = -jnp.ones((n_all, n_all))*jnp.inf
                    
    # generating the upper indices of the lower square matrix
    lower_indices = jnp.triu_indices(n_ancestors)
    shifted_lower_indices = (lower_indices[0] + n_ancestors, lower_indices[1] + n_leaves )

    inf_matrix = inf_matrix.at[:n_ancestors, n_leaves:].set(perturbed_tree_params[:n_ancestors])
    inf_matrix = inf_matrix.at[shifted_lower_indices].set(perturbed_tree_params[n_ancestors:, :][lower_indices])
    inf_matrix = inf_matrix.at[-1,-1].set(1)
    
    return jax.nn.softmax(inf_matrix, axis = 1)

@jit
def update_tree_sankoff(params : Dict[str, Array], base_tree : Float[Array, "nodes nodes"], epoch : int = 0 ) -> Float[Array, "nodes nodes"]:
    '''
        Uses base_tree and returns the updated  tree topology
        Doens't use softmax over the ancestors
        
        Args:
            params : params['t'] denotes the trainable params; (ancestors)
                     params['t'] -> (n_nodes, n_ancestors)

            base_tree : base tree topology
                        base_tree -> (n_nodes, n_nodes)
        Returns:
            t : updated soft tree topology
                t -> (n_nodes, n_nodes)
    '''

    n_leaves = (base_tree.shape[1] - params['t'].shape[1])
    
    t = base_tree.at[0:-1,n_leaves:].set(params['t'])
    t = t.at[:,:].set(abs(jnp.triu(t,1))) ## enforce upper triangularity (above main diagonal)
    
    # #custom softmax over the topologically ordered ancestors
    # for i in range(0, t.shape[0]-1): ## loop until last node (excluding root)
    #     pos = max(i+1, n_leaves)
        
    #     noisy_t = t[i, pos:]

    #     ## softmax over the topologically ordered ancestors 
    #     prob = noisy_t #jax.nn.softmax(noisy_t) # nn.softmax(t[i, pos:])

    #     t = t.at[i, pos:].set(prob)
    
    
    t = t.at[-1,-1].set(1) ### need to add this (so that the last sequence is matched with itself. i.e. root) 
    
    return t

@jit
def update_seq(params : Dict[str, Array], seqs : Float[Array, "nodes letters"], temperature : Float = 1 ) -> Float[Array, "nodes letters"]:
    '''
        Updates the ancestor sequences using the trainable params

        Args:
            params : params['n'] denotes the nth ancestor sequence
    '''


    n_all = seqs.shape[0]
    n_leaves = (n_all + 1)//2

    for i in range(0, n_all - n_leaves):
        key = str(i) #chr(97+i + n_leaves)
        
        # print(params[key].shape)
        # print(seq.shape)
        # break
        seq      = nn.softmax(params[key]*temperature)
        #hard_seq = (seq > 0.5).astype(int)
        
        seqs = seqs.at[- n_leaves + i + 1].set(seq)
        #seqs = seqs.at[- n_leaves + i + 1].set(jax.lax.stop_gradient(hard_seq - seq) + seq)
    
#     seq_1 = nn.sigmoid(temperature*params['d'])
#     seq_2 = nn.sigmoid(temperature*params['e'])
    
#     hard_seq_1 = (seq_1 > 0.5).astype(int)
#     hard_seq_2 = (seq_2 > 0.5).astype(int)
    
    
    
    
#     seqs = seqs.at[-2].set(seq_1)
#     seqs = seqs.at[-1].set(seq_2)
    
    return seqs

@jit
def enforce_graph(t_ : Float[Array, "nodes nodes"], s : Float, metadata = None, verbose = False) -> List[Float]:
    ''' 

        Enforces constraints such that,
            1) the tree is bifurcating
            2) there are no self loops
        
        Args:
            t_: tree topology (after softmax)
            s : scaling factor
            verbose: print the loss values
        Returns:
            loss: loss value as list [tree_forcing_loss, loop_loss, bidirectional_loss]
    '''

    #n_all = metadata['n_all']
    n_all = t_.shape[0]
    n_leaves = (n_all + 1)//2
    n_ancestors = n_all - n_leaves

    # n_leaves = metadata['n_leaves']
    # n_ancestors = metadata['n_ancestors']


    ancestor_columns = t_[:-1, n_all - n_ancestors: n_all]
    # scaling_vecotr = 

    tree_force_loss = jnp.sum(jnp.power(s*jnp.abs(jnp.sum(ancestor_columns, axis = 0) - 2),2))
    
    if(verbose):
        print("bifurcating tree_forcing_loss = ", tree_force_loss)

    return tree_force_loss

@jit
def compute_surrogate_cost(sequences : Float[Array, "nodes seq_length letters"], tree : Float[Array, "nodes nodes"]) -> Float:
    '''
    This is an approximation of the traversal cost, because seq_onehot is soft 
    '''
    
    seq_onehot = jnp.transpose(sequences, (2,0,1)) # bring onehot encoding to the front (onehot, n_all, seq_length)
    sel        = jnp.expand_dims(tree, axis = 0)   # make it (1, n_all, n_all)

    selection  = jnp.matmul(sel, seq_onehot)
    

    out = jnp.transpose(selection, (1,2,0))
    ans = (jnp.sum(abs((out - sequences)), axis = [1,2])/(2)).sum()
    
    return ans

@jit
def compute_cost(sequences : Float[Array, "nodes seq_length letters"], tree : Float[Array, "nodes nodes"], sm) -> Float:
    '''
        compute the traversal cost (character level changes while traversing the tree)
        This is the exact traversal cost, but it is not differentiable.
        similarity matrix is assumed to be a 1s matrix with 0s on the diagonal

        Args:
            sequences : one-hot encoded sequences
                        shape = (n_all, seq_length, letters)
            tree      : tree topology
                        shape = (n_all, n_all)
            
        Returns: 
            ans : traversal cost
    '''
    selection_ = jnp.matmul(discretize_tree_topology(tree, tree.shape[0]), jnp.argmax(sequences, axis = 2))
    sequences_ = jnp.argmax(sequences, axis = 2)
    
    ans = sm[jnp.round(selection_).astype(jnp.int64),jnp.round(sequences_).astype(jnp.int64)].sum()

    return ans

def compute_loss(params : Dict[str, Array], seqs : Float[Array, "nodes length letters"], base_tree : Float[Array, "nodes nodes"], metadata : Dict, temp : Float, epoch : int = 0, verbose = False) -> Float:
    """Computes the total loss for a given set of sequences and a tree

    Args:
        params : contains the tree and seq trainable parameters
        seqs   : sequences array. shape = (n_all, length, n_letters
        base_tree : base tree with zeros
        metadata (Dict) : contains the metadata
        temp (Float) : temperature for the loss
        verbose : returns individual loss item if True. Defaults to False.

    Returns:
        cost : total loss
    """    
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
    

    cost_surrogate = compute_surrogate_cost(seqs_,t_) # surrogate traversal cost

    tree_force_loss = enforce_graph(t_,10,metadata)
    
    loss = cost_surrogate + temp*(tree_force_loss)

    if(verbose):
        sm   = jnp.ones((metadata['n_letters'],metadata['n_letters'])) - jnp.identity(metadata['n_letters']).astype(jnp.bfloat16)
        cost = compute_cost(seqs_,t_, sm) # real traversal cost

        return cost, cost_surrogate, tree_force_loss, loss

    return loss

@jit
def compute_loss_optimized(tree_params : Dict[str, Array], seq_params : Dict[str, Array], seqs : Float[Array, "nodes length letters"], metadata : Dict, temp : Float, epoch : int) -> Float:
    """Computes the total loss for a given set of sequences and a tree

    Args:
        params : contains the tree and seq trainable parameters
        seqs   : sequences array. shape = (n_all, length, n_letters
        base_tree : base tree with zeros
        metadata (Dict) : contains the metadata
        temp (Float) : temperature for the loss
        verbose : returns individual loss item if True. Defaults to False.

    Returns:
        cost : total loss
    """    
    n_leaves = metadata['n_leaves']
    n_all    = metadata['n_all']

    seqs_ = update_seq(seq_params, seqs, temp )
    
    t_ = update_tree(tree_params, epoch, temp)
    

    cost_surrogate = compute_surrogate_cost(seqs_,t_) # surrogate traversal cost

    tree_force_loss = enforce_graph(t_,10,metadata)
    
    loss = cost_surrogate + temp*(tree_force_loss)

    return loss

@jit
def compute_detailed_loss_optimized(tree_params : Dict[str, Array], seq_params : Dict[str, Array], seqs : Float[Array, "nodes length letters"], metadata : Dict, temp : Float, sm : Float[Array, "letters letters"], epoch : int = 0) -> Float:
    """Computes the total loss for a given set of sequences and a tree

    Args:
        params : contains the tree and seq trainable parameters
        seqs   : sequences array. shape = (n_all, length, n_letters
        base_tree : base tree with zeros
        metadata (Dict) : contains the metadata
        temp (Float) : temperature for the loss
        verbose : returns individual loss item if True. Defaults to False.

    Returns:
        cost : total loss
    """    
    n_leaves = metadata['n_leaves']
    n_all    = metadata['n_all']

    seqs_ = update_seq(seq_params, seqs, temp )
    
    t_ = update_tree(tree_params, epoch, temp)
    

    cost_surrogate = compute_surrogate_cost(seqs_,t_) # surrogate traversal cost

    tree_force_loss = enforce_graph(t_,10,metadata)
    
    loss = cost_surrogate + temp*(tree_force_loss)

    cost = compute_cost(seqs_,t_, sm) # real traversal cost

    return cost, cost_surrogate, tree_force_loss, loss

@jax.jit
def compute_loss_sankoff(params : Dict[str, Array], seqs : Float[Array, "nodes length letters"], base_tree : Float[Array, "nodes nodes"], metadata : Dict, temp : Float, verbose = False) -> Float:
    """Computes the total loss for a given set of sequences and a tree

    Args:
        params : contains the tree and seq trainable parameters
        seqs   : sequences array. shape = (n_all, length, n_letters
        base_tree : base tree with zeros
        metadata (Dict) : contains the metadata
        temp (Float) : temperature for the loss
        verbose : returns individual loss item if True. Defaults to False.

    Returns:
        cost : total loss
    """    
    n_leaves = metadata['n_leaves']
    n_all    = metadata['n_all']

    
    if(metadata['args']['fix_tree']):
        t_ = base_tree
    else:   
        t_ = update_tree_sankoff(params, base_tree)
    
    recurse_cost = 0 #_h_det(t_)*100 #compute_cost_recursive(t_, verbose = False)
    
    cost_mat = jnp.ones((metadata['n_letters'],metadata['n_letters'])) - jnp.eye(metadata['n_letters']).astype(jnp.float64)
    _, cost_surrogate_sankoff = run_diff_sankoff(t_, cost_mat, jnp.argmax(seqs, axis = 2), metadata, epsilon = (0.145e-1)/(1+temp))
    
   

    tree_force_loss = enforce_graph(t_,10,metadata)
    
    loss = cost_surrogate_sankoff #+ 100*tree_force_loss #+ temp*(tree_force_loss + loop_loss + bidirection_loss + recurse_cost)

    if(verbose):
        _, _, cost = run_sankoff(discretize_tree_topology(t_, n_all), cost_mat, jnp.argmax(seqs, axis = 2), metadata, return_path = False)
        #compute_cost(seqs_,t_, metadata, surrogate_loss = False) # real traversal cost
        return cost, cost_surrogate_sankoff, tree_force_loss, loss

    return loss


def compute_loss_upgma(params : Dict[str, Array], seqs : Float[Array, "nodes length letters"], base_tree : Float[Array, "nodes nodes"], metadata : Dict, temp : Float, verbose = False) -> Float:
    """Computes the total loss for a given set of sequences and a tree

    Args:
        params : contains the tree and seq trainable parameters
        seqs   : sequences array. shape = (n_all, length, n_letters
        base_tree : base tree with zeros
        metadata (Dict) : contains the metadata
        temp (Float) : temperature for the loss
        verbose : returns individual loss item if True. Defaults to False.

    Returns:
        cost : total loss
    """    
    n_leaves = metadata['n_leaves']
    n_all    = metadata['n_all']

    if(metadata['args']['fix_seqs']):
        seqs_ = seqs
    else:
        seqs_ = update_seq(params, seqs, temp )
    
    if(metadata['args']['fix_tree']):
        t_ = base_tree
    else:   
        t_ = differentiable_upgma(params['t'], [0.1, 1, 0.5+temp])#update_tree(params, base_tree)
    
    
    recurse_cost = 0 #compute_cost_recursive(t_, verbose = False)

    cost_surrogate = compute_cost(seqs_,t_, metadata) # surrogate traversal cost
    
    tree_force_loss, loop_loss, bidirection_loss = 0.0, 0.0, 0.0 #enforce_graph(t_,10,metadata)
    
    loss = cost_surrogate #+ temp*(tree_force_loss + loop_loss + bidirection_loss + recurse_cost*500)

    if(verbose):
        cost           = compute_cost(seqs_,t_, metadata, surrogate_loss = False) # real traversal cost
        return cost, cost_surrogate, tree_force_loss, loop_loss, bidirection_loss, recurse_cost, loss

    return loss
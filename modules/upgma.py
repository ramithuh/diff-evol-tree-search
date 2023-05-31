import jax
import numpy as np
import jax.nn as nn
from jax import jit
import jax.numpy as jnp


import sys
import functools
from typing import Dict, List, Tuple
from jaxtyping import Array, Float

from ott.tools import soft_sort
softranks = jax.jit(soft_sort.ranks)

def print_matrix(mat, mask_value, masked_char='-', padding=4):
    '''
        Nicely print matrix -_-
    '''
    matrix = mat.astype(jnp.float64)
    
    for row in matrix:
        for value in row:
            if value >= mask_value - 1:
                print(f"{masked_char:>{padding}}", end=' ')
            else:
                print(f"{value:>{padding}.0f}", end=' ')
        print()

def get_prototypes(indices : Tuple[Array, Array], n_nodes : int) -> Float[Array, "options nodes"]:
    '''
        Given indices of the upper triangular matrix, 
        return the corresponding onehot encoded protypes
        indicating which two points are merged at each step.
        
        Args:
            indices : Tuple with two numpy arrays indicating the index pairs which are combined
            n_nodes : Dimension of the onehot encoded protypes (number of nodes currently in the tree)
            
        Returns:
            prototypes : Onehot encoded protypes

    '''
    dim = indices[0].shape[0]

    prototypes = jnp.zeros((dim, n_nodes)).astype(jnp.bfloat16)

    for i in range(0,dim):
        #consider ith pair of indices  
        prototypes = prototypes.at[i,indices[0][i]].set(1) #indicate indices[0][i] is selected
        prototypes = prototypes.at[i,indices[1][i]].set(1) #indicate indices[1][i] is selected

    return prototypes

@jit
def expand_matrix(mat : Float[Array, "nodes nodes"], sel : Float[Array, "options nodes"]) -> Float[Array, "nodes+1 nodes+1"]:
    '''
        Given a distance matrix, expand it based on the selected nodes to merge
        
        Args:
            mat : Distance matrix
            sel : Selected nodes to merge
            
        Returns:
            sol : Expanded distance matrix
    '''

    dim = mat.shape[0]
    e   = get_expansion_matrix(dim, sel)

    sol = jnp.zeros((dim+1,dim+1)) #expanded matrix with the added ancestor node
    
    sol = sol.at[:,:-1].set(sol[:,:-1] + (e.T @ mat))
    sol = sol.at[:-1,:].set(sol[:-1,:] + (mat @ e))

    return sol/2

def get_expansion_matrix(source_dim : int, sel : Float[Array, "options nodes"]) -> Float[Array, "nodes nodes+1"]:
    '''
        Returns the expansion matrix which can be used to combine two nodes to produce the ancestor entry
        
        Args:
            source_dim : Dimension of the source matrix
            sel : Selected nodes to merge
            
        Returns:
            [] : Expansion matrix

    '''
    e = jnp.eye(source_dim)

    return jnp.column_stack((e,sel))

@jit
def mask_matrix(mat : Float[Array, "nodes nodes"], idx : Float[Array, "options"], inf = 1e3) -> Float[Array, "nodes nodes"]:
    '''
        Given a distance matrix, mask the rows and columns specified by the indices provided (idx)
        
        Args:
            mat : Distance matrix
            idx : Indices to mask

        Returns:
            sol : Masked distance matrix (masked positions are based on the nodes connected)
    '''    
    dim = mat.shape[0]

    mat = mat.at[:,idx[0]].set(inf)
    mat = mat.at[:,idx[1]].set(inf)

    mat = mat.at[idx[0],:].set(inf)
    mat = mat.at[idx[1],:].set(inf)

    sol = mat

    return sol

@jit
def differentiable_upgma(dm : Float[Array, "initial_nodes initial_nodes"], e : List[Float] = [0.1, 1, 0.5], verbose : bool = False) -> Float[Array, "all_nodes all_nodes"]:
    '''
        Given a distance matrix, return the UPGMA tree
        Code was inspired by : http://ls50.solab.org/week_5, after looking at how UPGMA is implemented
        
        Args:
            dm : Distance matrix
            e : temperature parameters for 1) softrank and 2) softmax
        Returns:
            tree : tree (adjacency matrix)
    '''
    L = dm.shape[0] 
    all_nodes = L*2 - 1

    tree   = jnp.zeros((all_nodes,all_nodes)).astype(jnp.bfloat16) ## emptry tree (adjacency matrix)
    t      = jnp.zeros((all_nodes,all_nodes)).astype(jnp.bfloat16)
    vector = jnp.zeros((all_nodes,)).astype(jnp.bfloat16)          ## sample ancestor description vector

    current = dm
    
    for n in range(L-1):
        # Running iteration and adding a new parent node
        #print("Running iteration -> ", n)
    
        indices   = jnp.triu_indices_from(current,1)

        if(indices[0].shape[0] > 1): ##TODO : handle only one node case where softrank fails
            
            soft      = nn.softmax(softranks(-current[indices], epsilon=e[0])*e[1]).reshape(indices[0].shape[0],1)
            prototype = get_prototypes(indices, current.shape[0])
            
            sel = jnp.sum(soft*prototype,axis = 0)
        else:
            sel = jnp.array([1,1]).astype(jnp.bfloat16)

        pseudo_idx = jnp.argpartition(sel, -1)[-2::][::-1] #getting the most probable two nodes to merge
        
        current = expand_matrix(current, sel)
        current = mask_matrix(current,pseudo_idx)
        
        new_sel = vector.at[:sel.shape[0]].set(sel)
        tree    = tree.at[:, n + L].set(new_sel)
        
        if(verbose):
            print("interation -> ", n)
            print_matrix(current, 1e3)
            print("stacking column vector -> ", new_sel)


    t_softmax = nn.softmax(tree[:-1,:]*e[2],axis=1)
    t = t.at[:-1,:].set(t_softmax)
    t = t.at[-1,-1].set(1) 

    return t

def get_NJ(X, upgma=False, nw=True, nw_labels=None, nw_parent=False):
  '''
  Credit : Dr. Sergey Ovchinnikov (source : https://colab.research.google.com/drive/1WQiXA-qASGbYkhHHM0x8Fo7RNipYI5ip#scrollTo=SOJlcJp56bqs, 
                                            http://ls50.solab.org/week_5)
  given distance matrix, return treea
  --------------------------------------------------------
  - X:         input distance matrix
  - upgma:     [True/False] do upgma instead of neighbor joining
  ----------------------------------------------------------------
  - nw:        [True/False] return newick string
  - nw_labels: use provided labels
  - nw_parent: [True/False] label parent nodes in newick string
  ----------------------------------------------------------------
  '''
  def min_idx(q):
    '''given symmetric matrix, return indices of smallest value'''
    i = np.triu_indices_from(q,1)
    i_min = np.argmin(q[i])
    return [j[i_min] for j in i]

  dm = np.array(X)          # distance matrix
  L = dm.shape[0]           # num of nodes
  nodes = np.arange(L)      # list of nodes
  adj = [None] * (L-1) * 2  # adj. table

  if upgma:
    # keep track of total distances accounted for
    dist_to_tip = np.zeros(L)
  
  if nw:
    # initialize newick-string
    if nw_labels is None: nw_labels = nodes.astype(np.str)
    else: nw_labels = np.array(nw_labels)

  # build tree
  for n in range(L-1):
    # new parent node    
    parent_node = n + L

    if upgma:
      # indices of min(dm matrix)
      idx = min_idx(dm)
      
      # compute branch lengths
      branch_len_avg = dm[idx[0],idx[1]]/2
      
      # substract distance already accounted for
      branch_len_0, branch_len_1 = branch_len_avg - dist_to_tip[idx]

    else:
      # compute q matrix
      dm_len = dm.shape[0]-2
      dm_sum = dm.sum(0)
      q = (dm_len * dm) - dm_sum[None,:] - dm_sum[:,None]
      
      # indices of min(q matrix)
      idx = min_idx(q) 

      # compute branch lengths
      branch_len_avg = dm[idx[0],idx[1]]/2
      if dm_len == 0:
        branch_len_0 = branch_len_1 = branch_len_avg
      else:
        y = dm_sum[idx]/(2*dm_len)
        branch_len_0 = branch_len_avg + (y[0] - y[1])
        branch_len_1 = branch_len_avg + (y[1] - y[0])

    # update adj. table
    child_node_0, child_node_1 = nodes[idx]
    adj[child_node_0] = [child_node_0, parent_node, branch_len_0]
    adj[child_node_1] = [child_node_1, parent_node, branch_len_1]

    # update newick string
    if nw:
      nw_labels_ = f"({nw_labels[idx[0]]}:{branch_len_0},"
      nw_labels_ += f"{nw_labels[idx[1]]}:{branch_len_1})"
      if nw_parent: nw_labels_ += f"p{parent_node}"
      nw_labels = np.append(np.delete(nw_labels,idx),nw_labels_)

    # del children nodes, add parent node
    nodes = np.append(np.delete(nodes,idx),parent_node)

    ###############
    ## UPDATE DM ##
    ###############

    # add parent
    parent_dist = dm[idx].mean(0,keepdims=True)
    if upgma: dist_to_tip = np.append(np.delete(dist_to_tip, idx), branch_len_avg)
    else: parent_dist -= branch_len_avg
    dm = np.append(dm,parent_dist,0) # add row
    dm = np.append(dm,np.append(parent_dist,0)[:,None],1) # add col

    # del children
    dm = np.delete(dm,idx,0) # del row
    dm = np.delete(dm,idx,1) # del col

  if nw: return adj, nw_labels[0]
  else: return adj
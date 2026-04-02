import os
import sys
import wandb
import pydot
import argparse
import numpy as np

from matplotlib import rc
import plotly.express as px
rc('animation', html='jshtml')

import jax
## use cpu for the time being, let's see whether user requested a gpu later.
jax.config.update("jax_default_device", jax.devices("cpu")[0]) 
import jax.nn as nn
import jax.numpy as jnp

if(os.path.exists('/content/sample_data')):
  sys.path.append('differentiable-trees/')

from modules.vis_utils import *
from modules.gt_tree_gen import *
from arg_parser_v2 import *

def compute_distance(seq1, seq2, cost_mat):
    return jnp.sum(cost_mat[seq1.astype(int),seq2.astype(int)])

## Parse Command Line Arguments and perform checks
args = vars(parse_args_simple())

#### Define Sequence length and number of leaves
seq_length  = int(args['seq_length']) if args['seq_length']!=None else 20
n_leaves    = int(args['leaves']) if args['leaves']!=None else 4
n_ancestors = n_leaves - 1
n_all       = n_leaves + n_ancestors
n_mutations = int(args['mutations']) if args['mutations']!=None else 3
n_letters   = int(args['letters']) if args['letters']!=None else 20
 
metadata = {
    'n_all' : n_all,
    'n_leaves' : n_leaves, 
    'n_ancestors' : n_ancestors,
    'seq_length' : seq_length,
    'n_letters' : n_letters,
    'n_mutations' : n_mutations,
    'seed' : int(args['seed']) if args['seed']!=None else 42,
    }

#### Generate a random sequence of 0s and 1s
key = jax.random.PRNGKey(args['seed'])

#### Generate the groundtruth tree and sequences
seqs, gt_seqs, tree = generate_groundtruth(metadata, args['seed'])

#### Shuffle the sequences if requested
if(args['shuffle_seqs']):        
    shuffled_leaves = jax.random.permutation(key, seqs[0:n_leaves], independent=False) 
    seqs = seqs.at[0:n_leaves].set(shuffled_leaves)
    
    shuffled_ancestors = jax.random.permutation(key, seqs[n_leaves:-1], independent=False) 
    seqs = seqs.at[n_leaves:-1].set(shuffled_ancestors)


cost_matrix = jnp.ones((metadata['n_letters'],metadata['n_letters'])) - jnp.identity(metadata['n_letters']).astype(jnp.float64)

print("Sequences: (truncated to 20 characters)")
arr = []
for i in range(0,n_leaves):
    print(f"{i} -> {seqs[i][:20]}")
    
    arr.append(str(i))

## Compute distance matrix
distance_mat = jnp.zeros((n_leaves, n_leaves))
for i in range(0, n_leaves):
    for j in range(i+1, n_leaves):
        distance_mat = distance_mat.at[i,j].set(compute_distance(seqs[i], seqs[j], cost_matrix))
        distance_mat = distance_mat.at[j,i].set(distance_mat[i,j])
        
print(distance_mat)
print(arr)
import math
import jax.numpy as jnp
from jax import vmap
from jax import random
from jax import jit
from functools import partial

from typing import Dict, List
from jaxtyping import Array, Float


target_depth = 0
num_mutations = 0
sequences = []
n_letters = 0

def mutate(key, exclude_indexes):
  '''
    Mutates a sequence by randomly changing one of the letters

    Accepts:
      exclude_indexes: list of indexes to exclude from the mutation (shape = (n_mutations,1))

    Returns:
      mutated sequence
  '''
  global n_letters

  space = jnp.tile(jnp.arange(n_letters), (num_mutations, 1)) 
  options  = space[space != exclude_indexes].reshape((space.shape[0],space.shape[1] - exclude_indexes.shape[1]))

  key_ = random.split(key, num_mutations)

  mutations = vmap(random.choice)(key_, options)

  return jnp.expand_dims(mutations, axis=1)

def create_seq(key, seq, depth):

  k1, k2 = random.split(key)
  
  if(depth > target_depth):
    return

  random_indexes = [[] for x in range(2)]
  
  random_indexes[0]= random.choice(k1, jnp.array(range(len(seq))), (num_mutations,1), replace=False)
  random_indexes[1]= random.choice(k2, jnp.array(range(len(seq))), (num_mutations,1), replace=False)


  seq_1 = seq
  seq_2 = seq

  seq_1 = seq_1.at[random_indexes[0]].set(mutate(k1, seq[random_indexes[0]]))
  seq_2 = seq_2.at[random_indexes[1]].set(mutate(k2, seq[random_indexes[1]]))


  #seq
  sequences[depth].append(seq_1)
  create_seq(k1, seq_1, depth+1)
  
  #new_seq
  sequences[depth].append(seq_2)
  create_seq(k2, seq_2, depth+1)


def generate_groundtruth(metadata : Dict[str, int], seed = 42, verbose = False) -> List[Array]: 
    '''
        Generates a groundtruth example based on the metadata provided


        Args : 
            metadata: dictionary containing the required specifications
            seed: random seed
            verbose: print the number of leaves generated

        Returns:
            masked_main : masked sequences (shape = (n_all, seq_length))
            true_main   : true sequences   (shape = (n_all, seq_length))
            tree        : tree structure   (shape = (n_all, n_all))
    '''

    key = random.PRNGKey(seed)

    global target_depth
    global num_mutations
    global sequences
    global n_letters

    num_mutations = metadata['n_mutations']
    n_all       = metadata['n_all']
    n_leaves    = metadata['n_leaves']
    n_ancestors = metadata['n_ancestors']
    seq_length  = metadata['seq_length']
    n_letters   = metadata['n_letters']

    target_depth = int(math.log2(n_leaves))-1

    seq = jnp.zeros(seq_length, dtype=jnp.int64)

    sequences = [[] for x in range(target_depth + 1)]
    create_seq(key,seq,0)

    if(verbose):
      print(len(sequences[target_depth]))

    ### copy the leaves
    masked_main = sequences[target_depth].copy()
    true_main   = sequences[target_depth].copy()

    n_leaves    = len(masked_main)
    n_ancestors = n_leaves - 1


    for i in range(0,n_ancestors):
        masked_main.append(seq)

    ## Generate the true sequences by traversing through the leaves to the root
    for i in range(len(sequences)-2, -1, -1):
        for k in range(0,len(sequences[i])):
            true_main.append(sequences[i][k])
    true_main.append(seq)

    if(verbose):
      i = 0
      for item in true_main:
        print(">seq"+str(i))
        for char in item:
          print(char, end="")
        print()

        i+=1

    tree = []

    iter = n_ancestors

    for i in range(0,len(masked_main)):
        leave = [0]*len(masked_main)

        if(i%2==0 and i!=len(masked_main)-1):
            iter +=1 

        leave[iter] = 1
        tree.append(leave)

    return jnp.array(masked_main).astype(jnp.bfloat16), jnp.array(true_main).astype(jnp.bfloat16), jnp.array(tree).astype(jnp.bfloat16)

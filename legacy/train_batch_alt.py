from jax import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

import os
import sys

import optax
import jaxopt
from jaxopt import OptaxSolver
from jaxopt import linear_solve
from jax.tree_util import Partial

import wandb
import pydot
import warnings
import argparse
import functools
from typing import Dict

from matplotlib import rc
import plotly.express as px
rc('animation', html='jshtml')

import jax
## use cpu for the time being, let's see whether user requested a gpu later.
jax.config.update("jax_default_device", jax.devices("cpu")[0]) 
import jax.nn as nn
from jax import vmap
import jax.numpy as jnp
from jax.lib import xla_bridge
from jaxtyping import Array, Float

if(os.path.exists('/content/sample_data')):
  sys.path.append('differentiable-trees/')

from modules.vis_utils import *
from modules.tree_func import *
from modules.gt_tree_gen import *
from modules.sankoff import *
from arg_parser_v2 import *


def generate_vmap_keys(seq_params):
    vmap_keys = {}
    
    for key in seq_params.keys():
        vmap_keys[key] = 0
        
    return vmap_keys

@jit
def get_one_tree_and_seq(tree_params, seq_params, pos):
    new_params = {}
    ## extract correct tree
    new_params['t'] = tree_params['t'][pos]
    
    for i in range(0, len(seq_params.keys())):
        new_params[str(i)] = seq_params[str(i)][pos]
    
    return new_params


def inner_objective(seq_params, tree_params, data):
    seqs, metadata, temp, epoch = data
    return compute_loss_optimized(tree_params, seq_params, seqs, metadata, temp, epoch)


def outer_objective(tree_params, seq_params, data):
    seqs, metadata, temp, epoch = data
    
    return compute_loss_optimized(tree_params, seq_params, seqs, metadata, temp, epoch)



## Parse Command Line Arguments and perform checks
args = vars(parse_args_v2())
args = sanity_check(args)

#### Define Sequence length and number of leaves
seq_length  = int(args['seq_length']) if args['seq_length']!=None else 20
n_leaves    = int(args['leaves']) if args['leaves']!=None else 4
n_ancestors = n_leaves - 1
n_all       = n_leaves + n_ancestors
n_mutations = int(args['mutations']) if args['mutations']!=None else 3
n_letters   = int(args['letters']) if args['letters']!=None else 20

args['tree_loss_schedule'] = eval(args['tree_loss_schedule']) if args['tree_loss_schedule']!=None else [0,0.01,100,5]
 
metadata = {
    'n_all' : n_all,
    'n_leaves' : n_leaves, 
    'n_ancestors' : n_ancestors,
    'seq_length' : seq_length,
    'n_letters' : n_letters,
    'n_mutations' : n_mutations,
    'args': args,
    'exp_name' : f"l={n_leaves}, m={n_mutations}, s={seq_length}, fs={args['fix_seqs']}, ft={args['fix_tree']}" ,
    'seed' : int(args['seed']) if args['seed']!=None else 42,
    'seq_temp': 0.5,
    'lr': args['learning_rate'],
    'lr_seq' : args['learning_rate_seq'] if args['learning_rate_seq']!=None else args['learning_rate']*10,
    'epochs': args['epochs'],
    'tLs': args['tree_loss_schedule'],
    'init_count' : args['init_count'] if args['init_count']!=None else 1,
    }

if(args['log_wandb']):
    wandb.login(key = os.environ.get('WANDB_API_KEY_so'))
    wandb.init(project=args['project'], name = args['notes'] + metadata['exp_name'], entity="<your_username>", config = metadata, tags=["alt", args['tags']], notes = args['notes'])

if(args['gpu']!=None):
    print_critical_info(f"Utilizing gpu -> {args['gpu']} \n")
    jax.config.update("jax_default_device", jax.devices("gpu")[args['gpu']])
else:
    if(xla_bridge.get_backend().platform == "gpu"):
        print_critical_info("There's a gpu available, but you didn't specify to use it 😏. So using cpu instead 🤷🏻‍♂️. \n")    
        print(f"Available GPUs: {jax.devices('gpu')}")
    jax.config.update("jax_default_device", jax.devices("cpu")[0])

print(pretty_print_dict(metadata))

## JAX Doesn't like some data types when jitting
metadata['exp_name'] = None
metadata['args']['notes'] = None
metadata['notes'] = None
metadata['tags']  = None
metadata['project'] = None
args['notes'] = None
args['tags'] = None
args['project'] = None

#### Generate a random sequence of 0s and 1s
key = jax.random.PRNGKey(metadata['seed'])
offset = 10

#### Generate a base tree
base_tree = jnp.zeros((n_all, n_all))

sm = jnp.ones((metadata['n_letters'],metadata['n_letters'])) - jnp.identity(metadata['n_letters']).astype(jnp.float64)

if(args['groundtruth']):
    seqs, gt_seqs, tree = generate_groundtruth(metadata, metadata['seed'])

    seqs    = jax.nn.one_hot(seqs, n_letters).astype(jnp.float64)
    gt_seqs = jax.nn.one_hot(gt_seqs, n_letters).astype(jnp.float64)

    #if we don't want sequences to change, set seqs to gt_seqs
    if(args['fix_seqs'] or args['initialize_seq']):
        seqs = gt_seqs

    #if we don't want tree to change, set base_tree to tree
    if(args['fix_tree']):
        base_tree = tree
        
    if(not(args['fix_seqs'])):
        ## since we know the gt tree, let's get the real ancestors using sankoff algorithm! 🎉 
        ## This will help us to see whether there's a better groundtruth ancestors for this tree
        cost_mat = (jnp.ones((n_letters,n_letters)) - jnp.eye(n_letters)).astype(jnp.float64)
        
        print_critical_info("running sankoff on groundtruth tree\n")
        _, _, sankoff_cost = run_sankoff(tree, cost_mat, jnp.argmax(seqs, axis = 2), metadata)
        print_success_info("done running sankoff on groundtruth tree. optimal cost = %d\n" % sankoff_cost)
        
        if(args['log_wandb']):
            wandb.log(
                {
                    #'sankoff_seqs': wandb.data_types.Plotly(px.imshow(sankoff_seqs, text_auto=True)),
                    'sankoff_cost': sankoff_cost,
                    #'cost_for_sankoff_seqs' : compute_cost(jax.nn.one_hot(sankoff_seqs, n_letters).astype(jnp.float64), tree, metadata, surrogate_loss = False)
                })
        
    if(args['shuffle_seqs']):        
        shuffled_leaves = jax.random.permutation(key, seqs[0:n_leaves], independent=False) 
        seqs = seqs.at[0:n_leaves].set(shuffled_leaves)
        
        shuffled_ancestors = jax.random.permutation(key, seqs[n_leaves:-1], independent=False) 
        seqs = seqs.at[n_leaves:-1].set(shuffled_ancestors)

    gt_tree = show_graph_with_labels(tree, n_leaves, True)
    gt_seqs_plot = px.imshow(jnp.argmax(gt_seqs, axis = 2), text_auto=True)

    
    gt_cost           = compute_cost(gt_seqs, tree, sm)
    gt_cost_surrogate = compute_surrogate_cost(gt_seqs, tree)
    
    gt_tree_force_loss = enforce_graph(tree, 10, metadata)

    if(args['log_wandb']):
        wandb.log({
            "Groundtruth Tree" : wandb.Image(gt_tree),
            "Groundtruth Seq" : wandb.data_types.Plotly(gt_seqs_plot),

            "Groundtruth Tree Force Loss" : gt_tree_force_loss,
            "Groundtruth Traversal Cost (surrogate)" : gt_cost_surrogate,
            "Groundtruth Traversal Cost" : gt_cost,
            "Groundtruth Total Loss" : gt_tree_force_loss + gt_cost
            })


#### Define parameters (t and the ancestor sequences)
initializer = jax.nn.initializers.kaiming_normal()
#jax.nn.initializers.orthogonal()
#jax.nn.initializers.kaiming_normal()
#jax.nn.initializers.orthogonal()


tree_params : Dict[str, Array] = {
    't': initializer(key + offset, (metadata['init_count'], n_all - 1,n_ancestors), jnp.float64)
    }

seq_params : Dict[str, Array] = {}


## Manually override the parameters such that the updated tree is the same as the base tree
if(args['initialize_tree']):
    print_critical_info("Initializing tree using groundtruth tree \n")
    tree_params['t'] = tree[0:-1,n_leaves:]*100
    
#### Add the ancestor sequences to the parameters
for i in range(0, n_ancestors):
    
    if(args['initialize_seq']):
        if(n_leaves > 1024):
            raise NotImplementedError("Sankoff backtracking not implemented for n_leaves > 1024 due to execution time")
        else:
            print_critical_info("Initializing sequences using sankoff ancestors \n")
            seq_params[str(i)] = jax.nn.one_hot(sankoff_seqs[n_leaves + i], n_letters).astype(jnp.float64)*100
    else:
        seq_params[str(i)] = initializer(key+i+offset, (metadata['init_count'], seq_length, n_letters), jnp.float64)
        #jax.random.normal(key+i+offset, (seq_length, n_letters)).astype(jnp.float64)
    
#### Initialize the optimizer

copy_seq_params = seq_params.copy()
#eps_root = 1e-8, eps = 1e-8

seq_optimizer = OptaxSolver(opt = optax.adam(metadata['lr_seq']), fun = inner_objective, maxiter  = args['alternate_interval']) 
vmap_seq_init = jax.vmap(seq_optimizer.init_state, (0, 0, None),0)
seq_opt_state = vmap_seq_init(seq_params, tree_params, [seqs, metadata, metadata['tLs'][0], 0])
jitted_seq_update = jit(vmap(seq_optimizer.update, (0, 0, 0, None),0))

tree_optimizer = OptaxSolver(opt = optax.adam(metadata['lr']), fun = outer_objective) #, implicit_diff = True)
vmap_tree_init = jax.vmap(tree_optimizer.init_state, (0, 0, None),0)
tree_opt_state = vmap_tree_init(tree_params, seq_params, [seqs, metadata, metadata['tLs'][0], 0])
jitted_tree_update = jit(vmap(tree_optimizer.update, (0, 0, 0, None),0))

vmap_keys = generate_vmap_keys(seq_params)
vmap_compute_detailed_loss_optimized =  jit(vmap(compute_detailed_loss_optimized, ({'t':0}, vmap_keys, None, None, None, None, None),0))


fixed_dummy_pos = 0 

params = get_one_tree_and_seq(tree_params, seq_params, fixed_dummy_pos)

fig2 = show_graph_with_labels(discretize_tree_topology(update_tree(params), n_all),n_leaves, True)

compute_loss(params, seqs, base_tree, metadata, metadata['tLs'][0])


best_ans = 1e9
best_seq = None
best_tree = None

pos = 0
#### Training loop
for _ in range(metadata['epochs']):

    if(_%200==0):
        ###~ Get the current discretized tree
        if(args['fix_tree']): 
            t_ = base_tree
        else:
            t_ = update_tree(params, _, metadata['tLs'][0])
        t_d           = discretize_tree_topology(t_,n_all)
        tree_at_epoch        = show_graph_with_labels(t_d, n_leaves, True)
        tree_matrix_at_epoch = px.imshow(t_, text_auto=True)

        ###~ Get the current sequences as a plot
        if(args['fix_seqs']):
            seqs_ = seqs
        else:
            seqs_ = update_seq(params, seqs, metadata['seq_temp'])
        new_seq = px.imshow(jnp.argmax(seqs_, axis = 2), text_auto=True)
        
    
    #=> Update the parameters
    tree_params, tree_opt_state = jitted_tree_update(tree_params, tree_opt_state, seq_params, [seqs, metadata, metadata['tLs'][0],_])
    
    for i in range(0, seq_optimizer.maxiter):
        seq_params, seq_opt_state = jitted_seq_update(seq_params, seq_opt_state, tree_params, [seqs, metadata, metadata['tLs'][0],_])

    cost, cost_surrogate, tree_force_loss, loss = vmap_compute_detailed_loss_optimized(tree_params, seq_params, seqs, metadata, metadata['tLs'][0], sm,  _)
    pos = jnp.argmin(cost)
    
    params = get_one_tree_and_seq(tree_params, seq_params, pos)
    
    if(cost.min() < best_ans):
        if(_%20==0):
            print_success_info("Found a better tree at epoch %d with cost %f from tree %d. (delta at epoch = %d) \n" % (_, cost.min(), pos, cost.max()-cost.min()))
        best_ans = cost.min()
        
        t_ = update_tree(params, _, metadata['tLs'][0])
        best_tree = discretize_tree_topology(t_,n_all)
        best_seq = update_seq(params, seqs, metadata['seq_temp'])
        
        ## Log the metrics
    if(_%200==0 and args['log_wandb']):
        wandb.log(
            {"epoch":_, 

            "loss":loss[pos],
            "traversal cost": cost[pos],
            "traversal cost (surrogate)": cost_surrogate[pos],
            "tree force loss": tree_force_loss[pos],
            
            "tree":wandb.Image(tree_at_epoch),
            "tree matrix":wandb.data_types.Plotly(tree_matrix_at_epoch),
            "Seq":wandb.data_types.Plotly(new_seq),
            "last ancestor" :  wandb.data_types.Plotly(px.imshow((seqs_[-1]), text_auto=True)),
            "tLs":metadata['tLs'][0],})

    if(_%200==0):
        print_bold_info(f"epoch {_}")
        print("{:.3f}".format(metadata['tLs'][0]), end = " ")
        print("{:.3f}".format(loss[pos].item()), end=" ")
        print("{:.3f}".format(cost[pos].item()), end = "\n")
    

    ### update the tree loss schedule
    if(_%metadata['tLs'][3]==0):
        metadata['tLs'][0] = min(metadata['tLs'][2], metadata['tLs'][0] + metadata['tLs'][1])
        
print_success_info("Optimization done!\n")
print_success_info("Final cost: {:.5f}\n".format(cost[pos]))
print_success_info("Best cost encountered: {:.5f}\n".format(best_ans))

if(args['fix_tree']):
    print_success_info("Sankoff cost for groundtruth tree: {:.5f}\n".format(sankoff_cost.item()))
    target_cost = sankoff_cost
elif(args['fix_seqs']):
    print_success_info("Groundtruth tree cost: {:.5f}\n".format(gt_cost))
    target_cost = gt_cost
else:
    print_critical_info("No groundtruth to compare to\n")
    target_cost = 0


if(abs(target_cost - best_ans) == 0):
    print_success_info("Optimization succeeded! Reached groundtruth 🚀\n")
    if(args['log_wandb']) : wandb.log({"success":True, "Error" : 0})
else:
    if(args['log_wandb']) : wandb.log({"success":False, "Error" : (best_ans - sankoff_cost)})

best_tree_img = show_graph_with_labels(best_tree, n_leaves, True)
best_tree_adj = px.imshow(best_tree, text_auto=True)

if(args['log_wandb']):
    wandb.log({
            "best cost":best_ans, 
            "best_tree_adj" : wandb.data_types.Plotly(best_tree_adj),
            "best_tree" : wandb.Image(best_tree_img),
            "best_seq" : wandb.data_types.Plotly(px.imshow(jnp.argmax(best_seq, axis = 2), text_auto=True))
            })

    wandb.finish()
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from jax import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
import pprint
import argparse
import warnings

import jax
jax.config.update("jax_default_device", jax.devices("cpu")[0])
import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import Array, Float
from typing import Dict

from .tree_func import *
from .gt_tree_gen import *
from .sankoff import *
from .vis_utils import *

# ── Colored console output ──────────────────────────────────────────

class bc:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_critical_info(msg):
    print(f"{bc.FAIL}{bc.BOLD}INFO : {msg} {bc.ENDC}", end = "")

def print_warning_info(msg):
    print(f"{bc.WARNING}{bc.BOLD}WARNING : {msg} {bc.ENDC}", end = "")

def print_bold_info(msg):
    print(f"{bc.BOLD}{msg} {bc.ENDC}", end = "")

def print_success_info(msg):
    print(f"{bc.OKGREEN}{bc.BOLD}{msg} {bc.ENDC}", end = "")

def pretty_print_dict(d):
    formatted_dict = pprint.pformat(d, width=1)
    return formatted_dict

# ── Arg parsing ──────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Differentiable tree search.\n'
                    'Usage: python search_alt.py -l 8 -m 10 -sl 50 -e 2000')

    parser.add_argument('-gt', '--groundtruth', action='store_true', help='Retrieve groundtruth?')
    parser.set_defaults(groundtruth=True)

    parser.add_argument('-e','--epochs', help='# of epochs?', required=True, type=int)

    parser.add_argument('-ic','--init_count', help='# of initializations to run', required=False, type=int)
    parser.add_argument('-l','--leaves', help='# of leaves?', required=True, type=int)
    parser.add_argument('-m','--mutations', help='# of mutations?', required=False, type=int)
    parser.add_argument('-sl','--seq_length', help='length of seq', required=False, type=int)
    parser.add_argument('-nl','--letters', help='# of letters?', required=False, type=int)
    parser.add_argument('-s','--seed', help='seed', required=False, type=int)

    parser.add_argument('-ai','--alternate_interval', help='alternate_interval', required=False, type=int)
    parser.add_argument('-tLs','--tree_loss_schedule', help='tree loss schedule', required=False, type=str)
    parser.add_argument('-lr','--learning_rate', help='learning rate', required=False, type=float, default=0.001)
    parser.add_argument('-lr_seq','--learning_rate_seq', help='learning rate for seq', required=False, type=float)

    parser.add_argument('-fs', '--fix_seqs', action='store_true', help='Fix sequences when training?')
    parser.set_defaults(fix_seqs=False)

    parser.add_argument('-shs', '--shuffle_seqs', action='store_true', help='Shuffle groundtruth sequences?')
    parser.set_defaults(shuffle_seqs=False)

    parser.add_argument('-ft', '--fix_tree', action='store_true', help='Fix tree when training?')
    parser.set_defaults(fix_tree=False)

    parser.add_argument('-it', '--initialize_tree', action='store_true', help='Initialize Tree with groundtruth?')
    parser.set_defaults(initialize_tree=False)

    parser.add_argument('-is', '--initialize_seq', action='store_true', help='Initialize Seq with groundtruth?')
    parser.set_defaults(initialize_seq=False)

    parser.add_argument('-g','--gpu', help='specify device', required=False, type=int)

    parser.add_argument('-vi','--vis_interval', help='visualization interval (epochs)', required=False, type=int, default=200)
    parser.add_argument('-li','--log_interval', help='log interval (epochs)', required=False, type=int, default=200)

    return parser.parse_args()

def sanity_check(args):
    args = args.copy()

    if(args['fix_seqs']):
        warnings.warn("Hey, you just asked to fix seqs. Hope you are sure about the decision.", UserWarning)

    if(args['fix_tree']):
        warnings.warn("Hey, you just asked to fix tree. Hope you are sure about the decision.", UserWarning)

    if(args['fix_tree'] and args['fix_seqs']):
        raise ValueError("Hey, you just asked to fix tree and seqs both! What should we optimize???")

    return args

# ── Metadata + device config ─────────────────────────────────────────

def build_metadata(args):
    seq_length  = int(args['seq_length']) if args['seq_length'] is not None else 20
    n_leaves    = int(args['leaves']) if args['leaves'] is not None else 4
    n_ancestors = n_leaves - 1
    n_all       = n_leaves + n_ancestors
    n_mutations = int(args['mutations']) if args['mutations'] is not None else 3
    n_letters   = int(args['letters']) if args['letters'] is not None else 20

    args['tree_loss_schedule'] = eval(args['tree_loss_schedule']) if args['tree_loss_schedule'] is not None else [0,0.01,100,5]

    metadata = {
        'n_all' : n_all,
        'n_leaves' : n_leaves,
        'n_ancestors' : n_ancestors,
        'seq_length' : seq_length,
        'n_letters' : n_letters,
        'n_mutations' : n_mutations,
        'args': args,
        'exp_name' : f"l={n_leaves}, m={n_mutations}, s={seq_length}, fs={args['fix_seqs']}, ft={args['fix_tree']}",
        'seed' : int(args['seed']) if args['seed'] is not None else 42,
        'seq_temp': 0.5,
        'lr': args['learning_rate'],
        'lr_seq' : args['learning_rate_seq'] if args['learning_rate_seq'] is not None else args['learning_rate']*10,
        'epochs': args['epochs'],
        'tLs': args['tree_loss_schedule'],
        'init_count' : args['init_count'] if args['init_count'] is not None else 1,
    }
    return metadata

def setup_device(args):
    if(args['gpu'] is not None):
        print_critical_info(f"Utilizing gpu -> {args['gpu']} \n")
        jax.config.update("jax_default_device", jax.devices("gpu")[args['gpu']])
    elif any(d.platform == "gpu" for d in jax.devices()):
        print_success_info(f"Auto-detected GPU: {jax.devices('gpu')[0]}\n")
        jax.config.update("jax_default_device", jax.devices("gpu")[0])
    else:
        jax.config.update("jax_default_device", jax.devices("cpu")[0])

# ── Data generation ──────────────────────────────────────────────────

def generate_data(metadata):
    """Generate GT data, compute Sankoff baseline, return seqs + tree + sankoff_cost."""
    args = metadata['args']
    n_leaves = metadata['n_leaves']
    n_letters = metadata['n_letters']
    n_all = metadata['n_all']

    key = jax.random.PRNGKey(metadata['seed'])
    sm = jnp.ones((n_letters, n_letters)) - jnp.identity(n_letters).astype(jnp.float64)

    seqs, gt_seqs, tree = generate_groundtruth(metadata, metadata['seed'])

    seqs    = jax.nn.one_hot(seqs, n_letters).astype(jnp.float64)
    gt_seqs = jax.nn.one_hot(gt_seqs, n_letters).astype(jnp.float64)

    if(args['fix_seqs'] or args['initialize_seq']):
        seqs = gt_seqs

    base_tree = jnp.zeros((n_all, n_all))
    if(args['fix_tree']):
        base_tree = tree

    sankoff_cost = None
    if(not args['fix_seqs']):
        cost_mat = (jnp.ones((n_letters, n_letters)) - jnp.eye(n_letters)).astype(jnp.float64)
        print_critical_info("running sankoff on groundtruth tree\n")
        _, _, sankoff_cost = run_sankoff(tree, cost_mat, jnp.argmax(seqs, axis = 2), metadata)
        print_success_info("done running sankoff on groundtruth tree. optimal cost = %d\n" % sankoff_cost)

    if(args['shuffle_seqs']):
        shuffled_leaves = jax.random.permutation(key, seqs[0:n_leaves], independent=False)
        seqs = seqs.at[0:n_leaves].set(shuffled_leaves)
        shuffled_ancestors = jax.random.permutation(key, seqs[n_leaves:-1], independent=False)
        seqs = seqs.at[n_leaves:-1].set(shuffled_ancestors)

    # Print GT info
    gt_tree_fig = show_graph_with_labels(tree, n_leaves, True)
    gt_cost = compute_cost(gt_seqs, tree, sm)
    gt_cost_surrogate = compute_surrogate_cost(gt_seqs, tree)
    print_success_info(f"GT cost (hard): {gt_cost:.3f}, GT cost (surrogate): {gt_cost_surrogate:.3f}\n")

    return seqs, gt_seqs, tree, base_tree, sm, sankoff_cost, gt_cost

# ── Parameter initialization ─────────────────────────────────────────

def init_params(metadata, seqs, n_leaves, n_ancestors, init_count):
    """Initialize tree_params and seq_params."""
    args = metadata['args']
    key = jax.random.PRNGKey(metadata['seed'])
    offset = 10
    n_all = metadata['n_all']
    seq_length = metadata['seq_length']
    n_letters = metadata['n_letters']

    initializer = jax.nn.initializers.kaiming_normal()

    tree_params : Dict[str, Array] = {
        't': initializer(key + offset, (init_count, n_all - 1, n_ancestors), jnp.float64)
    }

    seq_params : Dict[str, Array] = {}

    if(args['initialize_tree']):
        print_critical_info("Initializing tree using groundtruth tree \n")
        # Access tree from caller scope — passed via metadata workaround not needed,
        # caller should handle tree init override after this function returns
        pass

    for i in range(0, n_ancestors):
        seq_params[str(i)] = initializer(key+i+offset, (init_count, seq_length, n_letters), jnp.float64)

    return tree_params, seq_params

# ── Shared helpers ────────────────────────────────────────────────────

def generate_vmap_keys(seq_params):
    vmap_keys = {}
    for key in seq_params.keys():
        vmap_keys[key] = 0
    return vmap_keys

@jit
def get_one_tree_and_seq(tree_params, seq_params, pos):
    new_params = {}
    new_params['t'] = tree_params['t'][pos]
    for i in range(0, len(seq_params.keys())):
        new_params[str(i)] = seq_params[str(i)][pos]
    return new_params

def clear_metadata_for_jit(metadata, args):
    """JAX doesn't like some data types when jitting."""
    metadata['exp_name'] = None
    metadata['args']['notes'] = None
    metadata['notes'] = None
    metadata['tags']  = None
    metadata['project'] = None
    args['notes'] = None
    args['tags'] = None
    args['project'] = None

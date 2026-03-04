import pprint
import argparse
import warnings

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

def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    orange = '\033[33m'
    reset = '\033[0m'
    
    print('\n{}{}{}'.format(orange, msg, reset))
    return "\n"

def parse_args_v2():
    #### Parse arguments
    parser = argparse.ArgumentParser(description='Train a differentiable tree model. \n sample usage :  python train.py -l 8 -m 10 -sl 50 -fs')

    parser.add_argument('-gt', '--groundtruth', action='store_true', help='Retrieve groundtruth?')
    parser.set_defaults(groundtruth=True)

    parser.add_argument('-p','--project'     , help='wandb project'            , required=True, type = str, default = 'test')
    parser.add_argument('-n','--notes'       , help='wandb notes'            , required=False, type = str, default = '')
    parser.add_argument('-t','--tags'        , help='wandb tags'            , required=False, type = str, default = 'default')

    parser.add_argument('-e','--epochs'      , help='# of epochs?'    , required=True, type = int)

    #tree and seq generate parameters
    parser.add_argument('-ic','--init_count' , help='# of initializations to run' , required=False, type = int)
    parser.add_argument('-l','--leaves'      , help='# of leaves?'    , required=True, type = int)
    parser.add_argument('-m','--mutations'   , help='# of muations?'  , required=False, type = int)
    parser.add_argument('-sl','--seq_length' , help='length of seq'   , required=False, type = int)
    parser.add_argument('-nl','--letters'    , help='# of letters?'    , required=False, type = int)
    parser.add_argument('-s','--seed'        , help='seed'            , required=False, type = int)

    parser.add_argument('-ai','--alternate_interval', help='alternate_interval', required=False, type = int)
    parser.add_argument('-tLs','--tree_loss_schedule' , help='tree loss schedule', required=False, type = str)
    parser.add_argument('-lr','--learning_rate'         , help='learning rate'    , required=False, type = float, default = 0.001)
    parser.add_argument('-lr_seq','--learning_rate_seq' , help='learning rate for seq' , required=False, type = float)

    ## what to train?
    parser.add_argument('-fs', '--fix_seqs', action='store_true', help='Fix sequences when training ?')
    parser.set_defaults(fix_seqs=False)
    
    parser.add_argument('-w', '--log_wandb', action='store_true', help='Log experiments to wandb?')
    parser.set_defaults(log_wandb=False)

    parser.add_argument('-shs', '--shuffle_seqs', action='store_true', help='Shuffle groundtruth sequences?')
    parser.set_defaults(shuffle_seqs=False)

    parser.add_argument('-ft', '--fix_tree', action='store_true', help='Fix tree when training ?')
    parser.set_defaults(fix_tree=False)

    parser.add_argument('-it', '--initialize_tree', action='store_true', help='Initialize Tree with groundtruth?')
    parser.set_defaults(initialize_tree=False)

    parser.add_argument('-is', '--initialize_seq', action='store_true', help='Initialize Seq with groundtruth?')
    parser.set_defaults(initialize_seq=False)

    parser.add_argument('-alt', '--alternate_optimization', action='store_true', help='Alternate tree and seq optimization?')
    parser.set_defaults(alternate_optimization=False)

    parser.add_argument('-g','--gpu', help='specify device', required=False, type = int)

    return parser.parse_args()

def parse_args_simple():
    #### Parse arguments
    parser = argparse.ArgumentParser(description='Get statistics of the generated random tree. \n sample usage :  python gt_stats.py -l 8 -m 10 -sl 50')

    #tree and seq generate parameters
    parser.add_argument('-l','--leaves'      , help='# of leaves?'    , required=True, type = int)
    parser.add_argument('-m','--mutations'   , help='# of muations?'  , required=False, type = int)
    parser.add_argument('-sl','--seq_length' , help='length of seq'   , required=False, type = int)
    parser.add_argument('-nl','--letters'    , help='# of letters?'    , required=False, type = int)
    parser.add_argument('-s','--seed'        , help='seed'            , required=False, type = int)
    
    parser.add_argument('-shs', '--shuffle_seqs', action='store_true', help='Shuffle groundtruth sequences?')
    parser.set_defaults(shuffle_seqs=False)

    return parser.parse_args()

def sanity_check(args_):
    ### set contradictory arguments (otherwise could be confusing in wandb)
    args = args_.copy()
    
    if(args['fix_seqs']):
        warnings.warn("Hey, you just asked to fix seqs. Hope you are sure about the decision.", UserWarning)
        
    if(args['fix_tree']):
        warnings.warn("Hey, you just asked to fix tree, Hope you are sure about the decision.", UserWarning)
    
    if(args['fix_tree'] and args['fix_seqs']):
        raise ValueError("Hey, you just asked to fix tree and seqs both! that what should we optimize???")
    
    return args

def pretty_print_dict(d):
    # Format the dictionary using the pprint module
    
    formatted_dict = pprint.pformat(d, width=1)
    
    # Return the formatted dictionary as a string
    return formatted_dict

def print_critical_info(msg):
    print(f"{bc.FAIL}{bc.BOLD}INFO : {msg} {bc.ENDC}", end = "")
    
def print_warning_info(msg):
    print(f"{bc.WARNING}{bc.BOLD}WARNING : {msg} {bc.ENDC}", end = "")

def print_bold_info(msg):
    print(f"{bc.BOLD}{msg} {bc.ENDC}", end = "")
    
def print_success_info(msg):
    print(f"{bc.OKGREEN}{bc.BOLD}{msg} {bc.ENDC}", end = "")
    
warnings.formatwarning = custom_formatwarning
import argparse


"""
To compare fairly with previous work, we do not change the parameters of the RGCN configuration, 
static graph configuration, and ConvTransE-based decoder configuration.
"""


parser = argparse.ArgumentParser(description='DSEP')


# General Settings
parser.add_argument("--gpu", type=int, default=-1, help="GPU index (set -1 for CPU)")
parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset to use")
parser.add_argument("--test", action='store_true', default=False, help="Load model and directly test")
parser.add_argument("--add-static-graph", action='store_true', default=False, help="Use static graph information")
parser.add_argument("--run-analysis", action='store_true', default=False, help="print log info")
parser.add_argument("--relation-evaluation", action='store_true', default=False, help="save model accordding to the relation evalution")
parser.add_argument("--write-output", action='store_true', default=False, help="Write output to file")
parser.add_argument("--save", action='store_true', help="Save the model")


# RGCN Configuration
parser.add_argument("--gcn", type=str, default="convgcn", help="GCN method")
parser.add_argument("--aggregation", type=str, default="none", help="Aggregation method")
parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
parser.add_argument("--skip-connect", action='store_true', default=False, help="Use skip connection in RGCN Unit")
parser.add_argument("--n-hidden", type=int, default=200, help="Number of hidden units")
parser.add_argument("--opn", type=str, default="sub", help="Operation for CompGCN")
parser.add_argument("--n-bases", type=int, default=100, help="Number of weight blocks per relation")
parser.add_argument("--n-basis", type=int, default=100, help="Number of basis vectors for CompGCN")
parser.add_argument("--n-layers", type=int, default=2, help="Number of propagation rounds")
parser.add_argument("--self-loop", action='store_true', default=False, help="Perform layer normalization in every layer")
parser.add_argument("--layer-norm", action='store_true', default=False, help="Perform layer normalization in every layer")
parser.add_argument("--weight", type=float, default=0.5, help="weight of static constraint")
parser.add_argument("--task-weight", type=float, default=0.7, help="weight of entity prediction task")
parser.add_argument("--discount", type=float, default=1, help="discount of weight of static constraint")
parser.add_argument("--angle", type=int, default=10, help="evolution speed")


# Training Configuration
parser.add_argument("--n-epochs", type=int, default=30, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    

# Evaluation Configuration
parser.add_argument("--evaluate-every", type=int, default=1, help="Perform evaluation every n epochs")


# Decoder Configuration
parser.add_argument("--decoder", type=str, default="seconvtranse", help="Decoder method")
parser.add_argument("--input-dropout", type=float, default=0.2, help="Input dropout for decoder")
parser.add_argument("--hidden-dropout", type=float, default=0.2, help="Hidden dropout for decoder")
parser.add_argument("--feat-dropout", type=float, default=0.2, help="Feature dropout for decoder")


# History Configuration
parser.add_argument("--history-len", type=int, default=9, help="History length")
parser.add_argument("--history-rate", type=float, default=0.3, help="History rate")


# Diachronic Semantic Encoder Configuration
parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
parser.add_argument('--num-k', type=int, default=5, help="Number of neighbors for DSEP")
parser.add_argument('--model-type', type=str, required=True, help="Type of pre-trained model (e.g., bert, t5)")
parser.add_argument('--plm', type=str, required=True, help="Pre-trained language model for DSEP")


args = parser.parse_args()
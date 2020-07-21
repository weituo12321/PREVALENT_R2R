import argparse
import os
import torch

def create_folders(path):
    """ recursively create folders """
    if not os.path.isdir(path):
        while True:
            try:
                os.makedirs(path)
            except:
                pass
                time.sleep(1)
            else:
                break

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        # General
        self.parser.add_argument('--iters', type=int, default=100000)
        self.parser.add_argument('--name', type=str, default='default')
        self.parser.add_argument('--train', type=str, default='speaker')

        # Data preparation
        self.parser.add_argument('--maxInput', type=int, default=80, help="max input instruction")
        self.parser.add_argument('--maxDecode', type=int, default=120, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=20, help='Max Action sequence')
        self.parser.add_argument('--batchSize', type=int, default=64)
        self.parser.add_argument('--ignoreid', type=int, default=-100)
        self.parser.add_argument('--feature_size', type=int, default=2048)
        self.parser.add_argument("--loadOptim",action="store_const", default=False, const=True)

        # Load the model from
        self.parser.add_argument("--speaker", default=None)
        self.parser.add_argument("--listener", default=None)
        self.parser.add_argument("--load", type=str, default=None)

        # More Paths from
        self.parser.add_argument("--aug", default=None)

        # Listener Model Config
        self.parser.add_argument("--zeroInit", dest='zero_init', action='store_const', default=False, const=True)
        self.parser.add_argument("--mlWeight", dest='ml_weight', type=float, default=0.05)
        self.parser.add_argument("--teacherWeight", dest='teacher_weight', type=float, default=1.)
        self.parser.add_argument("--accumulateGrad", dest='accumulate_grad', action='store_const', default=False, const=True)
        self.parser.add_argument("--features", type=str, default='imagenet')

        # Env Dropout Param
        self.parser.add_argument('--featdropout', type=float, default=0.3)

        # SSL configuration
        self.parser.add_argument("--selfTrain", dest='self_train', action='store_const', default=False, const=True)

        # Submision configuration
        self.parser.add_argument("--candidates", type=int, default=1)
        self.parser.add_argument("--paramSearch", dest='param_search', action='store_const', default=False, const=True)
        self.parser.add_argument("--submit", action='store_const', default=False, const=True)
        self.parser.add_argument("--beam", action="store_const", default=False, const=True)
        self.parser.add_argument("--alpha", type=float, default=0.5)

        # Training Configurations
        self.parser.add_argument('--optim', type=str, default='rms')    # rms, adam
        self.parser.add_argument('--lr', type=float, default=0.0001, help="The learning rate")
        self.parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--feedback', type=str, default='sample',
                            help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``')
        self.parser.add_argument('--teacher', type=str, default='final',
                            help="How to get supervision. one of ``next`` and ``final`` ")
        self.parser.add_argument('--epsilon', type=float, default=0.1)

        # Model hyper params:
        self.parser.add_argument('--rnnDim', dest="rnn_dim", type=int, default=512)
        self.parser.add_argument('--critic_dim', dest="critic_dim", type=int, default=512)
        self.parser.add_argument('--wemb', type=int, default=256)
        self.parser.add_argument('--aemb', type=int, default=64)
        self.parser.add_argument('--proj', type=int, default=512)
        self.parser.add_argument("--fast", dest="fast_train", action="store_const", default=False, const=True)
        self.parser.add_argument("--valid", action="store_const", default=False, const=True)
        self.parser.add_argument("--candidate", dest="candidate_mask",
                                 action="store_const", default=False, const=True)

        self.parser.add_argument("--bidir", type=bool, default=True)    # This is not full option
        self.parser.add_argument("--encode", type=str, default="word")  # sub, word, sub_ctx
        self.parser.add_argument("--subout", dest="sub_out", type=str, default="tanh")  # tanh, max
        self.parser.add_argument("--attn", type=str, default="soft")    # soft, mono, shift, dis_shift

        self.parser.add_argument("--angleFeatSize", dest="angle_feat_size", type=int, default=4)
        self.parser.add_argument('--philly', type=bool, default=False, help='if use philly')
        self.parser.add_argument('--update_bert', type=bool, default=False, help='if update bert encoder')
        self.parser.add_argument('--include_vision', type=bool, default=False, help='if include vision')
        self.parser.add_argument('--use_dropout_vision', type=bool, default=False, help='if use dropout vision')
        self.parser.add_argument("--encoderType", type=str, default="EncoderLSTM")    # EncoderLSTM,DicEncoder
        self.parser.add_argument("--schedule_ratio", type=float, default=-1)    # schedule sampling

        # For dicencoder

        self.parser.add_argument('--d_hidden_size', dest='d_hidden_size', default=1024, type=int, help='decoder hidden_size')
        self.parser.add_argument('--d_ctx_size', dest='d_ctx_size', default=2048, type=int, help='ctx hidden_size')
        self.parser.add_argument('--d_enc_hidden_size', dest='d_enc_hidden_size', default=768, type=int, help='encoder hidden_size')
        self.parser.add_argument('--d_dropout_ratio', dest='d_dropout_ratio', default=0.4, type=float, help='dropout_ratio')
        self.parser.add_argument('--d_bidirectional', dest='d_bidirectional', type=bool, default=True, help='bidirectional')
        self.parser.add_argument('--d_transformer_update', dest='d_transformer_update', type=bool, default=False, help='update Bert')
        self.parser.add_argument('--d_update_add_layer', dest='d_update_add_layer', type=bool, default=False, help='update fusion layer in Bert')
        self.parser.add_argument('--d_bert_n_layers', dest='d_bert_n_layers', type=int, default=1, help='bert_n_layers')
        self.parser.add_argument('--d_reverse_input', dest='d_reverse_input', type=bool, default=True, help='reverse')
        self.parser.add_argument('--d_top_lstm', dest='d_top_lstm', type=bool, default=True, help='add lstm to the top of transformers')
        self.parser.add_argument('--d_vl_layers', dest='d_vl_layers', type=int, default=4, help='vl_layers')
        self.parser.add_argument('--d_la_layers', dest='d_la_layers', type=int, default=9, help='la_layers')
        self.parser.add_argument('--d_bert_type', dest='d_bert_type', type=str, default="small", help='small or large')
        self.parser.add_argument('--pretrain_model_name', dest='pretrain_model_name', type=str, default=None, help='the name of pretrained model')



        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument("--normalize", dest="normalize_loss", default="total", type=str, help='batch or total')

        self.args = self.parser.parse_args()

        if self.args.optim == 'rms':
            print("Optimizer: Using RMSProp")
            self.args.optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            print("Optimizer: Using Adam")
            self.args.optimizer = torch.optim.Adam
        elif self.args.optim == 'sgd':
            print("Optimizer: sgd")
            self.args.optimizer = torch.optim.SGD
        else:
            assert False

param = Param()
args = param.args
args.TRAIN_VOCAB = 'tasks/R2R/data/train_vocab.txt'
args.TRAINVAL_VOCAB = 'tasks/R2R/data/trainval_vocab.txt'

args.IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
args.CANDIDATE_FEATURES = 'img_features/ResNet-152-candidate.tsv'
args.features_fast = 'img_features/ResNet-152-imagenet-fast.tsv'
args.log_dir = 'snap/%s' % args.name

if args.philly:
    new_logdir = os.path.join(os.getenv('PT_OUTPUT_DIR'), 'snap/%s' % args.name)
    args.log_dir = new_logdir

if not os.path.exists(args.log_dir):
    if args.philly:
        create_folders(args.log_dir)
    else:
        os.makedirs(args.log_dir)
#DEBUG_FILE = open(os.path.join('snap', args.name, "debug.log"), 'w')


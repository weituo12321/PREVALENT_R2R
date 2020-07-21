import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,BTokenizer,padding_idx,timeSince,preprocess_get_pano_states,current_best
from env import R2RBatch
from model import EncoderLSTM, AttnDecoderLSTM, R2RAttnDecoderLSTM
from r2rmodel import BertEncoder,DicEncoder
from r2rpretrain_class import DicAddActionPreTrain
from pytorch_transformers import BertForMaskedLM,BertTokenizer
from agent import Seq2SeqAgent
from feature import Feature
from eval import Evaluation
import json
import copy
import pdb



# For philly
philly = False

def save_best_model(best_model, SNAPSHOT_DIR, model_prefix, split_string, best_model_iter):
    """ Save the current best model """
    enc_path = '%s%s_%s_enc_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, best_model_iter)
    dec_path = '%s%s_%s_dec_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, best_model_iter)

    torch.save(best_model['encoder'], enc_path)
    torch.save(best_model['decoder'], dec_path)

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


TRAIN_VOCAB = 'tasks/NDH/data/train_vocab.txt'
TRAINVAL_VOCAB = 'tasks/NDH/data/trainval_vocab.txt'
RESULT_DIR = 'tasks/NDH/results/'
SNAPSHOT_DIR = 'tasks/NDH/snapshots/'
PLOT_DIR = 'tasks/NDH/plots/'

if philly:
    RESULT_DIR = os.path.join(os.getenv('PT_OUTPUT_DIR'), 'tasks/NDH/results/')
    PLOT_DIR = os.path.join(os.getenv('PT_OUTPUT_DIR'),'tasks/NDH/plots/')
    SNAPSHOT_DIR = os.path.join(os.getenv('PT_OUTPUT_DIR'), 'tasks/NDH/snapshots/')
    TRAIN_VOCAB = os.path.join(SNAPSHOT_DIR, 'train_vocab.txt')
    TRAINVAL_VOCAB = os.path.join(SNAPSHOT_DIR, 'trainval_vocab.txt')
    print("using philly, output are rest")

    print('RESULT_DIR', RESULT_DIR)
    print('PLOT_DIR', PLOT_DIR)
    print('SNAPSHOT_DIR', SNAPSHOT_DIR)
    print('TRAIN_VOC', TRAIN_VOCAB)

IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
FEATURE_SIZE = 2048
FEATURE_ALL_SIZE = 2176 # 2048

# Training settings.
agent_type = 'seq2seq'

# Fixed params from MP.
features = IMAGENET_FEATURES
batch_size = 100
word_embedding_size = 256
action_embedding_size = 32
target_embedding_size = 32
hidden_size = 512
bidirectional = False
dropout_ratio = 0.5
learning_rate = 0.0001
weight_decay = 0.0005

def train(train_env, encoder, decoder, n_iters, path_type, history, feedback_method, max_episode_len, MAX_INPUT_LENGTH, model_prefix,
    log_every=100, val_envs=None, args=None):
    ''' Train on training set, validating on both seen and unseen. '''
    if val_envs is None:
        val_envs = {}

    if agent_type == 'seq2seq':
        agent = Seq2SeqAgent(train_env, "", encoder, decoder, max_episode_len, path_type=args.path_type,args=args)
    else:
        sys.exit("Unrecognized agent_type '%s'" % agent_type)
    print('Training a %s agent with %s feedback' % (agent_type, feedback_method))
    if args.optm == 'Adam':
        optim_func = optim.Adam
    elif args.optm == 'Adamax':
        optim_func = optim.Adamax

    encoder_optimizer = optim_func(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    decoder_optimizer = optim_func(decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    data_log = defaultdict(list)
    start = time.time()

    best_model = {
        'iter': -1,
        'encoder': copy.deepcopy(agent.encoder.state_dict()),
        'decoder': copy.deepcopy(agent.decoder.state_dict()),
    }
    best_dr = 0
    best_spl = 0
    best_iter = 0
    myidx = 0

    for idx in range(0, n_iters, log_every):

        interval = min(log_every,n_iters-idx)
        iter = idx + interval
        data_log['iteration'].append(iter)

        myidx += interval
        print("PROGRESS: {}%".format(round((myidx) * 100 / n_iters, 4)))

        # Train for log_every interval
        agent.train(encoder_optimizer, decoder_optimizer, interval, feedback=feedback_method)
        train_losses = np.array(agent.losses)
        assert len(train_losses) == interval
        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg

        # Run validation
        for env_name, (env, evaluator) in val_envs.items():
            agent.env = env
            agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, env_name, iter)
            # Get validation loss under the same conditions as training
            agent.test(use_dropout=True, feedback=feedback_method, allow_cheat=True)
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            data_log['%s loss' % env_name].append(val_loss_avg)
            # Get validation distance from goal under test evaluation conditions
            agent.test(use_dropout=False, feedback='argmax')
            agent.write_results()
            score_summary, _ = evaluator.score(agent.results_path)
            loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
            for metric, val in score_summary.items():
                data_log['%s %s' % (env_name, metric)].append(val)
                if metric in ['success_rate', 'oracle success_rate', 'oracle path_success_rate', 'dist_to_end_reduction','sc_dr']:
                    loss_str += ', %s: %.3f' % (metric, val)


        eval_spl = current_best(data_log, -1, 'spl_unseen')
        eval_dr = current_best(data_log, -1, 'dr_unseen')
        if eval_dr > best_dr:
            best_dr = eval_dr
            best_iter = iter
            best_model['iter'] = iter
            best_model['encoder'] = copy.deepcopy(agent.encoder.state_dict())
            best_model['decoder'] = copy.deepcopy(agent.decoder.state_dict())

        if eval_spl>best_spl:
            best_spl=eval_spl
            loss_str+=' bestSPL'

        agent.env = train_env

        print('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str))

        print("EVALERR: {}%".format(best_dr))


        df = pd.DataFrame(data_log)
        df.set_index('iteration')
        df_path = '%s%s-log.csv' % (PLOT_DIR, model_prefix)
        write_num = 0
        while (write_num < 10):
            try:
                df.to_csv(df_path)
                break
            except:
                write_num += 1

        #split_string = "-".join(train_env.splits)
        #enc_path = '%s%s_%s_enc_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
        #dec_path = '%s%s_%s_dec_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
        #agent.save(enc_path, dec_path)
    split_string = "-".join(train_env.splits)
    save_best_model(best_model, SNAPSHOT_DIR,model_prefix, split_string, best_iter)

def setup(action_space=-1, navigable_locs_path=None):
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(RESULT_DIR):
        create_folders(RESULT_DIR)
    if not os.path.exists(PLOT_DIR):
        create_folders(PLOT_DIR)
    if not os.path.exists(SNAPSHOT_DIR):
        create_folders(SNAPSHOT_DIR)
    if not os.path.exists(navigable_locs_path):
        create_folders(navigable_locs_path)

    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train', 'val_seen', 'val_unseen']), TRAINVAL_VOCAB)

    if navigable_locs_path:
        #if philly:
        #    navigable_locs_path = os.path.join(os.getenv('PT_OUTPUT_DIR'), "tasks/NDH/data")
        #    if not os.path.exists(navigable_locs_path):
        #        create_folders(navigable_locs_path)

        navigable_locs_path += '/navigable_locs.json'

        print('navigable_locs_path', navigable_locs_path)
    #preprocess_get_pano_states(navigable_locs_path)
    global nav_graphs
    nav_graphs = None
    if action_space == -1:  # load navigable location cache
        with open(navigable_locs_path, 'r') as f:
            nav_graphs = json.load(f)
    return nav_graphs


def test_submission(path_type, max_episode_len, history, MAX_INPUT_LENGTH, feedback_method, n_iters, model_prefix, blind):
    ''' Train on combined training and validation sets, and generate test submission. '''

    setup()

    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAINVAL_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    train_env = R2RBatch(features, batch_size=batch_size, splits=['train', 'val_seen', 'val_unseen'], tokenizer=tok,
                         path_type=path_type, history=history, blind=blind)

    # Build models and train
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx,
                  dropout_ratio, bidirectional=bidirectional).cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                  action_embedding_size, hidden_size, dropout_ratio).cuda()
    train(train_env, encoder, decoder, n_iters, path_type, history, feedback_method, max_episode_len, MAX_INPUT_LENGTH, model_prefix)

    # Generate test submission
    test_env = R2RBatch(features, batch_size=batch_size, splits=['test'], tokenizer=tok,
                        path_type=path_type, history=history, blind=blind)
    agent = Seq2SeqAgent(test_env, "", encoder, decoder, max_episode_len)
    agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, 'test', 20000)
    agent.test(use_dropout=False, feedback='argmax')
    agent.write_results()


# NOTE: only available to us, now, for writing the paper.
def train_test(path_type, max_episode_len, history, MAX_INPUT_LENGTH, feedback_method, n_iters, model_prefix, blind):
    ''' Train on the training set, and validate on the test split. '''

    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAINVAL_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    train_env = R2RBatch(features, batch_size=batch_size, splits=['train', 'val_seen', 'val_unseen'], tokenizer=tok,
                         path_type=path_type, history=history, blind=blind)

    # Creat validation environments
    val_envs = {split: (R2RBatch(features, batch_size=batch_size, splits=[split],
                                 tokenizer=tok, path_type=path_type, history=history, blind=blind),
                        Evaluation([split], path_type=path_type)) for split in ['test']}

    # Build models and train
    enc_hidden_size = hidden_size // 2 if bidirectional else hidden_size
    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx,
                          dropout_ratio, bidirectional=bidirectional).cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                              action_embedding_size, hidden_size,dropout_ratio).cuda()
    train(train_env, encoder, decoder, n_iters, path_type, history, feedback_method, max_episode_len, MAX_INPUT_LENGTH,
          model_prefix, val_envs=val_envs)


def train_val(path_type, max_episode_len, history, MAX_INPUT_LENGTH, feedback_method, n_iters, model_prefix, blind, args):
    ''' Train on the training set, and validate on seen and unseen splits. '''

    nav_graphs = setup(args.action_space, args.navigable_locs_path)
    # Create a batch training environment that will also preprocess text
    use_bert = (args.encoder_type in ['bert','vlbert'])  # for tokenizer and dataloader
    if use_bert:
        tok = BTokenizer(MAX_INPUT_LENGTH)
    else:
        vocab = read_vocab(TRAIN_VOCAB)
        tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    #train_env = R2RBatch(features, batch_size=batch_size, splits=['train'], tokenizer=tok,
    #                     path_type=path_type, history=history, blind=blind)

    feature_store = Feature(features, args.panoramic)
    train_env = R2RBatch(feature_store, nav_graphs, args.panoramic,args.action_space,batch_size=args.batch_size, splits=['train'], tokenizer=tok,
                         path_type=path_type, history=history, blind=blind)

    # Creat validation environments
    #val_envs = {split: (R2RBatch(features, batch_size=batch_size, splits=[split],
    #            tokenizer=tok, path_type=path_type, history=history, blind=blind),
    #            Evaluation([split], path_type=path_type)) for split in ['val_seen', 'val_unseen']}

    val_envs = {split: (R2RBatch(feature_store,nav_graphs, args.panoramic, args.action_space,batch_size=args.batch_size, splits=[split],
                tokenizer=tok, path_type=path_type, history=history, blind=blind),
                Evaluation([split], path_type=path_type)) for split in ['val_seen', 'val_unseen']}

    # Build models and train
    #enc_hidden_size = hidden_size//2 if bidirectional else hidden_size

    if args.encoder_type == 'vlbert':
        if args.pretrain_model_name is not None:
            print("Using the pretrained lm model from %s" %(args.pretrain_model_name))
            encoder = DicEncoder(FEATURE_ALL_SIZE,args.enc_hidden_size, args.hidden_size, args.dropout_ratio, args.bidirectional, args.transformer_update, args.bert_n_layers, args.reverse_input, args.top_lstm,args.vl_layers,args.la_layers,args.bert_type)
            premodel = DicAddActionPreTrain.from_pretrained(args.pretrain_model_name)
            encoder.bert = premodel.bert
            encoder.drop = nn.Dropout(p=args.dropout_ratio)
            encoder.bert._resize_token_embeddings(len(tok)) # remember to resize tok embedding size
            encoder.bert.update_lang_bert, encoder.bert.config.update_lang_bert = args.transformer_update, args.transformer_update
            encoder.bert.update_add_layer, encoder.bert.config.update_add_layer = args.update_add_layer, args.update_add_layer
            encoder = encoder.cuda()

        else:
            encoder = DicEncoder(FEATURE_ALL_SIZE,args.enc_hidden_size, args.hidden_size, args.dropout_ratio, args.bidirectional, args.transformer_update, args.bert_n_layers, args.reverse_input, args.top_lstm,args.vl_layers,args.la_layers,args.bert_type).cuda()
            encoder.bert._resize_token_embeddings(len(tok)) # remember to resize tok embedding size

    elif args.encoder_type == 'bert':
        if args.pretrain_model_name is not None:
            print("Using the pretrained lm model from %s" %(args.pretrain_model_name))
            encoder = BertEncoder(args.enc_hidden_size, args.hidden_size, args.dropout_ratio, args.bidirectional, args.transformer_update, args.bert_n_layers, args.reverse_input, args.top_lstm, args.bert_type)
            premodel = BertForMaskedLM.from_pretrained(args.pretrain_model_name)
            encoder.bert = premodel.bert
            encoder.drop = nn.Dropout(p=args.dropout_ratio)
            encoder.bert._resize_token_embeddings(len(tok)) # remember to resize tok embedding size
            #encoder.bert.update_lang_bert, encoder.bert.config.update_lang_bert = args.transformer_update, args.transformer_update
            #encoder.bert.update_add_layer, encoder.bert.config.update_add_layer = args.update_add_layer, args.update_add_layer
            encoder = encoder.cuda()
            pdb.set_trace()
        else:
            encoder = BertEncoder(args.enc_hidden_size, args.hidden_size, args.dropout_ratio, args.bidirectional, args.transformer_update, args.bert_n_layers, args.reverse_input, args.top_lstm, args.bert_type).cuda()
            encoder.bert._resize_token_embeddings(len(tok))
    else:
        enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
        encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx,
                            dropout_ratio, bidirectional=bidirectional).cuda()


    #decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
    #              action_embedding_size, args.hidden_size, args.dropout_ratio).cuda()
    ctx_hidden_size = args.enc_hidden_size * (2 if args.bidirectional else 1)
    if use_bert and not args.top_lstm:
        ctx_hidden_size = 768

    decoder = R2RAttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                  action_embedding_size, ctx_hidden_size, args.hidden_size, args.dropout_ratio,FEATURE_SIZE, args.panoramic,args.action_space,args.dec_h_type).cuda()


    train(train_env, encoder, decoder, n_iters,
          path_type, history, feedback_method, max_episode_len, MAX_INPUT_LENGTH, model_prefix, val_envs=val_envs, args=args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_type', type=str, required=True,
                        help='planner_path, player_path, or trusted_path')
    parser.add_argument('--history', type=str, required=True,
                        help='none, target, oracle_ans, nav_q_oracle_ans, or all')
    parser.add_argument('--feedback', type=str, required=True,
                        help='teacher or sample')
    parser.add_argument('--eval_type', type=str, required=True,
                        help='val or test')
    parser.add_argument('--blind', action='store_true', required=False,
                        help='whether to replace the ResNet encodings with zero vectors at inference time')
    parser.add_argument('--encoder_type', dest='encoder_type', type=str, default='bert', help='lstm transformer bert or gpt')
    parser.add_argument('--hidden_size', dest='hidden_size', default=1024, type=int, help='decoder hidden_size')
    parser.add_argument('--enc_hidden_size', dest='enc_hidden_size', default=1024, type=int, help='encoder hidden_size')
    parser.add_argument('--dropout_ratio', dest='dropout_ratio', default=0.4, type=float, help='dropout_ratio')
    parser.add_argument('--bidirectional', dest='bidirectional', type=bool, default=True, help='bidirectional')
    parser.add_argument('--transformer_update', dest='transformer_update', type=bool, default=False, help='update Bert')
    parser.add_argument('--update_add_layer', dest='update_add_layer', type=bool, default=False, help='update fusion layer in Bert')
    parser.add_argument('--bert_n_layers', dest='bert_n_layers', type=int, default=1, help='bert_n_layers')
    parser.add_argument('--reverse_input', dest='reverse_input', type=bool, default=True, help='reverse')
    parser.add_argument('--top_lstm', dest='top_lstm', type=bool, default=True, help='add lstm to the top of transformers')
    parser.add_argument('--vl_layers', dest='vl_layers', type=int, default=1, help='vl_layers')
    parser.add_argument('--la_layers', dest='la_layers', type=int, default=1, help='la_layers')
    parser.add_argument('--bert_type', dest='bert_type', type=str, default="small", help='small or large')
    parser.add_argument('--batch_size', dest='batch_size', default=20, type=int, help='batch size')
    parser.add_argument('--action_space', dest='action_space', type=int, default=-1, help='6 or -1(navigable viewpoints)')
    parser.add_argument('--dec_h_type', dest='dec_h_type', type=str, default='vc', help='none or vc')
    parser.add_argument('--schedule_ratio', dest='schedule_ratio', default=0.2, type=float, help='ratio for sample or teacher')
    parser.add_argument('--navigable_locs_path', type=str, default='tasks/NDH/data', help='navigable graphs')
    parser.add_argument('--panoramic', dest='panoramic', type=bool, default=True, help='panoramic img')
    parser.add_argument('--optm', dest='optm', default='Adamax', type=str, help='Adam, Adamax, RMSprop')
    parser.add_argument('--learning_rate', dest='learning_rate', default=5e-05, type=float, help='learning_rate')
    parser.add_argument('--weight_decay', dest='weight_decay', default=0.0005, type=float, help='weight_decay')
    parser.add_argument('--n_iters', dest='n_iters', default=20000, type=int, help='n_iters')
    parser.add_argument('--max_episode_len', dest='max_episode_len', default=40, type=int, help='max_episode_len')
    parser.add_argument('--max_input_length', dest='max_input_length', default=300, type=int, help='max_input_length')
    parser.add_argument('--pretrain_model_name', dest='pretrain_model_name', type=str, default=None, help='the name of pretrained model')


    args = parser.parse_args()

    assert args.path_type in ['planner_path', 'player_path', 'trusted_path']
    assert args.history in ['none', 'target', 'oracle_ans', 'nav_q_oracle_ans', 'all']
    assert args.feedback in ['sample', 'teacher']
    assert args.eval_type in ['val', 'test']

    blind = args.blind

    # Set default args.
    path_type = args.path_type
    # In MP, max_episode_len = 20 while average hop range [4, 7], e.g. ~3x max.
    # max_episode_len has to account for turns; this heuristically allowed for about 1 turn per hop.
    if path_type == 'planner_path':
        max_episode_len = 20  # [1, 6], e.g., ~3x max
    else:
        #max_episode_len = 80  # [2, 41], e.g., ~2x max (120 ~3x) (80 ~2x) [for player/trusted paths]
        max_episode_len = 40  # [2, 41], e.g., ~2x max (120 ~3x) (80 ~2x) [for player/trusted paths]

    # Input settings.
    history = args.history
    # In MP, MAX_INPUT_LEN = 80 while average utt len is 29, e.g., a bit less than 3x avg.
    if history == 'none':
        MAX_INPUT_LENGTH = 1  # [<EOS>] fixed length.
    elif history == 'target':
        MAX_INPUT_LENGTH = 3  # [<TAR> target <EOS>] fixed length.
    elif history == 'oracle_ans':
        MAX_INPUT_LENGTH = 70  # 16.16+/-9.67 ora utt len, 35.5 at x2 stddevs. 71 is double that.
    elif history == 'nav_q_oracle_ans':
        #MAX_INPUT_LENGTH = 120  # 11.24+/-6.43 [plus Ora avg], 24.1 at x2 std. 71+48 ~~ 120 per QA doubles both.
        MAX_INPUT_LENGTH = 80  # 11.24+/-6.43 [plus Ora avg], 24.1 at x2 std. 71+48 ~~ 120 per QA doubles both.
    else:  # i.e., 'all'
        #MAX_INPUT_LENGTH = 120 * 6  # 4.93+/-3.21 turns -> 2.465+/-1.605 Q/A. 5.67 at x2 std. Call it 6 (real max 13).
        MAX_INPUT_LENGTH = 300


    # Training settings.
    feedback_method = args.feedback
    n_iters = args.n_iters

    # Model prefix to uniquely id this instance.
    model_prefix = '%s-seq2seq-%s-%s-%d-%s-imagenet' % (args.eval_type, history, path_type, max_episode_len, feedback_method)
    if blind:
        model_prefix += '-blind'

    if args.eval_type == 'val':
        train_val(path_type, max_episode_len, history, MAX_INPUT_LENGTH, feedback_method, n_iters, model_prefix, blind, args)
    else:
        train_test(path_type, max_episode_len, history, MAX_INPUT_LENGTH, feedback_method, n_iters, model_prefix, blind)

    # test_submission(path_type, max_episode_len, history, MAX_INPUT_LENGTH, feedback_method, n_iters, model_prefix, blind)

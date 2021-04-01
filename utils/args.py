#coding=utf-8
import sys
import argparse
from utils.constants import GK_EMB_DIM

def init_args(params, task='slu'):
    parser = argparse.ArgumentParser(allow_abbrev=False, conflict_handler='resolve')
    parser = add_argument_base(parser)
    parser = add_argument_dict[task](parser)
    opt = parser.parse_args(params)
    return opt

def add_argument_base(parser):
    #### General configuration ####
    parser.add_argument('--task', default='slu', help='task name, used in hyper-param path')
    parser.add_argument('--dataset', choices=['atis', 'snips'], default='atis', help='which dataset to experiment on')
    parser.add_argument('--labeled', type=float, default=1.0, help='ratio of labeled data during training')
    parser.add_argument('--unlabeled', type=float, default=1.0, help='ratio of unlabeled data during semi-supervised learning')
    parser.add_argument('--seed', default=999, type=int, help='Random seed')
    parser.add_argument('--deviceId', type=int, default=0, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    parser.add_argument('--testing', action='store_true', help='training or evaluation mode')
    parser.add_argument('--read_model_path', type=str, help='read pretrained model path')
    #### Training Hyperparams ####
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate at non-recurrent layers')
    parser.add_argument('--batchSize', default=20, type=int, help='Batch size')
    parser.add_argument('--test_batchSize', default=20, type=int, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-4, help='weight decay coefficient, should be larger when using bert')
    parser.add_argument('--layerwise_decay', type=float, default=1.0, help='layerwise decay rate for lr, used for bert')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup steps proportion')
    parser.add_argument('--lr_schedule', default='linear', choices=['constant', 'linear', 'cosine'], help='lr scheduler with warmup strategy')
    parser.add_argument('--eval_after_epoch', default=10, type=int, help='Start to evaluate after x epoch')
    parser.add_argument('--init_weight', type=float, default=0.2, help='weights will be initialized uniformly among [-init_weight, init_weight]')
    parser.add_argument('--max_epoch', type=int, default=50, help='terminate after maximum epochs')
    parser.add_argument('--max_norm', default=5., type=float, help='clip gradients')
    parser.add_argument('--beam', default=5, type=int, help='beam search size')
    parser.add_argument('--n_best', default=1, type=int, help='return n best results')
    return parser

def add_argument_slu(parser):
    parser.add_argument('--model_type', choices=['birnn', 'birnn+crf', 'focus'], default='focus')
    parser.add_argument('--use_bert', action='store_true', help='whether use bert as embeddings')
    parser.add_argument('--subword_aggregation', choices=['mean-pooling', 'max-pooling', 'attentive-pooling'],
            default='attentive-pooling', help='how to aggregate subword feats into word feats')
    parser.add_argument('--intent_method', choices=['head+tail', 'hiddenAttn'], default='hiddenAttn',
            help='how to perform intent classification')
    parser.add_argument('--cell', default='lstm', choices=['lstm', 'gru'], help='rnn cell choice')
    parser.add_argument('--emb_size', type=int, default=GK_EMB_DIM, help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers')
    return parser

def add_argument_nlg(parser):
    parser.add_argument('--model_type', choices=['sclstm', 'sclstm+copy'], default='sclstm+copy')
    parser.add_argument('--cell', default='lstm', choices=['lstm', 'gru'], help='rnn cell choice')
    parser.add_argument('--emb_size', type=int, default=GK_EMB_DIM, help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers')
    parser.add_argument('--slot_aggregation', choices=['mean-pooling', 'max-pooling', 'attentive-pooling'],
            default='attentive-pooling', help='how to aggregate the feats of a slot-value pair into slot-level feats')
    parser.add_argument('--slot_weight', default=1.0, type=float, help='coefficient for slot-controlled loss')
    return parser

def add_argument_lm(parser):
    parser.add_argument('--cell', default='lstm', choices=['lstm', 'gru'], help='rnn cell choice')
    parser.add_argument('--emb_size', type=int, default=GK_EMB_DIM, help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers')
    parser.add_argument('--decoder_tied', action='store_true', help='whether combine the weights of embedding and decoder')
    parser.add_argument('--surface_level', action='store_true', help='whether train a surface-level language model')
    return parser

def add_argument_dual_pseudo_labeling(parser):
    parser.add_argument('--read_slu_model_path', type=str, help='read pretrained slu model path')
    parser.add_argument('--read_nlg_model_path', type=str, help='read pretrained nlg model path')
    parser.add_argument('--deviceIds', type=int, nargs='+', help='Use device list for slu and nlg model: -1 -> cpu ; the index of gpu o.w.')
    parser.add_argument('--discount', type=float, default=1.0, help='discount factor for pseudo samples')
    parser.add_argument('--conf_schedule', default='linear', choices=['constant', 'linear'], help='confidence scheduler for pseudo samples')
    parser.add_argument('--cycle_choice', default='slu+nlg', choices=['slu', 'nlg', 'slu+nlg'], help='use which model to generate pseudo samples')
    return parser

def add_argument_dual_learning(parser):
    parser.add_argument('--read_slu_model_path', type=str, help='read pretrained slu model path')
    parser.add_argument('--read_nlg_model_path', type=str, help='read pretrained nlg model path')
    parser.add_argument('--read_lm_path', type=str, help='read pretrained language model path')
    parser.add_argument('--deviceIds', type=int, nargs='+', help='Use device list for slu and nlg model: -1 -> cpu ; the index of gpu o.w.')
    parser.add_argument('--sample', type=int, default=5, help='sampling size during dual learning')
    parser.add_argument('--alpha', type=float, default=0.5, help='slu coefficient combining validity and reconstruction reward')
    parser.add_argument('--beta', type=float, default=0.5, help='nlg coefficient combining validity and reconstruction reward')
    parser.add_argument('--cycle_choice', default='slu+nlg', choices=['slu', 'nlg', 'slu+nlg'], help='use which model to generate pseudo samples')
    return parser

def add_argument_dual_plus_pseudo(parser):
    parser.add_argument('--read_slu_model_path', type=str, help='read pretrained slu model path')
    parser.add_argument('--read_nlg_model_path', type=str, help='read pretrained nlg model path')
    parser.add_argument('--read_lm_path', type=str, help='read pretrained language model path')
    parser.add_argument('--deviceIds', type=int, nargs='+', help='Use device list for slu and nlg model: -1 -> cpu ; the index of gpu o.w.')
    parser.add_argument('--sample', type=int, default=5, help='sampling size during dual learning')
    parser.add_argument('--alpha', type=float, default=0.5, help='slu coefficient combining validity and reconstruction reward')
    parser.add_argument('--beta', type=float, default=0.5, help='nlg coefficient combining validity and reconstruction reward')
    parser.add_argument('--discount', type=float, default=1.0, help='discount factor for pseudo samples')
    parser.add_argument('--conf_schedule', default='linear', choices=['constant', 'linear'], help='confidence scheduler for pseudo samples')
    parser.add_argument('--cycle_choice', default='slu+nlg', choices=['slu', 'nlg', 'slu+nlg'], help='use which model to generate pseudo samples')
    return parser

add_argument_dict = {
    'slu': add_argument_slu,
    'nlg': add_argument_nlg,
    'lm': add_argument_lm,
    'dual_pseudo_labeling': add_argument_dual_pseudo_labeling,
    'dual_learning': add_argument_dual_learning,
    'dual_plus_pseudo': add_argument_dual_plus_pseudo
}

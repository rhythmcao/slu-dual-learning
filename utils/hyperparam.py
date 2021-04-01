#coding=utf8
import os, sys

EXP = 'exp'

def hyperparam_path(opt, task='slu'):
    if opt.testing and opt.read_model_path is not None:
        return opt.read_model_path
    exp_path = hyperparam_path_dict[task](opt)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    return exp_path

def hyperparam_base(opt, bert=False):
    exp_name = ''
    exp_name += 'dp_%s__' % (opt.dropout)
    exp_name += 'lr_%s__' % (opt.lr) if not bert else \
        'lr_%s_ld_%s_wr_%s_ls_%s__' % (opt.lr, opt.layerwise_decay, opt.warmup_ratio, opt.lr_schedule)
    exp_name += 'l2_%s__' % (opt.l2)
    exp_name += 'mn_%s__' % (opt.max_norm)
    exp_name += 'bs_%s__' % (opt.batchSize)
    exp_name += 'me_%s__' % (opt.max_epoch)
    exp_name += 'bm_%s__' % (opt.beam)
    exp_name += 'nb_%s' % (opt.n_best)
    return exp_name

def hyperparam_slu(opt):
    task_name = 'task_%s' % (opt.task)
    bert = 'bert-' if opt.use_bert else ''
    category_name = 'dataset_%s__%smodel_%s__labeled_%s' % \
        (opt.dataset, bert, opt.model_type, opt.labeled)

    exp_name = ''
    exp_name += 'cell_%s__' % (opt.cell)
    exp_name += 'emb_%s__' % (opt.emb_size)
    exp_name += 'hidden_%s_x_%s__' % (opt.hidden_size, opt.num_layers)
    exp_name += hyperparam_base(opt, opt.use_bert)
    return os.path.join(EXP, task_name, category_name, exp_name)

def hyperparam_nlg(opt):
    task_name = 'task_%s' % (opt.task)
    category_name = 'dataset_%s__model_%s__labeled_%s' % (opt.dataset, opt.model_type, opt.labeled)

    exp_name = ''
    exp_name += 'cell_%s__' % (opt.cell)
    exp_name += 'emb_%s__' % (opt.emb_size)
    exp_name += 'hidden_%s_x_%s__' % (opt.hidden_size, opt.num_layers)
    exp_name += 'sw_%s__' % (opt.slot_weight)
    exp_name += hyperparam_base(opt, False)
    return os.path.join(EXP, task_name, category_name, exp_name)

def hyperparam_lm(opt):
    task_name = 'task_%s' % (opt.task)
    category_name = 'dataset_%s__surface_%s' % (opt.dataset, opt.surface_level)

    exp_name = ''
    exp_name += 'cell_%s__' % (opt.cell)
    exp_name += 'emb_%s__' % (opt.emb_size)
    exp_name += 'hidden_%s_x_%s__' % (opt.hidden_size, opt.num_layers)
    exp_name += 'decTied__' if opt.decoder_tied else ''
    exp_name += hyperparam_base(opt, False)
    return os.path.join(EXP, task_name, category_name, exp_name)

def hyperparam_dual_pseudo_labeling(opt):
    task_name = 'task_%s' % (opt.task)
    category_name = 'dataset_%s__labeled_%s__unlabeled_%s' % (opt.dataset, opt.labeled, opt.unlabeled)

    exp_name = 'dis_%s__conf_%s__cycle_%s__' % (opt.discount, opt.conf_schedule, opt.cycle_choice)
    exp_name += 'wr_%s_ls_%s__' % (opt.warmup_ratio, opt.lr_schedule)
    exp_name += 'bs_%s__' % (opt.batchSize)
    exp_name += 'me_%s__' % (opt.max_epoch)
    exp_name += 'bm_%s__' % (opt.beam)
    exp_name += 'nb_%s' % (opt.n_best)
    return os.path.join(EXP, task_name, category_name, exp_name)

def hyperparam_dual_learning(opt):
    task_name = 'task_%s' % (opt.task)
    category_name = 'dataset_%s__labeled_%s__unlabeled_%s' % (opt.dataset, opt.labeled, opt.unlabeled)

    exp_name = 'sample_%s__alpha_%s__beta_%s__cycle_%s__' % (opt.sample, opt.alpha, opt.beta, opt.cycle_choice)
    exp_name += 'wr_%s_ls_%s__' % (opt.warmup_ratio, opt.lr_schedule)
    exp_name += 'bs_%s__' % (opt.batchSize)
    exp_name += 'me_%s__' % (opt.max_epoch)
    exp_name += 'bm_%s__' % (opt.beam)
    exp_name += 'nb_%s' % (opt.n_best)
    return os.path.join(EXP, task_name, category_name, exp_name)

def hyperparam_dual_plus_pseudo(opt):
    task_name = 'task_%s' % (opt.task)
    category_name = 'dataset_%s__labeled_%s__unlabeled_%s' % (opt.dataset, opt.labeled, opt.unlabeled)

    exp_name = 'dis_%s__conf_%s__' % (opt.discount, opt.conf_schedule)
    exp_name += 'sample_%s__alpha_%s__beta_%s__cycle_%s__' % (opt.sample, opt.alpha, opt.beta, opt.cycle_choice)
    exp_name += 'wr_%s_ls_%s__' % (opt.warmup_ratio, opt.lr_schedule)
    exp_name += 'bs_%s__' % (opt.batchSize)
    exp_name += 'me_%s__' % (opt.max_epoch)
    exp_name += 'bm_%s__' % (opt.beam)
    exp_name += 'nb_%s' % (opt.n_best)
    return os.path.join(EXP, task_name, category_name, exp_name)

hyperparam_path_dict = {
    'slu': hyperparam_slu,
    'nlg': hyperparam_nlg,
    'lm': hyperparam_lm,
    'dual_pseudo_labeling': hyperparam_dual_pseudo_labeling,
    'dual_learning': hyperparam_dual_learning,
    'dual_plus_pseudo': hyperparam_dual_plus_pseudo
}

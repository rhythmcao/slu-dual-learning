#coding=utf8
import argparse, os, sys, time, json, gc, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.args import init_args
from utils.initialization import *
from utils.vocab import Vocab
from utils.dataset import read_dataset, split_dataset
from utils.loss import set_celoss_function, set_scloss_function
from utils.optimization import set_optimizer
from utils.evaluator import Evaluator
from utils.batch import get_minibatch
from utils.constants import PAD
from utils.word2vec import load_embeddings
from utils.hyperparam import hyperparam_path
from models.slu.make_model import construct_models as slu_model_constructor
from models.nlg.make_model import construct_models as nlg_model_constructor
from models.language_model import LanguageModel as lm_constructor
from models.reward import RewardModel
from models.dual_learning import DualLearning
from scripts.decode import slu_decode, nlg_decode, slu_pseudo_labeling, nlg_pseudo_labeling

task = 'dual_plus_pseudo'

##### Arguments parsing and preparations #####
opt = init_args(params=sys.argv[1:], task=task)
exp_path = hyperparam_path(opt, task=task)
logger = set_logger(exp_path, testing=opt.testing)
set_random_seed(opt.seed)
slu_device, nlg_device = set_torch_device(opt.deviceIds[0]), set_torch_device(opt.deviceIds[1])
lm_device = nlg_device # slu may use bert model
logger.info("Initialization finished ...")
logger.info("Parameters: " + str(json.dumps(vars(opt))))
logger.info("Output path is: %s" % (exp_path))
logger.info("Random seed is set to: %d" % (opt.seed))
logger.info("Use GPU with index %s as target slu device" % (opt.deviceIds[0]) if opt.deviceIds[0] >= 0 else "Use CPU as target slu torch device")
logger.info("Use GPU with index %s as target nlg device" % (opt.deviceIds[1]) if opt.deviceIds[1] >= 0 else "Use CPU as target nlg torch device")

##### Vocab and Dataset Reader #####
slu_vocab, nlg_vocab = Vocab(dataset=opt.dataset, task='slu'), Vocab(dataset=opt.dataset, task='nlg')
lm_vocab = Vocab(dataset=opt.dataset, task='lm')
slu_evaluator, nlg_evaluator = Evaluator.get_evaluator_from_task(task='slu', vocab=slu_vocab), Evaluator.get_evaluator_from_task(task='nlg', vocab=nlg_vocab)

if not opt.testing:
    train_dataset, dev_dataset = read_dataset(opt.dataset, choice='train'), read_dataset(opt.dataset, choice='valid')
    labeled_dataset, unlabeled_dataset = split_dataset(train_dataset, opt.labeled)
    logger.info("Labeled/Unlabeled train and dev dataset size is: %s/%s and %s" % (len(labeled_dataset), len(unlabeled_dataset), len(dev_dataset)))
    unlabeled_dataset = labeled_dataset + unlabeled_dataset
test_dataset = read_dataset(opt.dataset, choice='test')
logger.info("Test dataset size is: %s" % (len(test_dataset)))

##### Model Construction and Init #####
if not opt.testing:
    params = vars(opt)
    json.dump(params, open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)
else:
    params = json.load(open(os.path.join(exp_path, 'params.json'), 'r'))
slu_params = json.load(open(os.path.join(params['read_slu_model_path'], 'params.json'), 'r'))
slu_model = slu_model_constructor(**slu_params).to(slu_device)
nlg_params = json.load(open(os.path.join(params['read_nlg_model_path'], 'params.json'), 'r'))
nlg_model = nlg_model_constructor(**nlg_params).to(nlg_device)
if not opt.testing:
    slu_ckpt = torch.load(open(os.path.join(params['read_slu_model_path'], 'model.pkl'), 'rb'), map_location=slu_device)
    slu_model.load_state_dict(slu_ckpt)
    logger.info("Load SLU model from path %s" % (params['read_slu_model_path']))
    nlg_ckpt = torch.load(open(os.path.join(params['read_nlg_model_path'], 'model.pkl'), 'rb'), map_location=nlg_device)
    nlg_model.load_state_dict(nlg_ckpt)
    logger.info("Load NLG model from path %s" % (params['read_nlg_model_path']))
    lm_params = json.load(open(os.path.join(params['read_lm_path'], 'params.json'), 'r')) 
    lm_model = lm_constructor(**lm_params).to(lm_device)
    lm_ckpt = torch.load(open(os.path.join(params['read_lm_path'], 'model.pkl'), 'rb'), map_location=lm_device)
    lm_model.load_state_dict(lm_ckpt)
    logger.info("Load language model from path %s" % (params['read_lm_path']))
    reward_model = RewardModel(params['dataset'], lm_model, lm_vocab, surface_level=lm_params['surface_level'], device=lm_device)
    train_model = DualLearning(slu_model, nlg_model, reward_model, slu_vocab, nlg_vocab, slu_evaluator, nlg_evaluator,
        alpha=params['alpha'], beta=params['beta'], sample=params['sample'], slu_device=slu_device, nlg_device=nlg_device)
else:
    slu_ckpt = torch.load(open(os.path.join(opt.read_model_path, 'slu_model.pkl'), 'rb'), map_location=slu_device)
    slu_model.load_state_dict(slu_ckpt)
    logger.info("Load SLU model from path %s" % (opt.read_model_path))
    nlg_ckpt = torch.load(open(os.path.join(opt.read_model_path, 'nlg_model.pkl'), 'rb'), map_location=nlg_device)
    nlg_model.load_state_dict(nlg_ckpt)
    logger.info("Load NLG model from path %s" % (opt.read_model_path))

##### Training and Decoding #####
if not opt.testing:
    nsentences = max(len(labeled_dataset), len(unlabeled_dataset))
    slot_loss_function = set_celoss_function(ignore_index=slu_vocab.slot2id[PAD])
    intent_loss_function = set_celoss_function()
    num_training_steps = ((nsentences + opt.batchSize - 1) // opt.batchSize) * opt.max_epoch
    num_warmup_steps = int(num_training_steps * opt.warmup_ratio)
    slu_optimizer, slu_scheduler = set_optimizer(train_model.slu_model, lr=slu_params['lr'], l2=slu_params['l2'],
        max_norm=slu_params['max_norm'], layerwise_decay=slu_params['layerwise_decay'],
        num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, lr_schedule=opt.lr_schedule)
    surface_loss_function = set_celoss_function(ignore_index=nlg_vocab.word2id[PAD])
    slot_control_function = set_scloss_function(slot_weight=nlg_params['slot_weight'])
    nlg_optimizer, nlg_scheduler = set_optimizer(train_model.nlg_model, lr=nlg_params['lr'], l2=nlg_params['l2'], max_norm=nlg_params['max_norm'], lr_schedule='constant')
    logger.info("Training starts at %s" % (time.asctime(time.localtime(time.time()))))
    slu_pseudo_train_index, nlg_pseudo_train_index = np.arange(len(unlabeled_dataset)), np.arange(len(unlabeled_dataset))
    slu_start_train_index, nlg_start_train_index = np.arange(len(unlabeled_dataset)), np.arange(len(unlabeled_dataset))
    labeled_train_index = np.arange(len(labeled_dataset))
    slu_coefficient, nlg_coefficient = 0.5, 0.5
    slu_best_result = {"losses": [], "iter": 0, "dev_slot": 0., "dev_intent": 0.,
        "test_slot": 0., "test_intent": 0.,}
    nlg_best_result = {"losses": [], "iter": 0, "dev_bleu": 0., "dev_slot": 0.,
        "test_bleu": 0., "test_slot": 0.,}
    for i in range(opt.max_epoch):
        np.random.shuffle(slu_pseudo_train_index)
        np.random.shuffle(nlg_pseudo_train_index)
        np.random.shuffle(slu_start_train_index)
        np.random.shuffle(nlg_start_train_index)
        np.random.shuffle(labeled_train_index)
        conf = opt.discount if opt.conf_schedule == 'constant' else opt.discount * (i + 1) / opt.max_epoch
        slu_pseudo_dataset = slu_pseudo_labeling(train_model.slu_model, slu_vocab, slu_evaluator, unlabeled_dataset, opt.test_batchSize, device=slu_device, beam=opt.beam) \
            if 'slu' in opt.cycle_choice else []
        logger.info('Generate %d pseudo samples with SLU model in epoch %d with confidence %s' % (len(slu_pseudo_dataset), i, conf))
        nlg_pseudo_dataset = nlg_pseudo_labeling(train_model.nlg_model, nlg_vocab, nlg_evaluator, unlabeled_dataset, opt.test_batchSize, device=nlg_device, beam=opt.beam) \
            if 'nlg' in opt.cycle_choice else []
        logger.info('Generate %d pseudo samples with NLG model in epoch %d with confidence %s' % (len(nlg_pseudo_dataset), i, conf))
        train_model.train()
        start_time, losses = time.time(), []
        for j in range(0, nsentences, opt.batchSize):
            slu_optimizer.zero_grad()
            nlg_optimizer.zero_grad()
            # pseudo samples generated by slu model
            if 'slu' in opt.cycle_choice and len(slu_pseudo_dataset) != 0:
                inputs, outputs, intents, lens, _ = get_minibatch(slu_pseudo_dataset, slu_vocab, task='slu',
                    data_index=slu_pseudo_train_index, index=j, batch_size=opt.batchSize, device=slu_device, use_bert=slu_params['use_bert'])
                slot_scores, intent_scores = train_model.slu_model(inputs, lens, outputs)
                slot_loss = slot_loss_function(slot_scores, outputs[:, 1:]) if 'crf' not in slu_params['model_type'] else slot_scores
                intent_loss = intent_loss_function(intent_scores, intents)
                slu_pseudo_loss = conf * (slu_coefficient * slot_loss + (1 - slu_coefficient) * intent_loss)
                slu_pseudo_loss.backward()
                intents, slots, slot_lens, lens, outputs, out_lens, slot_states, copy_tokens, _ = \
                    get_minibatch(slu_pseudo_dataset, nlg_vocab, task='nlg', data_index=slu_pseudo_train_index, index=j, batch_size=opt.batchSize, device=nlg_device)
                surface_scores, slot_hist = train_model.nlg_model(intents, slots, slot_lens, lens, outputs[:, :-1], slot_states, copy_tokens)
                surface_loss = surface_loss_function(surface_scores, outputs[:, 1:])
                control_loss = slot_control_function(slot_hist, out_lens)
                nlg_pseudo_loss = conf * (nlg_coefficient * surface_loss + (1 - nlg_coefficient) * control_loss)
                nlg_pseudo_loss.backward()
                losses.append([slu_pseudo_loss.item(), nlg_pseudo_loss.item()])
            # pseudo samples generated by nlg model
            if 'nlg' in opt.cycle_choice and len(nlg_pseudo_dataset) != 0:
                inputs, outputs, intents, lens, _ = get_minibatch(nlg_pseudo_dataset, slu_vocab, task='slu',
                    data_index=nlg_pseudo_train_index, index=j, batch_size=opt.batchSize, device=slu_device, use_bert=slu_params['use_bert'])
                slot_scores, intent_scores = train_model.slu_model(inputs, lens, outputs)
                slot_loss = slot_loss_function(slot_scores, outputs[:, 1:]) if 'crf' not in slu_params['model_type'] else slot_scores
                intent_loss = intent_loss_function(intent_scores, intents)
                slu_pseudo_loss = conf * (slu_coefficient * slot_loss + (1 - slu_coefficient) * intent_loss)
                slu_pseudo_loss.backward()
                intents, slots, slot_lens, lens, outputs, out_lens, slot_states, copy_tokens, _ = \
                    get_minibatch(nlg_pseudo_dataset, nlg_vocab, task='nlg', data_index=nlg_pseudo_train_index, index=j, batch_size=opt.batchSize, device=nlg_device)
                surface_scores, slot_hist = train_model.nlg_model(intents, slots, slot_lens, lens, outputs[:, :-1], slot_states, copy_tokens)
                surface_loss = surface_loss_function(surface_scores, outputs[:, 1:])
                control_loss = slot_control_function(slot_hist, out_lens)
                nlg_pseudo_loss = conf * (nlg_coefficient * surface_loss + (1 - nlg_coefficient) * control_loss)
                nlg_pseudo_loss.backward()
                losses.append([slu_pseudo_loss.item(), nlg_pseudo_loss.item()])
            # dual learning start from slu model
            if 'slu' in opt.cycle_choice and len(unlabeled_dataset) != 0:
                inputs, _, _, lens, raw_in = get_minibatch(unlabeled_dataset, slu_vocab, task='slu',
                    data_index=slu_start_train_index, index=j, batch_size=opt.batchSize, device=slu_device, use_bert=slu_params['use_bert'])
                slu_loss, nlg_loss = train_model(inputs, lens, raw_in, start_from='slu')
                slu_loss.backward()
                nlg_loss.backward()
                losses.append([slu_loss.item(), nlg_loss.item()])
            # dual learning start from nlg model
            if 'nlg' in opt.cycle_choice and len(unlabeled_dataset) != 0:
                intents, slots, slot_lens, lens, _, _, slot_states, copy_tokens, (raw_intents, raw_slots) = \
                    get_minibatch(unlabeled_dataset, nlg_vocab, task='nlg', data_index=nlg_start_train_index, index=j, batch_size=opt.batchSize, device=nlg_device)
                slu_loss, nlg_loss = train_model(intents, slots, slot_lens, lens, slot_states, copy_tokens, raw_intents, raw_slots, start_from='nlg')
                slu_loss.backward()
                nlg_loss.backward()
                losses.append([slu_loss.item(), nlg_loss.item()])
            # traditional supervised training
            inputs, outputs, intents, lens, _ = get_minibatch(labeled_dataset, slu_vocab, task='slu',
                data_index=labeled_train_index, index=j, batch_size=opt.batchSize, device=slu_device, use_bert=slu_params['use_bert'])
            slot_scores, intent_scores = train_model.slu_model(inputs, lens, outputs)
            slot_loss = slot_loss_function(slot_scores, outputs[:, 1:]) if 'crf' not in slu_params['model_type'] else slot_scores
            intent_loss = intent_loss_function(intent_scores, intents)
            slu_labeled_loss = slu_coefficient * slot_loss + (1 - slu_coefficient) * intent_loss
            slu_labeled_loss.backward()
            intents, slots, slot_lens, lens, outputs, out_lens, slot_states, copy_tokens, _ = \
                get_minibatch(labeled_dataset, nlg_vocab, task='nlg', data_index=labeled_train_index, index=j, batch_size=opt.batchSize, device=nlg_device)
            surface_scores, slot_hist = train_model.nlg_model(intents, slots, slot_lens, lens, outputs[:, :-1], slot_states, copy_tokens)
            surface_loss = surface_loss_function(surface_scores, outputs[:, 1:])
            control_loss = slot_control_function(slot_hist, out_lens)
            nlg_labeled_loss = nlg_coefficient * surface_loss + (1 - nlg_coefficient) * control_loss
            nlg_labeled_loss.backward()
            losses.append([slu_labeled_loss.item(), nlg_labeled_loss.item()])
            # backward and optimize
            train_model.pad_embedding_grad_zero()
            slu_optimizer.step()
            slu_scheduler.step()
            nlg_optimizer.step()
            nlg_scheduler.step()
        epoch_loss = np.mean(losses, axis=0)
        slu_best_result['losses'].append(epoch_loss[0])
        nlg_best_result['losses'].append(epoch_loss[1])
        logger.info('Training epoch : %d\tTime : %.4fs\tSLU/NLG Loss : %.5f/%.5f' % (i, time.time() - start_time, epoch_loss[0], epoch_loss[1]))
        gc.collect()
        torch.cuda.empty_cache()
        if i < opt.eval_after_epoch: continue

        ##### Evaluate on dev and test dataset #####
        start_time = time.time()
        dev_slot, dev_intent = slu_decode(train_model.slu_model, slu_vocab, slu_evaluator, dev_dataset, os.path.join(exp_path, 'slu_valid.iter' + str(i)),
            opt.test_batchSize, device=slu_device, beam=opt.beam, n_best=opt.n_best)
        logger.info('SLU Evaluation epoch : %d\tTime : %.4fs\tValid (slot : %.4f ; intent : %.4f)' \
                        % (i, time.time() - start_time, dev_slot, dev_intent))
        start_time = time.time()
        test_slot, test_intent = slu_decode(train_model.slu_model, slu_vocab, slu_evaluator, test_dataset, os.path.join(exp_path, 'slu_test.iter' + str(i)),
            opt.test_batchSize, device=slu_device, beam=opt.beam, n_best=opt.n_best)
        logger.info('SLU Evaluation epoch : %d\tTime : %.4fs\tTest (slot : %.4f ; intent : %.4f)' \
                            % (i, time.time() - start_time, test_slot, test_intent))
        ##### Pick best result on dev and save #####
        prev_best = slu_coefficient * slu_best_result['dev_slot'] + (1 - slu_coefficient) * slu_best_result['dev_intent']
        if slu_coefficient * dev_slot + (1 - slu_coefficient) * dev_intent > prev_best:
            torch.save(train_model.slu_model.state_dict(), open(os.path.join(exp_path, 'slu_model.pkl'), 'wb'))
            slu_best_result['iter'] = i
            slu_best_result['dev_slot'], slu_best_result['dev_intent'] = dev_slot, dev_intent
            slu_best_result['test_slot'], slu_best_result['test_intent'] = test_slot, test_intent
            logger.info('SLU New best epoch : %d\tBest Valid (slot : %.4f ; intent : %.4f)\tBest Test (slot : %.4f ; intent : %.4f)' \
                            % (i, dev_slot, dev_intent, test_slot, test_intent))

        ##### Evaluate on dev and test dataset #####
        start_time = time.time()
        dev_bleu, dev_slot = nlg_decode(train_model.nlg_model, nlg_vocab, nlg_evaluator, dev_dataset, os.path.join(exp_path, 'nlg_valid.iter' + str(i)),
            opt.test_batchSize, device=nlg_device, beam=opt.beam, n_best=opt.n_best)
        logger.info('NLG Evaluation epoch : %d\tTime : %.4fs\tValid (bleu : %.4f ; slot : %.4f)' \
                        % (i, time.time() - start_time, dev_bleu, dev_slot))
        start_time = time.time()
        test_bleu, test_slot = nlg_decode(train_model.nlg_model, nlg_vocab, nlg_evaluator, test_dataset, os.path.join(exp_path, 'nlg_test.iter' + str(i)),
            opt.test_batchSize, device=nlg_device, beam=opt.beam, n_best=opt.n_best)
        logger.info('NLG Evaluation epoch : %d\tTime : %.4fs\tTest (bleu : %.4f ; slot : %.4f)' \
                            % (i, time.time() - start_time, test_bleu, test_slot))
        ##### Pick best result on dev and save #####
        prev_best = nlg_coefficient * nlg_best_result['dev_bleu'] + (1 - nlg_coefficient) * nlg_best_result['dev_slot']
        if nlg_coefficient * dev_bleu + (1 - nlg_coefficient) * dev_slot > prev_best:
            torch.save(train_model.nlg_model.state_dict(), open(os.path.join(exp_path, 'nlg_model.pkl'), 'wb'))
            nlg_best_result['iter'] = i
            nlg_best_result['dev_bleu'], nlg_best_result['dev_slot'] = dev_bleu, dev_slot
            nlg_best_result['test_bleu'], nlg_best_result['test_slot'] = test_bleu, test_slot
            logger.info('NLG New best epoch : %d\tBest Valid (bleu : %.4f ; slot : %.4f)\tBest Test (bleu : %.4f ; slot : %.4f)' \
                            % (i, dev_bleu, dev_slot, test_bleu, test_slot))
        gc.collect()
        torch.cuda.empty_cache()

    logger.info('SLU Final best epoch : %d\tBest Valid (slot : %.4f ; intent : %.4f)\tBest Test (slot : %.4f ; intent : %.4f)'
        % (slu_best_result['iter'], slu_best_result['dev_slot'], slu_best_result['dev_intent'], slu_best_result['test_slot'], slu_best_result['test_intent']))
    logger.info('NLG Final best epoch : %d\tBest Valid (bleu : %.4f ; slot : %.4f)\tBest Test (bleu : %.4f ; slot : %.4f)'
        % (nlg_best_result['iter'], nlg_best_result['dev_bleu'], nlg_best_result['dev_slot'], nlg_best_result['test_bleu'], nlg_best_result['test_slot']))
else:
    logger.info("SLU Evaluation starts at %s" % (time.asctime(time.localtime(time.time()))))
    start_time = time.time()
    slot, intent = slu_decode(slu_model, slu_vocab, slu_evaluator, test_dataset, os.path.join(exp_path, 'slu_test.eval'),
        opt.test_batchSize, device=slu_device, beam=opt.beam, n_best=opt.n_best)
    logger.info('SLU Evaluation cost: %.4fs\tSlot : %.4f\tIntent : %.4f' % (time.time() - start_time, slot, intent))
    logger.info("NLG Evaluation starts at %s" % (time.asctime(time.localtime(time.time()))))
    start_time = time.time()
    bleu, slot = nlg_decode(nlg_model, nlg_vocab, nlg_evaluator, test_dataset, os.path.join(exp_path, 'nlg_test.eval'),
        opt.test_batchSize, device=nlg_device, beam=opt.beam, n_best=opt.n_best)
    logger.info('NLG Evaluation cost: %.4fs\tBleu : %.4f\tSlot : %.4f' % (time.time() - start_time, bleu, slot))

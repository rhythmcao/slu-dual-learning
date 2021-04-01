#coding=utf8
import argparse, os, sys, time, json, gc, torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.args import init_args
from utils.initialization import *
from utils.vocab import Vocab
from utils.dataset import read_dataset, split_dataset
from utils.loss import set_celoss_function
from utils.optimization import set_optimizer
from utils.evaluator import Evaluator
from utils.batch import get_minibatch
from utils.constants import PAD, BOS, EOS
from utils.word2vec import load_embeddings
from utils.hyperparam import hyperparam_path
from models.slu.make_model import construct_models as model
from scripts.decode import slu_decode

task = 'slu'

##### Arguments parsing and preparations #####
opt = init_args(params=sys.argv[1:], task=task)
exp_path = hyperparam_path(opt, task=task)
logger = set_logger(exp_path, testing=opt.testing)
set_random_seed(opt.seed)
device = set_torch_device(opt.deviceId)
logger.info("Initialization finished ...")
logger.info("Parameters: " + str(json.dumps(vars(opt))))
logger.info("Output path is: %s" % (exp_path))
logger.info("Random seed is set to: %d" % (opt.seed))
logger.info("Use GPU with index %s" % (opt.deviceId) if opt.deviceId >= 0 else "Use CPU as target torch device")

##### Vocab and Dataset Reader #####
vocab = Vocab(dataset=opt.dataset, task=task)
logger.info("Vocab size for input utterance is: %s" % (len(vocab.word2id)))
logger.info("Vocab size for output slot label is: %s" % (len(vocab.slot2id)))
logger.info("Vocab size for output intent is: %s" % (len(vocab.int2id)))
evaluator = Evaluator.get_evaluator_from_task(task=task, vocab=vocab)

if not opt.testing:
    train_dataset, dev_dataset = read_dataset(opt.dataset, choice='train'), read_dataset(opt.dataset, choice='valid')
    train_dataset, _ = split_dataset(train_dataset, opt.labeled)
    logger.info("Train and dev dataset size is: %s and %s" % (len(train_dataset), len(dev_dataset)))
test_dataset = read_dataset(opt.dataset, choice='test')
logger.info("Test dataset size is: %s" % (len(test_dataset)))

##### Model Construction and Init #####
if not opt.testing:
    opt.vocab_size, opt.slot_num, opt.intent_num = len(vocab.word2id), len(vocab.slot2id), len(vocab.int2id)
    opt.pad_token_idxs = {"word": vocab.word2id[PAD], "slot": vocab.slot2id[PAD]}
    opt.start_idx, opt.end_idx = vocab.slot2id[BOS], vocab.slot2id[EOS]
    params = vars(opt)
    json.dump(params, open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)
else:
    params = json.load(open(os.path.join(exp_path, 'params.json'), 'r'))
train_model = model(**params)
train_model = train_model.to(device)

##### Model Initialization #####
if not opt.testing:
    if not params['use_bert']:
        ratio = load_embeddings(opt.dataset, train_model.word_embed.embed, vocab.word2id, device)
        logger.info("%.2f%% word embeddings from pretrained vectors" % (ratio * 100))
else:
    model_path = os.path.join(opt.read_model_path, 'model.pkl')
    ckpt = torch.load(open(model_path, 'rb'), map_location=device)
    train_model.load_state_dict(ckpt)
    logger.info("Load model from path %s" % (model_path))

##### Training and Decoding #####
if not opt.testing:
    slot_loss_function = set_celoss_function(ignore_index=vocab.slot2id[PAD])
    intent_loss_function = set_celoss_function()
    num_training_steps = ((len(train_dataset) + opt.batchSize - 1) // opt.batchSize) * opt.max_epoch
    num_warmup_steps = int(num_training_steps * opt.warmup_ratio)
    optimizer, scheduler = set_optimizer(train_model, lr=opt.lr, l2=opt.l2, max_norm=opt.max_norm, layerwise_decay=opt.layerwise_decay,
        num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, lr_schedule=opt.lr_schedule)
    logger.info("Training starts at %s" % (time.asctime(time.localtime(time.time()))))
    train_data_index = np.arange(len(train_dataset))
    nsentences, coefficient = len(train_data_index), 0.5
    best_result = {"losses": [], "iter": 0, "dev_slot": 0., "dev_intent": 0.,
        "test_slot": 0., "test_intent": 0.,}
    for i in range(opt.max_epoch):
        start_time = time.time()
        np.random.shuffle(train_data_index)
        losses = []
        train_model.train()
        for j in range(0, nsentences, opt.batchSize):
            optimizer.zero_grad()
            inputs, outputs, intents, lens, _ = get_minibatch(train_dataset, vocab, task=task,
                data_index=train_data_index, index=j, batch_size=opt.batchSize, device=device, use_bert=params['use_bert'])
            slot_scores, intent_scores = train_model(inputs, lens, outputs)
            slot_loss = slot_loss_function(slot_scores, outputs[:, 1:]) if 'crf' not in params['model_type'] else slot_scores
            intent_loss = intent_loss_function(intent_scores, intents)
            batch_loss = coefficient * slot_loss + (1 - coefficient) * intent_loss
            losses.append(batch_loss.item())
            batch_loss.backward()
            train_model.pad_embedding_grad_zero()
            optimizer.step()
            scheduler.step()

        epoch_loss = np.sum(losses, axis=0)
        best_result['losses'].append(epoch_loss)
        logger.info('Training epoch : %d\tTime : %.4fs\tLoss : %.5f' % (i, time.time() - start_time, epoch_loss))
        gc.collect()
        torch.cuda.empty_cache()

        ##### Evaluate on dev and test dataset #####
        if i <= opt.eval_after_epoch:
            continue
        start_time = time.time()
        dev_slot, dev_intent = slu_decode(train_model, vocab, evaluator, dev_dataset, os.path.join(exp_path, 'valid.iter' + str(i)),
            opt.test_batchSize, device=device, beam=opt.beam, n_best=opt.n_best)
        logger.info('Evaluation epoch : %d\tTime : %.4fs\tValid (slot : %.4f ; intent : %.4f)' \
                        % (i, time.time() - start_time, dev_slot, dev_intent))
        start_time = time.time()
        test_slot, test_intent = slu_decode(train_model, vocab, evaluator, test_dataset, os.path.join(exp_path, 'test.iter' + str(i)),
            opt.test_batchSize, device=device, beam=opt.beam, n_best=opt.n_best)
        logger.info('Evaluation epoch : %d\tTime : %.4fs\tTest (slot : %.4f ; intent : %.4f)' \
                            % (i, time.time() - start_time, test_slot, test_intent))

        ##### Pick best result on dev and save #####
        prev_best = coefficient * best_result['dev_slot'] + (1 - coefficient) * best_result['dev_intent']
        if coefficient * dev_slot + (1 - coefficient) * dev_intent > prev_best:
            torch.save(train_model.state_dict(), open(os.path.join(exp_path, 'model.pkl'), 'wb'))
            best_result['iter'] = i
            best_result['dev_slot'], best_result['dev_intent'] = dev_slot, dev_intent
            best_result['test_slot'], best_result['test_intent'] = test_slot, test_intent
            logger.info('New best epoch : %d\tBest Valid (slot : %.4f ; intent : %.4f)\tBest Test (slot : %.4f ; intent : %.4f)' \
                            % (i, dev_slot, dev_intent, test_slot, test_intent))
        gc.collect()
        torch.cuda.empty_cache()

    logger.info('Final best epoch : %d\tBest Valid (slot : %.4f ; intent : %.4f)\tBest Test (slot : %.4f ; intent : %.4f)'
        % (best_result['iter'], best_result['dev_slot'], best_result['dev_intent'], best_result['test_slot'], best_result['test_intent']))
else:
    logger.info("Evaluation starts at %s" % (time.asctime(time.localtime(time.time()))))
    start_time = time.time()
    slot, intent = slu_decode(train_model, vocab, evaluator, test_dataset, os.path.join(exp_path, 'test.eval'),
        opt.test_batchSize, device=device, beam=opt.beam, n_best=opt.n_best)
    logger.info('Evaluation cost: %.4fs\tSlot : %.4f\tIntent : %.4f' % (time.time() - start_time, slot, intent))
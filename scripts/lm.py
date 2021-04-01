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
from models.language_model import LanguageModel as model
from scripts.decode import lm_decode

task = 'lm'

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
logger.info("Vocab size for surface-level and word-level language model is: %s/%s" % (len(vocab.sfm2id), len(vocab.word2id)))
evaluator = Evaluator.get_evaluator_from_task(task=task, vocab=vocab)

if not opt.testing:
    train_dataset, dev_dataset = read_dataset(opt.dataset, choice='train'), read_dataset(opt.dataset, choice='valid')
    logger.info("Train and dev dataset size is: %s and %s" % (len(train_dataset), len(dev_dataset)))
test_dataset = read_dataset(opt.dataset, choice='test')
logger.info("Test dataset size is: %s" % (len(test_dataset)))

##### Model Construction and Init #####
if not opt.testing:
    opt.vocab_size = len(vocab.sfm2id) if opt.surface_level else len(vocab.word2id)
    opt.pad_token_idx = vocab.sfm2id[PAD] if opt.surface_level else vocab.word2id[PAD]
    params = vars(opt)
    json.dump(params, open(os.path.join(exp_path, 'params.json'), 'w'), indent=4)
else:
    params = json.load(open(os.path.join(exp_path, 'params.json'), 'r'))
train_model = model(**params)
train_model = train_model.to(device)

##### Model Initialization #####
if not opt.testing:
    word2id = vocab.sfm2id if params['surface_level'] else vocab.word2id
    ratio = load_embeddings(opt.dataset, train_model.word_embed, word2id, device)
    logger.info("%.2f%% word embeddings from pretrained vectors" % (ratio * 100))
else:
    model_path = os.path.join(opt.read_model_path, 'model.pkl')
    ckpt = torch.load(open(model_path, 'rb'), map_location=device)
    train_model.load_state_dict(ckpt)
    logger.info("Load model from path %s" % (model_path))

##### Training and Decoding #####
if not opt.testing:
    word2id = vocab.sfm2id if params['surface_level'] else vocab.word2id
    loss_function = set_celoss_function(ignore_index=word2id[PAD])
    optimizer, scheduler = set_optimizer(train_model, lr=opt.lr, l2=opt.l2, max_norm=opt.max_norm, lr_schedule='constant')
    logger.info("Training starts at %s" % (time.asctime(time.localtime(time.time()))))
    train_data_index = np.arange(len(train_dataset))
    nsentences = len(train_data_index)
    best_result = {"losses": [], "iter": 0, "dev_ppl": float('inf'), "test_ppl": float('inf'),}
    for i in range(opt.max_epoch):
        start_time = time.time()
        np.random.shuffle(train_data_index)
        losses = []
        train_model.train()
        for j in range(0, nsentences, opt.batchSize):
            optimizer.zero_grad()
            inputs, lens, _ = get_minibatch(train_dataset, vocab, task=task, data_index=train_data_index, index=j,
                batch_size=opt.batchSize, device=device, surface_level=params['surface_level'])
            batch_scores = train_model(inputs, lens)
            batch_loss = loss_function(batch_scores, inputs[:, 1:])
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
        dev_ppl = lm_decode(train_model, vocab, evaluator, dev_dataset, os.path.join(exp_path, 'valid.iter' + str(i)),
            opt.test_batchSize, device=device, surface_level=params['surface_level'])
        logger.info('Evaluation epoch : %d\tTime : %.4fs\tValid (ppl : %.4f)' \
                        % (i, time.time() - start_time, dev_ppl))
        start_time = time.time()
        test_ppl = lm_decode(train_model, vocab, evaluator, test_dataset, os.path.join(exp_path, 'test.iter' + str(i)),
            opt.test_batchSize, device=device, surface_level=params['surface_level'])
        logger.info('Evaluation epoch : %d\tTime : %.4fs\tTest (ppl : %.4f)' \
                            % (i, time.time() - start_time, test_ppl))

        ##### Pick best result on dev and save #####
        if dev_ppl < best_result['dev_ppl']:
            torch.save(train_model.state_dict(), open(os.path.join(exp_path, 'model.pkl'), 'wb'))
            best_result['iter'] = i
            best_result['dev_ppl'], best_result['test_ppl'] = dev_ppl, test_ppl
            logger.info('New best epoch : %d\tBest Valid (ppl : %.4f)\tBest Test (ppl : %.4f)' % (i, dev_ppl, test_ppl))
        gc.collect()
        torch.cuda.empty_cache()

    logger.info('Final best epoch : %d\tBest Valid (ppl : %.4f)\tBest Test (ppl : %.4f)'
        % (best_result['iter'], best_result['dev_ppl'], best_result['test_ppl']))
else:
    logger.info("Evaluation starts at %s" % (time.asctime(time.localtime(time.time()))))
    start_time = time.time()
    ppl = lm_decode(train_model, vocab, evaluator, test_dataset, os.path.join(exp_path, 'test.eval'),
        opt.test_batchSize, device=device, surface_level=params['surface_level'])
    logger.info('Evaluation cost: %.4fs\tPpl : %.4f' % (time.time() - start_time, ppl))
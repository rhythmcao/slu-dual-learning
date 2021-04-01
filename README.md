# Dual Learning for Semi-Supervised Natural Language Understanding

----

This is the project containing source code and data for the journal [*Dual learning for semi-supervised natural language understanding*](https://arxiv.org/abs/2004.12299) in **IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP) 2020**. If you find it useful, please cite our work.

    @article{Zhu_2020,
        title={Dual Learning for Semi-Supervised Natural Language Understanding},
        ISSN={2329-9304},
        url={http://dx.doi.org/10.1109/TASLP.2020.3001684},
        DOI={10.1109/taslp.2020.3001684},
        journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
        publisher={Institute of Electrical and Electronics Engineers (IEEE)},
        author={Zhu, Su and Cao, Ruisheng and Yu, Kai},
        year={2020},
        pages={1–1}
    }

----

## Preparations

1. Create the conda environment `slu` and download dependencies such as char/word vectors and pretrained language model `bert-base-uncased`:

        ./environment.sh

2. Construct the vocabulary, slot-value database and intent-slot co-occurrence matrix:

        python utils/preparations.py --dataset atis snips

All outputs are saved in directory `data`.

----

## Supervised experiments

All the experimental outputs will be saved in the directory `exp` by default, see `utils/hyperparam.py`.

### SLU task

Running script: (`labeled` is the ratio of labeled examples in the entire training set)

    ./run/run_slu.sh [atis|snips] labeled [birnn|birnn+crf|focus]

Or with bert:

    ./run/run_slu_bert.sh [atis|snips] labeled [birnn|birnn+crf|focus]

### NLG task

Running script:

    ./run/run_nlg.sh [atis|snips] labeled [sclstm|sclstm+copy]

### Language Model task

Running script:

    ./run/run_lm.sh [atis|snips] [surface|sentence]

`surface` means training a LM with slot values replaced by its slot name; while `sentence` argument represents the LM trained at the natural language level.

----

## Semi-supervised experiments

Attention: all model paths such as `read_slu_model_path` in the running scripts below can be replaced with other supervised models.

### Dual pseudo labeling

Running script:

    ./run/run_dual_pseudo_labeling.sh [atis|snips] labeled [focus|bert]

### Dual learning

Running script:

    ./run/run_dual_learning.sh [atis|snips] labeled [focus|bert]

### Dual pseudo labeling + Dual learning

Running script:

    ./run/run_dual_plus_pseudo.sh [atis|snips] labeled [focus|bert]


import itertools
import json
from argparse import Namespace

default_conf = Namespace(
    primary_learner_hidden=[64, 32],
    adversary_hidden=[],  # linear adversary
    adversary_input=['X', 'Y'],
    train_steps=5000,
    opt='Adagrad',
    batch_size=256,
    primary_lr=0.1,
    sensitive_label=False,

    # ARL
    pretrain_steps=250,  # pretrain the learner before adversary

    # DRO
    eta=0.5,
    k=2.0,
)

models = ['Baseline', 'DRO', 'ARL', 'IPW(S)', 'IPW(S+Y)']
datasets = ['uci_adult', 'compas', 'law_school']
experiments = list(itertools.product(datasets, models))

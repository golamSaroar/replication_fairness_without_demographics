import itertools
import json
from argparse import Namespace

from utils import get_results

default_conf = Namespace(
    primary_learner_hidden=[64, 32],
    adversary_hidden=[],  # linear adversary
    adversary_input=['X', 'Y'],
    train_steps=5000,
    optimizer='Adagrad',
    batch_size=256,
    primary_lr=0.1,
    sensitive_label=False,

    eval_batch_size=512,
    num_workers=4,  # for the Dataloader
    progress_bar=True,

    # ARL
    pretrain_steps=250,  # pretrain the learner before adversary

    # DRO
    eta=0.5,
    k=2.0,
)

models = ['Baseline', 'DRO', 'ARL', 'IPW(S)', 'IPW(S+Y)']
datasets = ['uci_adult', 'compas', 'law_school']
experiments = list(itertools.product(datasets, models))

with open('hyper_parameters.json') as f:
    optimal_hyper_parameters = json.load(f)

results = []

conf = Namespace(**vars(default_conf))
conf.seed_run = True

for seed in range(1, 11):
    result_dict = get_results(seed, conf, optimal_hyper_parameters, experiments)
    results.append(result_dict)

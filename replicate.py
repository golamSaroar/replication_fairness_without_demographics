import itertools
import json

from utils import *

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
    num_workers=8,  # for the Dataloader
    progress_bar=True,
    no_early_stopping=False,
    seed_run_version=0,
    version=None,
    seed=0,
    seed_run=False,
    log_dir='training_logs',

    # ARL
    pretrain_steps=250,  # pretrain the learner before adversary

    # DRO
    eta=0.5,
    k=2.0,
)

models = ['ARL', 'baseline', 'DRO']
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

results_dict = convert_result_to_dict(results, experiments, performance_metrics_list)

with open("results.txt", "w") as results_file:
    results_file.write(str(results_dict))
    results_file.close()

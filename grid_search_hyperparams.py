import torch
import json
from argparse import Namespace
import itertools

import trainer

optimizer_dict = {
    'Adagrad': torch.optim.Adagrad,
    'Adam': torch.optim.Adam
}

models = ['ARL', 'baseline', 'DRO', 'IPW(S)', 'IPW(S+Y)']
datasets = ['uci_adult', 'compas', 'law_school']
parameters = ['lr', 'batch_size']


def run_grid_search(args):
    optimal_hyperparams = {dataset: {} for dataset in datasets}

    for model, dataset in itertools.product(models, datasets):
        print(f'Now running grid search for {model} on {dataset}')

        if model == 'IPW(S)':
            args.model = 'IPW'
            args.sensitive_label = False
        elif model == 'IPW(S+Y)':
            args.model = 'IPW'
            args.sensitive_label = True
        else:
            args.model = model
            args.sensitive_label = False
        
        args.dataset = dataset

        # run grid_search
        auc_scores, best_params = trainer.train_and_evaluate(args)

        print(f'Best params for {model} on {dataset} are {best_params}')

        # add params to dict
        optimal_hyperparams[dataset][model] = {key: best_params[key] for key in parameters}
        if model == 'DRO':
            optimal_hyperparams[dataset][model]['eta'] = best_params['eta']
        
        # write to disk, to ensure it is saved even if run is aborted later
        path = './hyper_parameters.json'
        with open(path, 'w') as f:
            json.dump(optimal_hyperparams, f)

    return optimal_hyperparams


args = Namespace(
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

    # grid_search
    grid_search=True,
    num_cpus=8,
    num_folds=5
)

run_grid_search(args)

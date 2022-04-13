from argparse import Namespace
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.metrics.functional.classification import auroc
import torch
from torch.utils.data import DataLoader
import os
import json
import numpy as np
from ray import tune
from sklearn.model_selection import KFold
from time import time

from datasets import FullDataset, CustomSubset
from models.arl import ARL
from models.baseline import Baseline
from models.dro import DRO
from models.ipw import IPW
from results import Logger, get_all_results

optimizer_dict = {
    'Adagrad': torch.optim.Adagrad,
    'Adam': torch.optim.Adam
}


def get_model(config, args, dataset):
    if args.model == 'ARL':
        model = ARL(config=config,
                    input_shape=dataset.dimensionality,
                    pretrain_steps=args.pretrain_steps,
                    prim_hidden=args.primary_learner_hidden,
                    adv_hidden=args.adversary_hidden,
                    optimizer=optimizer_dict[args.optimizer],
                    adv_input=set(args.adversary_input),
                    num_groups=len(dataset.sensitive_index_to_values),
                    opt_kwargs={})

    elif args.model == 'baseline':
        model = Baseline(config=config,
                         num_features=dataset.dimensionality,
                         hidden_units=args.primary_learner_hidden,
                         optimizer=optimizer_dict[args.optimizer],
                         opt_kwargs={})
        args.pretrain_steps = 0

    elif args.model == 'DRO':
        model = DRO(config=config,
                    num_features=dataset.dimensionality,
                    hidden_units=args.primary_learner_hidden,
                    pretrain_steps=args.pretrain_steps,
                    k=args.k,
                    optimizer=optimizer_dict[args.optimizer],
                    opt_kwargs={})
    elif args.model == 'IPW':
        model = IPW(config=config,
                    num_features=dataset.dimensionality,
                    hidden_units=args.primary_learner_hidden,
                    optimizer=optimizer_dict[args.optimizer],
                    group_probs=dataset.group_probs,
                    sensitive_label=args.sensitive_label,
                    opt_kwargs={})
        args.pretrain_steps = 0

    return model


def train(config,
          args,
          train_dataset,
          val_dataset=None,
          test_dataset=None,
          version=str(int(time())),
          fold_nbr=None):
    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)

    callbacks = [Logger(train_dataset, 'train', batch_size=args.eval_batch_size,
                        save_scatter=(args.model in ['ARL']))]

    if val_dataset is not None:
        callbacks.append(Logger(val_dataset, 'validation', batch_size=args.eval_batch_size))
        if not args.no_early_stopping:
            callbacks.append(EarlyStopping(
                monitor='validation/micro_avg_auc',
                min_delta=0.00,
                patience=10,
                verbose=True,
                mode='max'
            ))

    if test_dataset is not None:
        callbacks.append(
            Logger(test_dataset, 'test', batch_size=args.eval_batch_size, save_scatter=(args.model in ['ARL'])))

    model = get_model(config, args, train_dataset)

    logdir = args.log_dir
    os.makedirs(logdir, exist_ok=True)

    if not args.no_early_stopping:
        # create checkpoint
        checkpoint = ModelCheckpoint(save_weights_only=True,
                                     dirpath=logdir,
                                     mode='max',
                                     verbose=False,
                                     monitor='validation/micro_avg_auc')
        callbacks.append(checkpoint)

    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0,
                         max_steps=args.train_steps + args.pretrain_steps,
                         callbacks=callbacks,
                         gradient_clip_val=1 if args.model == 'DRO' else 0,
                         progress_bar_refresh_rate=1 if args.progress_bar else 0,
                         )

    if val_dataset is not None:
        trainer.fit(model, train_loader, val_dataloaders=DataLoader(val_dataset,
                                                                    batch_size=args.eval_batch_size,
                                                                    num_workers=args.num_workers))
    else:
        trainer.fit(model, train_loader)

    if not args.no_early_stopping:
        assert trainer.checkpoint_callback is not None
        if args.model == 'ARL':
            model = ARL.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        elif args.model == 'baseline':
            model = Baseline.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        elif args.model == 'DRO':
            model = DRO.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        elif args.model == 'IPW':
            model = IPW.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model, trainer


def train_and_evaluate(conf):
    conf = Namespace(**vars(conf))

    pl.seed_everything(conf.seed)
    np.random.seed(conf.seed)

    dataset = FullDataset(conf.dataset, sensitive_label=conf.sensitive_label)
    test_dataset = FullDataset(conf.dataset, sensitive_label=conf.sensitive_label, test=True)

    if conf.version is None:
        conf.version = str(int(time()))

    if conf.grid_search:
        lr_list = [0.001, 0.01, 0.1, 1, 2, 5]
        batch_size_list = [32, 64, 128, 256, 512]
        eta_list = [0.0]  # not required for other models except DRO

        if conf.model == 'DRO':
            eta_list = [0.5, 0.6, 0.7, 0.8, 0.9]

        config = {
            'lr': tune.grid_search(lr_list),
            'batch_size': tune.grid_search(batch_size_list),
            'eta': tune.grid_search(eta_list)
        }

        # perform n-fold cross-validation
        kfold = KFold(n_splits=conf.num_folds)
        fold_indices = list(kfold.split(dataset))

        if conf.model == 'IPW':
            if conf.sensitive_label:
                path = f'grid_search/IPW(S+Y)_{conf.dataset}_version_{conf.version}'
            else:
                path = f'grid_search/IPW(S)_{conf.dataset}_version_{conf.version}'
        else:
            path = f'grid_search/{conf.model}_{conf.dataset}_version_{conf.version}'

        grid_search = tune.run(
            tune.with_parameters(
                get_auc_per_fold,
                args=conf,
                dataset=dataset,
                fold_indices=fold_indices),
            resources_per_trial={
                'cpu': conf.num_cpus
            },
            config=config,
            metric='auc',
            mode='max',
            local_dir=os.getcwd(),
            name=path
        )

        print('Optimal hyperparameters are: ', grid_search.best_config)

        # save grid search results
        df = grid_search.results_df
        df.to_csv(os.path.join(path, 'results.csv'), index=False)

        # set hyperparameters for final run
        config['lr'] = grid_search.best_config['lr']
        config['batch_size'] = grid_search.best_config['batch_size']
        config['eta'] = grid_search.best_config['eta']

    else:
        config = {
            "lr": conf.primary_lr,
            "batch_size": conf.batch_size,
            "eta": conf.eta
        }

        path = get_path(conf)
        os.makedirs(path, exist_ok=True)

    conf.log_dir = path

    # create training and validation set
    permuted_indices = np.random.permutation(np.arange(0, len(dataset)))
    train_indices, val_indices = permuted_indices[:int(0.9 * len(permuted_indices))], permuted_indices[
                                                                                      int(0.9 * len(permuted_indices)):]
    train_dataset, val_dataset = CustomSubset(dataset, train_indices), CustomSubset(dataset, val_indices)

    model, _ = train(config, conf, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)

    dataloader = DataLoader(test_dataset, batch_size=conf.eval_batch_size)
    auc_scores = get_all_results(model, dataloader, test_dataset.minority_group)

    with open(os.path.join(path, 'auc_scores.json'), 'w') as f:
        json.dump(auc_scores, f)

    print(f'results ({conf.dataset}, {conf.model}) = {auc_scores}')

    if conf.grid_search:
        return auc_scores, grid_search.best_config

    return auc_scores


def get_path(conf):
    model = conf.model

    if model == 'IPW':
        if conf.sensitive_label:
            model = 'IPW(S+Y)'
        else:
            model = 'IPW(S)'

    if conf.seed_run:
        path = f'./{conf.log_dir}/{conf.dataset}/{model}/seed_run_version_{conf.seed_run_version}/seed_{conf.seed}'
    else:
        path = f'./{conf.log_dir}/{conf.dataset}/{model}/version_{conf.version}'

    return path


performance_metrics_list = ['micro_avg_auc', 'macro_avg_auc', 'min_auc', 'minority_auc', 'accuracy']


def get_results(seed, conf, optimal_hyper_parameters, experiments):
    result_per_seed = {}

    for dataset, model in experiments:
        current_conf = Namespace(**vars(conf))
        if model == 'IPW(S)':
            current_conf.model = 'IPW'
            current_conf.sensitive_label = False
        elif model == 'IPW(S+Y)':
            current_conf.model = 'IPW'
            current_conf.sensitive_label = True
        else:
            current_conf.model = model
        current_conf.dataset = dataset
        current_conf.seed = seed

        if optimal_hyper_parameters is not None:
            for k, v in optimal_hyper_parameters[dataset][model].items():
                setattr(current_conf, k, v)

        result_per_seed[(dataset, model)] = train_and_evaluate(current_conf)

    return result_per_seed


def convert_result_to_dict(results, experiments, performance_metrics_list):
    return {
        k: {
            metric: {
                'mean': np.mean([result_dict[k][metric] for result_dict in results]),
                'std': np.std([result_dict[k][metric] for result_dict in results])
            } for metric in performance_metrics_list
        } for k in experiments
    }


def get_auc_per_fold(config,
                     args,
                     dataset,
                     fold_indices):
    print(f'Starting run with seed {args.seed} - lr {config["lr"]} - batch_size {config["batch_size"]}')

    fold_nbr = 0
    aucs = []
    for train_indices, val_indices in fold_indices:
        fold_nbr += 1

        # create datasets for fold
        train_dataset = CustomSubset(dataset, train_indices)
        val_dataset = CustomSubset(dataset, val_indices)

        # train model
        model, _ = train(config, args, train_dataset=train_dataset, val_dataset=val_dataset, version=args.version,
                         fold_nbr=fold_nbr)

        # Evaluate on validation set to get an estimate of performance
        scores = torch.sigmoid(model(val_dataset.features))
        aucs.append(auroc(scores, val_dataset.labels).item())

    mean_auc = np.mean(aucs)
    print(f'Finished run with seed {args.seed} - lr {config["lr"]} - bs {config["batch_size"]} - mean auc: {mean_auc}')
    tune.report(auc=mean_auc)

    return mean_auc

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
import os
import json
import numpy as np

from datasets import FullDataset, CustomSubset
from arl import ARL
from baseline import Baseline
from utils import *
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

    return model


def train(config,
          args,
          train_dataset,
          val_dataset=None,
          test_dataset=None):
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

    return model, trainer


def train_and_evaluate(conf):
    conf = Namespace(**vars(conf))

    pl.seed_everything(conf.seed)
    np.random.seed(conf.seed)

    dataset = FullDataset(conf.dataset, sensitive_label=conf.sensitive_label)
    test_dataset = FullDataset(conf.dataset, sensitive_label=conf.sensitive_label, test=True)

    config = {
        "lr": conf.primary_lr,
        "batch_size": conf.batch_size,
        "eta": conf.eta
    }

    if conf.seed_run:
        path = f'./{conf.log_dir}/{conf.dataset}/{conf.model}/seed_run_version_{conf.seed_run_version}/seed_{conf.seed}'
    else:
        path = f'./{conf.log_dir}/{conf.dataset}/{conf.model}/version_{conf.version}'

    print(f'creating dir {path}')
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

    print(f'results = {auc_scores}')

    # save results
    with open(os.path.join(path, 'auc_scores.json'), 'w') as f:
        json.dump(auc_scores, f)

    return auc_scores

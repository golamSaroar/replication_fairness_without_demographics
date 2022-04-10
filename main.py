from argparse import Namespace
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping
import torch

from datasets import Dataset, CustomSubset
from arl import ARL

optimizer_dict = {
    'Adagrad': torch.optim.Adagrad,
    'Adam': torch.optim.Adam
}


def get_model(config, args, dataset):
    model = ARL(config=config,
                input_shape=dataset.dimensionality,
                pretrain_steps=args.pretrain_steps,
                prim_hidden=args.primary_learner_hidden,
                adv_hidden=args.adversary_hidden,
                optimizer=optimizer_dict[args.optimizer],
                adv_input=set(args.adversary_input),
                num_groups=len(dataset.sensitive_index_to_values),
                opt_kwargs={"initial_accumulator_value": 0.1} if args.tf_mode else {})

    return model


def train(config,
          args,
          train_dataset,
          val_dataset=None,
          test_dataset=None):
    # create fold loaders and callbacks
    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)

    callbacks = []

    if val_dataset is not None:
        callbacks.append(EarlyStopping(
            monitor='validation/micro_avg_auc',
            min_delta=0.00,
            patience=10,
            verbose=True,
            mode='max'
        ))

    model = get_model(config, args, train_dataset)

    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0,
                         max_steps=args.train_steps + args.pretrain_steps,
                         callbacks=callbacks,
                         gradient_clip_val=1 if args.model == 'DRO' else 0,
                         progress_bar_refresh_rate=1 if args.progress_bar else 0,
                         )

    return "model", "trainer"


def train_and_evaluate(conf):
    conf = Namespace(**vars(conf))

    pl.seed_everything(conf.seed)
    np.random.seed(conf.seed)

    dataset = Dataset(conf.dataset, sensitive_label=conf.sensitive_label)
    test_dataset = Dataset(conf.dataset, sensitive_label=conf.sensitive_label, test=True)

    config = {
        "lr": conf.primary_lr,
        "batch_size": conf.batch_size,
        "eta": conf.eta
    }

    # create training and validation set
    permuted_indices = np.random.permutation(np.arange(0, len(dataset)))
    train_indices, val_indices = permuted_indices[:int(0.9 * len(permuted_indices))], permuted_indices[
                                                                                      int(0.9 * len(permuted_indices)):]
    train_dataset, val_dataset = CustomSubset(dataset, train_indices), CustomSubset(dataset, val_indices)

    model, _ = train(config, conf, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)

    auc_scores = {}

    return auc_scores

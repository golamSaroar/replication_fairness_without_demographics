from argparse import Namespace
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import os
import numpy as np
from time import time
import json
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from datasets import FullDataset, CustomSubset
from models.arl import ARL
from results import Logger, get_all_results

optimizer_dict = {
    'Adagrad': torch.optim.Adagrad,
    'Adam': torch.optim.Adam
}


def get_arl_weights(args):
    args.version = str(int(time()))

    pl.seed_everything(args.seed)
    np.random.seed(args.seed)

    dataset = FullDataset(args.dataset)
    test_dataset = FullDataset(args.dataset, test=True)

    config = {'lr': args.primary_lr, 'batch_size': args.batch_size}

    path = f'./{args.log_dir}/{args.dataset}/ARL_example_weights/version_{args.version}'
    
    print(f'creating dir {path}')
    os.makedirs(path, exist_ok=True)

    args.log_dir = path

    permuted_idcs = np.random.permutation(np.arange(0, len(dataset)))
    train_idcs, val_idcs = permuted_idcs[:int(0.9*len(permuted_idcs))], permuted_idcs[int(0.9*len(permuted_idcs)):] 
    train_dataset, val_dataset = CustomSubset(dataset, train_idcs), CustomSubset(dataset, val_idcs)
    
    # single training run
    model = train(config, args, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)
    
    # compute final test scores
    dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    auc_scores= get_all_results(model, dataloader, test_dataset.minority_group)

    print(f'AUC Results = {auc_scores}')

    print(f'Evaluating adversary scores on test set')
    lambdas, predictions, true_labels, memberships = model.get_lambda(dataloader)

    lambdas = lambdas.detach().numpy().reshape(-1,1)
    predictions = predictions.detach().numpy().reshape(-1,1)
    true_labels = true_labels.detach().numpy().reshape(-1,1)
    memberships = memberships.detach().numpy().reshape(-1,1)

    # sort in ascending order of lambda
    sort_idcs = np.argsort(lambdas, axis=0)
    lambdas = lambdas[sort_idcs]
    predictions = predictions[sort_idcs]
    true_labels = true_labels[sort_idcs]
    memberships = memberships[sort_idcs]

    print(f'Saving results to {path}')
    np.save(os.path.join(path, 'lambdas'), lambdas)
    np.save(os.path.join(path, 'predictions'), predictions)
    np.save(os.path.join(path, 'true_labels'), true_labels)
    np.save(os.path.join(path, 'memberships'), memberships)

    print('Plotting adversary scores..')
    # init KD
    kde = KernelDensity(bandwidth=0.3)

    idx2val = test_dataset.sensitive_index_to_values
    race_dict = {'Black':'Black', 'Other':'White'}
    sex_dict = {'Female':'Female', 'Other':'Male'}

    score_ticks = np.linspace(0,5,100).reshape((100,1))

    # plot 0 - 0
    plt.figure()
    for i in idx2val:
        combi = idx2val[i]
        lam = lambdas[(predictions == 0) & (true_labels == 0) & (memberships == i)][:, np.newaxis]
        kde_00 = kde.fit(lam)
        density_00 = np.exp(kde_00.score_samples(score_ticks))
        plt.plot(score_ticks[:,0], density_00, label=f'{race_dict[combi[0]]} {sex_dict[combi[1]]}')
    plt.legend()
    plt.savefig(os.path.join(path, 'lambda_0_0.pdf'))

    # plot 0 - 1
    plt.figure()
    for i in idx2val:
        combi = idx2val[i]
        lam = lambdas[(predictions == 1) & (true_labels == 0) & (memberships == i)][:, np.newaxis]
        kde_01 = kde.fit(lam)
        density_01 = np.exp(kde_01.score_samples(score_ticks))
        plt.plot(score_ticks, density_01, label=f'{race_dict[combi[0]]} {sex_dict[combi[1]]}')
    plt.legend()
    plt.savefig(os.path.join(path, 'lambda_0_1.pdf'))

    # plot 1 - 0
    plt.figure()
    for i in idx2val:
        combi = idx2val[i]
        lam = lambdas[(predictions == 0) & (true_labels == 1) & (memberships == i)][:, np.newaxis]
        kde_10 = kde.fit(lam)
        density_10 = np.exp(kde_10.score_samples(score_ticks))
        plt.plot(score_ticks, density_10, label=f'{race_dict[combi[0]]} {sex_dict[combi[1]]}')
    plt.legend()
    plt.savefig(os.path.join(path, 'lambda_1_0.pdf'))

    # plot 1 - 1
    plt.figure()
    for i in idx2val:
        combi = idx2val[i]
        lam = lambdas[(predictions == 1) & (true_labels == 1) & (memberships == i)][:, np.newaxis]
        kde_11 = kde.fit(lam)
        density_11 = np.exp(kde_11.score_samples(score_ticks))
        plt.plot(score_ticks, density_11, label=f'{race_dict[combi[0]]} {sex_dict[combi[1]]}')
    plt.legend()
    plt.savefig(os.path.join(path, 'lambda_1_1.pdf'))

    # combined plot
    f, axes = plt.subplots(2,2,gridspec_kw={'hspace':.5})
    axes = axes.flatten()
    
    plt_settings = [(0,0), (0,1), (1,0), (1,1)]
    titles = ['no-error; class 0', 'error; class 0', 'error; class 1', 'no-error; class 1']
    for i in range(len(plt_settings)):
        for idx in idx2val:
            combi = idx2val[idx]
            lam = lambdas[(predictions == plt_settings[i][1]) & (true_labels == plt_settings[i][0]) & (memberships == idx)][:, np.newaxis]
            kde_combined = kde.fit(lam)
            density_combined = np.exp(kde_combined.score_samples(score_ticks))
            axes[i].plot(score_ticks, density_combined, label=f'{race_dict[combi[0]]} {sex_dict[combi[1]]}')
            if i == 0 or i == 2:
                axes[i].set_ylabel('density')
            if i == 2 or i == 3:
                axes[i].set_xlabel(r'test weight $\lambda$')
            axes[i].set_title(titles[i])
            if i == 0:
                axes[i].legend()
        f.savefig('ARL_learnt_weights.pdf')

    # save results
    with open(os.path.join(path, 'auc_scores.json'),'w') as f:
        json.dump(auc_scores, f)
    

def get_model(config, args, dataset):
    return ARL(config=config, # for hparam tuning
                input_shape=dataset.dimensionality,
                pretrain_steps=args.pretrain_steps,
                prim_hidden=args.primary_learner_hidden,
                adv_hidden=args.adversary_hidden,
                optimizer=optimizer_dict[args.optimizer],
                adv_input=set(args.adversary_input),
                num_groups=len(dataset.sensitive_index_to_values),
                opt_kwargs={})


def train(config,
          args,
          train_dataset,
          val_dataset=None,
          test_dataset=None):
    logdir = args.log_dir
    os.makedirs(logdir, exist_ok=True)

    # create fold loaders and callbacks
    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)

    callbacks = []

    if val_dataset is not None:
        callbacks.append(Logger(val_dataset, 'validation', batch_size=args.eval_batch_size))
        callbacks.append(EarlyStopping(
            monitor='validation/micro_avg_auc', 
            min_delta=0.00,
            patience=10,
            verbose=True,
            mode='max'
        ))
    
    if test_dataset is not None:
        callbacks.append(Logger(test_dataset, 'test', batch_size=args.eval_batch_size))
        
    # Select model and instantiate
    model = get_model(config, args, train_dataset)

    # create checkpoint
    checkpoint = ModelCheckpoint(save_weights_only=True,
                                dirpath=logdir,
                                mode='max', 
                                verbose=False,
                                monitor='validation/micro_avg_auc')
    callbacks.append(checkpoint)
    
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0,
                         max_steps=args.train_steps + args.pretrain_steps,
                         callbacks=callbacks,
                         gradient_clip_val=0,
                         progress_bar_refresh_rate=1 if args.progress_bar else 0)
    
    # Training
    fit_time = time()
    if val_dataset is not None:
        trainer.fit(model, train_loader, val_dataloaders=DataLoader(val_dataset, batch_size=args.eval_batch_size))
    else:
        trainer.fit(model, train_loader)
    print(f'time to fit was {time()-fit_time}')

    assert trainer.checkpoint_callback is not None
        
    model = ARL.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    return model


args = Namespace(
    dataset='uci_adult',
    model='ARL',
    primary_learner_hidden=[64, 32],
    adversary_hidden=[],  # linear adversary
    adversary_input=['X', 'Y'],
    optimizer='Adagrad',
    primary_lr=0.01,
    train_steps=5000,
    batch_size=512,
    eval_batch_size=512,
    num_workers=8,  # for the Dataloader
    progress_bar=True,
    seed=0,
    log_dir='training_logs',
    pretrain_steps=250  # pretrain the learner before adversary
)

get_arl_weights(args)

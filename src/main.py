import os
import logging
import hydra
import h5py
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob
from omegaconf import DictConfig

from data.datagenerator import *
from data.features import *
from utils import EpisodicBatchSampler
from protonet import Protonet


@hydra.main(config_path='../config', config_name='config')
def main(conf: DictConfig):
    logger = logging.getLogger(__name__)

    if not os.path.isdir(conf.path.feat_path):
        os.makedirs(conf.path.feat_path)

    if not os.path.isdir(conf.path.train_feat):
        os.makedirs(conf.path.train_feat)

    if not os.path.isdir(conf.path.eval_feat):
        os.makedirs(conf.path.eval_feat)

    if conf.set.features:
        logger.info("### Feature Extraction ###")
        Num_extract_train, data_shape = featureExtract(conf=conf,mode="train")
        logger.info("Shape of dataset is {}".format(data_shape))
        logger.info("Total training samples is {}".format(Num_extract_train))

        Num_extract_eval = featureExtract(conf=conf,mode='eval')
        logger.info("Total number of samples used for evaluation: {}".format(Num_extract_eval))
        logger.info("### Feature Extraction Complete ###")

    if conf.set.train:
        if not os.path.isdir(conf.path.model):
            os.makedirs(conf.path.model)
        
    train_gen = Datagen(conf)
    X_train, Y_train, X_val, Y_val = train_gen.generate_train()
    X_train = torch.tensor(X_train)
    Y_train = torch.LongTensor(Y_train)
    X_val = torch.tensor(X_val)
    Y_val = torch.LongTensor(Y_val)

    samples_per_class = conf.train.n_shot * 2

    batch_size = samples_per_class * conf.train.k_way
    num_batches_train = len(Y_train) // batch_size
    num_batches_val = len(Y_val) // batch_size

    train_sampler = EpisodicBatchSampler(Y_train, num_batches_train, conf.train.k_way, samples_per_class)
    val_sampler = EpisodicBatchSampler(Y_val, num_batches_val, conf.train.k_way, samples_per_class)

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_sampler=train_sampler,
                                                num_workers=0,
                                                pin_memory=True,
                                                shuffle=False)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_sampler=val_sampler,
                                                num_workers=0,
                                                pin_memory=True,
                                                shuffle=False)

    protonet = Protonet(conf)
    tb_logger = TensorBoardLogger("logs", name="dcase_protonet")
    
    trainer = pl.Trainer(gpus=conf.train.gpus,
                         accelerator=conf.train.accelerator,
                         max_epochs=conf.train.epochs,
                         logger=tb_logger,
                         fast_dev_run=False)
    trainer.fit(protonet, 
                train_dataloader=train_loader, 
                val_dataloaders=val_loader)

    logger.info("Training Complete")

    if conf.set.eval:
        pass

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(modules)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    main()

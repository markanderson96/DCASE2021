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

from datagenerator import *
from features import *
from utils import EpisodicBatchSampler, euclidean_dist
from protonet import Protonet
from evaluation.prediction import *


@hydra.main(config_path='../config', config_name='config')
def main(conf: DictConfig):
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
        
        trainer = pl.Trainer(gpus=conf.set.gpus,
                            max_epochs=conf.train.epochs,
                            logger=tb_logger,
                            fast_dev_run=False)
        trainer.fit(protonet, 
                    train_dataloader=train_loader, 
                    val_dataloaders=val_loader)

        logger.info("Training Complete")

    if conf.set.eval:
        name_array = np.array([])
        onset_array = np.array([])
        offset_array = np.array([])

        feature_files = [file for file in glob(os.path.join(conf.path.eval_feat, '*.h5'))]
        for file in feature_files:
            feature_name = file.split('/')[-1]
            audio_name = feature_name.replace('h5', 'wav')

            logger.info('Processing file: {}'.format(audio_name))

            eval_feat = h5py.File(file, 'r')
            query_index_start = eval_feat['query_index_start'][:][0]
            onset, offset = eval_prototypes(conf,
                                            eval_feat, 
                                            query_index_start)

            name = np.repeat(audio_name, len(onset))
            name_array = np.append(name_array, name)
            onset_array = np.append(onset_array, onset)
            offset_array = np.append(offset_array, offset)

        output = { 'Audiofilename' : name_array,
                   'Starttime' : onset_array,
                   'Endtime' : offset_array
        }
        df_outputs = pd.DataFrame(output)
        csv_path = os.path.join(conf.path.root_dir,'eval_out.csv')
        df_outputs.to_csv(csv_path, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(modules)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    tb_logger = TensorBoardLogger("logs", name="dcase_protonet")
    main()

import yaml
import csv
import os
import hydra
import h5py
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob
from omegaconf import DictConfig

from data.datagenerator import *
from data.features import *
from training import train
from models.protonet import Protonet


@hydra.main(config_path='../config', config_name='config')
def main(conf: DictConfig):
    if not os.path.isdir(conf.path.feat_path):
        os.makedirs(conf.path.feat_path)

    if not os.path.isdir(conf.path.train_feat):
        os.makedirs(conf.path.train_feat)

    if not os.path.isdir(conf.path.eval_feat):
        os.makedirs(conf.path.feat_eval)

    if conf.set.features:
        print("### Feature Extraction ###")
        Num_extract_train, data_shape = featureExtract(conf=conf,mode="train")
        print("Shape of dataset is {}".format(data_shape))
        print("Total training samples is {}".format(Num_extract_train))

        Num_extract_eval = featureExtract(conf=conf,mode='eval')
        print("Total number of samples used for evaluation: {}".format(Num_extract_eval))
        print("### Feature Extraction Complete ###")

    if conf.set.train:
        if not os.path.isdir(conf.path.model):
            os.makedirs(conf.path.model)
        
        train_gen = TrainGenerator(conf)
        X_train, Y_train, X_val, Y_val = train_gen.generate_train_data()
        X_train = torch.tensor(X_train)
        Y_train = torch.LongTensor(Y_train)
        X_val = torch.tensor(X_val)
        Y_val = torch.LongTensor(Y_val)

        samples_per_class = conf.train.n_shot * 2

        batch_size = samples_per_class * conf.train.k_way
        num_batches_train = len(Y_train) // batch_size
        num_batches_val = len(Y_val) // batch_size

        train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   pin_memory=True,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 pin_memory=True,
                                                 shuffle=True)

        model = Protonet()
        best_acc, best_state = train(model, train_loader, val_loader, num_batches_train, num_batches_val, conf)
        print("Training Complete")

    if conf.set.eval:
        pass

if __name__ == '__main__':
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # else:
    #     device = "cpu"
    main()

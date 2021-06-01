import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import hydra
from omegaconf import DictConfig

from data.datagenerator import *
from data.features import *
from utils import EpisodicBatchSampler

class Protonet(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.encoder = nn.Sequential(
            self.conv_block(1, 128),
            self.conv_block(128, 128),
            self.conv_block(128,128),
            self.conv_block(128, 128)
        )
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 2, padding_mode='zeros'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        (num_samples, seq_len, fft_bins) = x.shape
        x = x.view(-1, 1, seq_len, fft_bins)
        x = self.encoder(x)
        
        return x.view(x.size(0), -1)

    def loss_function(self, Y_in, Y_target):
        def support_idxs(c):
            return Y_target.eq(c).nonzero()[:n_support].squeeze(1)

        def euclidean_dist(a, b):
            n = a.shape[0]
            m = b.shape[0]
            a = a.unsqueeze(1).expand(n, m, -1)
            b = b.unsqueeze(0).expand(n, m, -1)
            dist = torch.pow(a - b, 2).sum(dim=2)
            return dist

        n_support = self.conf.train.n_shot

        Y_target = Y_target.to('cpu')
        Y_in = Y_in.to('cpu')

        classes = torch.unique(Y_target)
        n_classes = len(classes)
        p = n_classes * n_support

        n_query = Y_target.eq(classes[0].item()).sum().item() - n_support
        s_idxs = list(map(support_idxs, classes))
        prototypes = torch.stack([Y_in[idx].mean(0) for idx in s_idxs])

        q_idxs = torch.stack(list(map(lambda c:Y_target.eq(c).nonzero()[n_support:], classes))).view(-1)
        q_samples = Y_in.cpu()[q_idxs]

        dists = euclidean_dist(q_samples, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

        target_idxs = torch.arange(0, n_classes)
        target_idxs = target_idxs.view(n_classes, 1, 1)
        target_idxs = target_idxs.expand(n_classes, n_query, 1).long()
        loss_val = -log_p_y.gather(2, target_idxs).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)

        acc_val = y_hat.eq(target_idxs.squeeze()).float().mean()

        return loss_val, acc_val

    def training_step(self, batch, batch_idx):
        X, Y = batch
        Y_out = self(X)
        train_loss, train_acc = self.loss_function(Y_out, Y)

        self.log('train_loss', train_loss)
        self.log('train_acc', train_acc)
        
        log = {'train_acc':train_acc}
        self.log_dict(log, prog_bar=True)

        return {'loss':train_loss}

    def validation_step(self, batch, batch_idx):
        X, Y = batch         
        Y_out = self(X)
        val_loss, val_acc = self.loss_function(Y_out, Y)

        log = {'val_loss':val_loss, 'val_acc':val_acc}
        self.log('val_loss', val_loss)
        self.log('val_acc', val_acc)

        return {'val_loss':val_loss, 'val_acc':val_acc}
    
    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        log = {'avg_val_loss':val_loss, 'avg_val_acc':val_acc}
        self.log('avg_val_loss', val_loss)
        self.log('avg_val_acc', val_acc)

        return {'log':log}
    
    def configure_optimizers(self):
        optimiser = torch.optim.SGD(self.parameters(), 
                                    lr=self.conf.train.lr, 
                                    momentum=self.conf.train.momentum)
        

        lr_scheduler = {'scheduler': ReduceLROnPlateau(optimiser,
                                                       patience=self.conf.train.patience,
                                                       verbose=True),
                         'monitor': 'val_loss'
        }

        return {'optimizer':optimiser, 'lr_scheduler': lr_scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
        
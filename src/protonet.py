import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import hydra
from omegaconf import DictConfig

from datagenerator import *
from features import *
from utils import euclidean_dist

class Protonet(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        def conv_block(in_channels, out_channels, kernel_size=2):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding_mode='zeros'),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
        )
        self.conf = conf
        self.encoder = nn.Sequential(
            conv_block(1, 128),
            conv_block(128, 128),
            conv_block(128, 128)
        )
    
    def forward(self, x):
        (num_samples, seq_len, fft_bins) = x.shape
        x = x.view(-1, 1, seq_len, fft_bins)
        x = self.encoder(x)
        return x.view(x.size(0), -1)

    def loss_function(self, Y_in, Y_target):
        def support_idxs(c):
            return Y_target.eq(c).nonzero()[:n_support].squeeze(1)
        
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
        optimizer = torch.optim.SGD(self.parameters(), 
                                    lr=self.conf.train.lr, 
                                    momentum=self.conf.train.momentum)

        # lr_scheduler = StepLR(optimizer=optimizer, 
        #                       gamma=self.conf.train.gamma,
        #                       step_size=self.conf.train.scheduler_step_size)
        lr_scheduler = {'scheduler': ReduceLROnPlateau(optimizer,
                                                       factor=self.conf.train.factor,
                                                       patience=self.conf.train.patience,
                                                       verbose=True),
                        'monitor': 'val_loss'
        }

        return {'optimizer':optimizer, 'lr_scheduler': lr_scheduler}

    # # learning rate warm-up
    # def optimizer_step(
    #     self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
    #     on_tpu=False, using_native_amp=False, using_lbfgs=False
    # ):

    #     # skip the first 500 steps
    #     if self.trainer.global_step in range(1, 500):
    #         lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
    #         for pg in optimizer.param_groups:
    #             pg['lr'] = lr_scale * pg.get('lr')
                

    #     # hold lr
    #     if self.trainer.global_step in range(500, 10000):
    #         for pg in optimizer.param_groups:
    #             pg['lr'] = pg.get('lr')
                

    #     # decay lr exponentially
    #     if self.trainer.global_step in range(10000, 80000):
    #         if (self.trainer.global_step % 10) == 0:
    #             for pg in optimizer.param_groups:
    #                 pg['lr'] = self.conf.train.gamma * pg.get('lr')
                    

    #     # hold lr again
    #     if self.trainer.global_step >= 80000:
    #         for pg in optimizer.param_groups:
    #             pg['lr'] = pg.get('lr')
                
    #     lr = np.mean([pg['lr'] for pg in optimizer.param_groups])
    #     self.log('lr', lr)
    #     # update params
    #     optimizer.step(closure=optimizer_closure)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def predict_step(self, batch, batch_idx, dataloader_idx):
        X, Y = batch         
        Y_out = self(X)
        return super().predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        

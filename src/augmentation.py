import os
import torch
import torch.nn as nn
from torch import Tensor
import torchaudio
import torchaudio.transforms  as T
from torchaudio import functional as F
import pandas as pd
import numpy as np

class Augementation():
    def __init__(self,   
                 conf=None) -> None:

        self.conf = conf
        self.time_stretch_factor = conf.features.time_stretch
        self.direction = conf.features.direction
    
    def timeStretch(self, file, save_path):
        filename = os.path.basename(file)
        data, sr = torchaudio.load(file)
        resample = T.Resample(sr, self.conf.features.sample_rate)
        data = resample(data)
        spectrogram = T.Spectrogram(
            n_fft=self.conf.features.n_fft,
            hop_length=self.conf.features.hop,
            window_fn=(torch.hamming_window),
            power=None
        )
        data = spectrogram(data)

        ts_u = T.TimeStretch(hop_length=self.conf.features.hop,
                             n_freq = self.conf.features.n_fft // 2 + 1,
                             fixed_rate = (1 + self.time_stretch_factor))
        ts_d = T.TimeStretch(hop_length=self.conf.features.hop,
                            n_freq = self.conf.features.n_fft // 2 + 1,
                            fixed_rate = (1 - self.time_stretch_factor))

        gl = T.GriffinLim(n_fft=self.conf.features.n_fft,
                          hop_length=self.conf.features.hop)
        
        if self.direction == 'up':
            stretched_data_up = ts_u(data)
            stretched_data_up = F.magphase(stretched_data_up)[0]
            stretched_data_up = gl(stretched_data_up)
            torchaudio.save(save_path + filename[:-4] + 'stetched_up_{}.wav'
                            .format(1 + self.time_stretch_factor),
                            stretched_data_up,
                            sr)
            
        elif self.direction == 'down':
            stretched_data_down = ts_d(data)
            stretched_data_up = F.magphase(stretched_data_down)[0]
            stretched_data_down = gl(stretched_data_down)
            torchaudio.save(save_path + filename[:-4] + 'stetched_down_{}.wav'
                            .format(1 - self.time_stretch_factor),
                            stretched_data_down,
                            sr)
        else:
            stretched_data_up = ts_u(data)
            stretched_data_down = ts_d(data)
            
            stretched_data_up = F.magphase(stretched_data_up)[0]
            stretched_data_up = gl(stretched_data_up)
            torchaudio.save(save_path + filename[:-4] + 'stetched_up_{}.wav'
                .format(1 + self.time_stretch_factor),
                stretched_data_up,
                sr)
            
            stretched_data_down = F.magphase(stretched_data_down)[0]
            stretched_data_down = gl(stretched_data_down)
            torchaudio.save(save_path + filename[:-4] + 'stetched_down_{}.wav'
                .format(1 - self.time_stretch_factor),
                stretched_data_down,
                sr)

    def labelAugment(self, labels, save_path):
        df = pd.read_csv(labels, header=0, index_col=False)
        
        if self.direction == 'up':
            filename = df['Audiofilename']
            filename = filename[:-4][0] + 'stetched_{}.wav'.format(1 + self.time_stretch_factor)
            df['Audiofilename'] = filename[:-4]
            df['Starttime'] = np.asarray(df['Starttime'] * (1 + self.time_stretch_factor))
            df['Endtime'] = np.asarray(df['Endtime'] * (1 + self.time_stretch_factor))
            df.to_csv(save_path + filename[:-4][0] + '.csv')
        
        elif self.direction == 'down':
            filename = df['Audiofilename']
            filename = filename[:-4][0] + 'stretched_{}.wav'.format(1 - self.time_stretch_factor)
            df['Audiofilename'] = filename[:-4]
            df['Starttime'] = np.asarray(df['Starttime'] * (1 - self.time_stretch_factor))
            df['Endtime'] = np.asarray(df['Endtime'] * (1 - self.time_stretch_factor))
            df.to_csv(save_path + filename[:-4][0] + '.csv')
        
        else:
            filename = df['Audiofilename']
            filename_up = filename[0][:-4] + 'stretched_{}.wav'.format(1 + self.time_stretch_factor)
            filename_down = filename[0][:-4] + 'stretched_{}.wav'.format(1 - self.time_stretch_factor)
            original_start = np.asarray(df['Starttime'])
            original_end = df['Endtime']
            
            df['Audiofilename'] = filename_up[:-4]
            df['Starttime'] = np.asarray(original_start * (1 + self.time_stretch_factor))
            df['Endtime'] = np.asarray(original_end * (1 + self.time_stretch_factor))
            df.to_csv(save_path + filename_up[:-4] + '.csv', index=None)

            df['Audiofilename'] = filename_down[:-4]
            df['Starttime'] = np.asarray(original_start * (1 - self.time_stretch_factor))
            df['Endtime'] = np.asarray(original_end * (1 - self.time_stretch_factor))
            df.to_csv(save_path + filename_down[:-4] + '.csv', index=None)
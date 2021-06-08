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
    def __init__(self, conf=None) -> None:
        self.conf = conf

    def frequencyMask(self, file):
        # create copy of labels for new file
        df = pd.read_csv(file.replace('wav', 'csv'), 
                         header=0, 
                         index_col=False)
        df2 = df.copy()
        
        # load file and perform timeMask
        filename = os.path.basename(file)
        data, sr = torchaudio.load(file)
        resample = T.Resample(sr, self.conf.features.sample_rate)
        data = resample(data)
        spectrogram = T.MelSpectrogram(
            n_fft=self.conf.features.n_fft,
            hop_length=self.conf.features.hop,
            window_fn=(torch.hamming_window),
            power=2.0
        )
        data_spec = spectrogram(data)
        frequency_mask = T.FrequencyMasking(freq_mask_param=self.conf.features.freq_mask)
        data_mask = frequency_mask(data_spec)

        df2['Audiofilename'] = filename[:-4] + '_freqmask.wav'
        new_labels = df2

        return data_mask, new_labels

    def timeMask(self, file): 
        # create copy of labels for new file
        df = pd.read_csv(file.replace('wav', 'csv'), 
                         header=0, 
                         index_col=False)
        df2 = df.copy()
        
        # load file and perform timeMask
        filename = os.path.basename(file)
        data, sr = torchaudio.load(file)
        resample = T.Resample(sr, self.conf.features.sample_rate)
        data = resample(data)
        spectrogram = T.MelSpectrogram(
            n_fft=self.conf.features.n_fft,
            hop_length=self.conf.features.hop,
            window_fn=(torch.hamming_window),
            power=2.0
        )
        data = spectrogram(data)
        time_mask = T.TimeMasking(time_mask_param=self.conf.features.time_mask)
        data_mask = time_mask(data)

        df2['Audiofilename'] = filename[:-4] + '_timemask.wav'
        new_labels = df2

        return data_mask, new_labels
    
    def timeStretch(self, file, direction):
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
                             fixed_rate = (1 + self.conf.features.time_stretch))
        ts_d = T.TimeStretch(hop_length=self.conf.features.hop,
                            n_freq = self.conf.features.n_fft // 2 + 1,
                            fixed_rate = (1 - self.conf.features.time_stretch))
        
        if direction == 'up':
            stretched_data = ts_u(data)
            stretched_data = F.magphase(stretched_data, power=2.0)[0]
            new_labels = self._labelAugment(file, direction)

        elif direction == 'down':
            stretched_data = ts_d(data)
            stretched_data = F.magphase(stretched_data, power=2.0)[0]
            new_labels = self._labelAugment(file, direction)
            
        stretched_data = T.MelScale()(stretched_data)
        return stretched_data, new_labels

    def _labelAugment(self, file, direction):
        df = pd.read_csv(file.replace('wav', 'csv'), 
                         header=0,
                         index_col=False)
        df2 = df.copy()
        
        if direction == 'up':
            filename = df2['Audiofilename']
            filename = filename[0][:-4] + 'stretched_{}'.format(1 + self.conf.features.time_stretch)
            df2['Audiofilename'] = filename
            df2['Starttime'] = np.asarray(df2['Starttime'] * (1 + self.conf.features.time_stretch))
            df2['Endtime'] = np.asarray(df2['Endtime'] * (1 + self.conf.features.time_stretch))
            return df2
        
        elif direction == 'down':
            filename = df2['Audiofilename']
            filename = filename[0][:-4] + 'stretched_{}'.format(1 - self.conf.features.time_stretch)
            df2['Audiofilename'] = filename
            df2['Starttime'] = np.asarray(df2['Starttime'] * (1 - self.conf.features.time_stretch))
            df2['Endtime'] = np.asarray(df2['Endtime'] * (1 - self.conf.features.time_stretch))
            return df2

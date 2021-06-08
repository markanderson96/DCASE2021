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
    def __init__(self, save_path, conf=None) -> None:
        self.save_path = save_path
        self.conf = conf

    def frequencyMask(self, file):
        # create copy of labels for new file
        df = pd.read_csv(file.replace('wav', 'csv'), 
                         header=0, 
                         index_col=False)
        
        # load file and perform timeMask
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
        frequency_mask = T.FrequencyMasking(freq_mask_param=80)
        data = frequency_mask(data)
        
        data = F.magphase(data, power=2.0)[0]
        
        gl = T.GriffinLim(n_fft=self.conf.features.n_fft,
                          hop_length=self.conf.features.hop)
        
        data_out = gl(data)
        torchaudio.save(self.save_path + filename[:-4] + '_freqmask.wav', data_out, self.conf.features.sample_rate)

        df['Audiofilename'] = filename[:-4] + '_freqmask.wav'
        df.to_csv(self.save_path + filename[:-4] + '_freqmask.csv')

    def timeMask(self, file): 
        # create copy of labels for new file
        df = pd.read_csv(file.replace('wav', 'csv'), 
                         header=0, 
                         index_col=False)
        
        # load file and perform timeMask
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
        time_mask = T.TimeMasking(time_mask_param=80)
        data = time_mask(data)
        
        data = F.magphase(data, power=2.0)[0]
        
        gl = T.GriffinLim(n_fft=self.conf.features.n_fft,
                          hop_length=self.conf.features.hop)
        
        data_out = gl(data)
        torchaudio.save(self.save_path + filename[:-4] + '_timemask.wav', data_out, self.conf.features.sample_rate)
       
        df['Audiofilename'] = filename[:-4] + '_timemask.wav'
        df.to_csv(self.save_path + filename[:-4] + '_timemask.csv')
    
    def timeStretch(self, file, labels):
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

        gl = T.GriffinLim(n_fft=self.conf.features.n_fft,
                          hop_length=self.conf.features.hop)
        
        if self.conf.features.direction == 'up':
            stretched_data_up = ts_u(data)
            stretched_data_up = F.magphase(stretched_data_up, power=2.0)[0]
            stretched_data_up = gl(stretched_data_up)
            torchaudio.save(self.save_path + filename[:-4] + 'stretched_{}.wav'
                            .format(1 + self.conf.features.time_stretch),
                            stretched_data_up,
                            self.conf.features.sample_rate)
            
        elif self.conf.features.direction == 'down':
            stretched_data_down = ts_d(data)
            stretched_data_up = F.magphase(stretched_data_down, power=2.0)[0]
            stretched_data_down = gl(stretched_data_down)
            torchaudio.save(self.save_path + filename[:-4] + 'stretched_{}.wav'
                            .format(1 - self.conf.features.time_stretch),
                            stretched_data_down,
                            self.conf.features.sample_rate)
        else:
            stretched_data_up = ts_u(data)
            stretched_data_down = ts_d(data)
            stretched_data_up = F.magphase(stretched_data_up, power=2.0)[0]
            stretched_data_up = gl(stretched_data_up)
            torchaudio.save(self.save_path + filename[:-4] + 'stretched_{}.wav'
                .format(1 + self.conf.features.time_stretch),
                stretched_data_up,
                self.conf.features.sample_rate)
            
            stretched_data_down = F.magphase(stretched_data_down, power=2.0)[0]
            stretched_data_down = gl(stretched_data_down)
            torchaudio.save(self.save_path + filename[:-4] + 'stretched_{}.wav'
                .format(1 - self.conf.features.time_stretch),
                stretched_data_down,
                self.conf.features.sample_rate)

        self._labelAugment(labels)

    def _labelAugment(self, labels):
        df = pd.read_csv(labels, header=0, index_col=False)
        
        if self.conf.features.direction == 'up':
            filename = df['Audiofilename']
            filename = filename[:-4][0] + 'stetched_{}.wav'.format(1 + self.conf.features.time_stretch)
            df['Audiofilename'] = filename[:-4]
            df['Starttime'] = np.asarray(df['Starttime'] * (1 + self.conf.features.time_stretch))
            df['Endtime'] = np.asarray(df['Endtime'] * (1 + self.conf.features.time_stretch))
            df.to_csv(self.save_path + filename[:-4][0] + '.csv')
        
        elif self.conf.features.direction == 'down':
            filename = df['Audiofilename']
            filename = filename[:-4][0] + 'stretched_{}.wav'.format(1 - self.conf.features.time_stretch)
            df['Audiofilename'] = filename[:-4]
            df['Starttime'] = np.asarray(df['Starttime'] * (1 - self.conf.features.time_stretch))
            df['Endtime'] = np.asarray(df['Endtime'] * (1 - self.conf.features.time_stretch))
            df.to_csv(self.save_path + filename[:-4][0] + '.csv')
        
        else:
            filename = df['Audiofilename']
            filename_up = filename[0][:-4] + 'stretched_{}.wav'.format(1 + self.conf.features.time_stretch)
            filename_down = filename[0][:-4] + 'stretched_{}.wav'.format(1 - self.conf.features.time_stretch)
            original_start = np.asarray(df['Starttime'])
            original_end = df['Endtime']
            
            df['Audiofilename'] = filename_up[:-4]
            df['Starttime'] = np.asarray(original_start * (1 + self.conf.features.time_stretch))
            df['Endtime'] = np.asarray(original_end * (1 + self.conf.features.time_stretch))
            df.to_csv(self.save_path + filename_up[:-4] + '.csv', index=None)

            df['Audiofilename'] = filename_down[:-4]
            df['Starttime'] = np.asarray(original_start * (1 - self.conf.features.time_stretch))
            df['Endtime'] = np.asarray(original_end * (1 - self.conf.features.time_stretch))
            df.to_csv(self.save_path + filename_down[:-4] + '.csv', index=None)

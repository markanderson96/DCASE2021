import torch
import os
import h5py
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

class _DataGenerator(object):
    def __init__(self, conf):
        self.mean = np.mean(self.X[self.train_index])
        self.std = np.std(self.X[self.train_index])
        pass

    def normalise(self, X):
        return (X - self.mean) / self.std

    def _balanceClasses(self, X, Y):
        """
        Class balancing via ROS

        Args:
        - X: Feature Vector
        - Y: Labels

        Returns:
        - X_balanced: Feature after ROS
        - Y_Balanced: Labels after ROS
        """
        x_idx = [[idx] for idx in range(len(X))]
        y_set = set(Y)

        oversample = RandomOverSampler(sampling_strategy='auto', random_state=None)
        x_resampled, _ = oversample.fit_resample(x_idx, Y)
        
        resampled_idx = [idx[0] for idx in x_resampled]

        X_balanced = np.array([X[i] for i in resampled_idx])
        Y_balanced = np.array([Y[i] for i in resampled_idx])

        return X_balanced, Y_balanced

    def _class_to_int(self, label_array,class_set):

        '''  
        Convert class label to integer
        
        Args:
        - label_array: label array
        - class_set: unique classes in label_array
        
        Returns
        - y: label to index values
        '''
        label2indx = {label:index for index,label in enumerate(class_set)}
        y = np.array([label2indx[label] for label in label_array])
        return y

class TrainGenerator(_DataGenerator):
    def __init__(self, conf):
        self.conf = conf
        filepath = os.path.join(self.conf.path.train_feat, 'train.h5')
        train_file = h5py.File(filepath, 'r+')
        
        self.X = train_file['features'][:]
        self.labels = [label.decode() for label in train_file['labels'][:]]
        classes = set(self.labels)
        self.Y = self._class_to_int(self.labels, classes)
        
        self.X, self.Y = self._balanceClasses(self.X, self.Y)
        _, _, self.train_index, self.val_index = train_test_split(self.X, self.Y, random_state=114, stratify=self.Y)

        super().__init__(conf=self.conf)

    def generate_train_data(self, normalise=True):
        """
        create the training and validation data

        Args:
        - normalise: boolean of whether to normalise data or not (default: True)

        Returns:
        - X_train: training features
        - X_val: validation features
        - Y_train: training labels
        - Y_val: validation labels
        """

        X_train = self.X[sorted(self.train_index)]
        Y_train = self.Y[sorted(self.train_index)]
        X_val = self.X[sorted(self.val_index)]
        Y_val = self.Y[sorted(self.val_index)]

        if normalise:
            X_train = self.normalise(X_train)
            Y_train = self.normalise(Y_train)

        return X_train, Y_train, X_val, Y_val

class Datagen_test(TrainGenerator):

    def __init__(self,hf,conf):
        super().__init__(conf= conf)
        self.x_pos = hf['feat_pos'][:]
        self.x_neg = hf['feat_neg'][:]
        self.x_query = hf['feat_query'][:]

    def generate_eval(self):

        '''
        Returns normalizedtest features
        
        Returns:
        - X_pos: Positive set features. Positive class prototypes will be calculated from this
        - X_query: Query set. Onset-offset prediction will be made on this set.
        - X_neg: The entire audio file. Will be used to calculate a negative prototype.
        '''

        X_pos = (self.x_pos)
        X_neg = (self.x_neg)
        X_query = (self.x_query)
        X_pos = self.feature_scale(X_pos)
        X_neg = self.feature_scale(X_neg)
        X_query = self.feature_scale(X_query)

        return X_pos, X_neg, X_query

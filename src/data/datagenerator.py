import h5py
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
import os
import warnings
warnings.filterwarnings("ignore")



def class_to_int(label_array,class_set):

    '''  Convert class label to integer
    Args:
    -label_array: label array
    -class_set: unique classes in label_array
    Out:
    -y: label to index values
    '''
    label2indx = {label:index for index,label in enumerate(class_set)}
    y = np.array([label2indx[label] for label in label_array])
    return y


def balance_class_distribution(X,Y):

    '''  Class balancing through Random oversampling
    Args:
    -X: Feature
    -Y: labels
    Out:
    -X_new: Feature after oversampling
    -Y_new: Oversampled label list
    '''

    x_index = [[index] for index in range(len(X))]
    set_y = set(Y)

    ros = RandomOverSampler()
    x_unifm, y_unifm = ros.fit_resample(x_index, Y)
    unifm_index = [index_new[0] for index_new in x_unifm]

    X_new = np.array([X[index] for index in unifm_index])

    sampled_index = [idx[0] for idx in x_unifm]
    Y_new = np.array([Y[idx] for idx in sampled_index])

    return X_new,Y_new

def norm_params(X):

    '''  Normalize features
        Args:
        - X : Features
        Out:
        - mean : Mean of the feature set
        - std: Standard deviation of the feature set
        '''
    mean = np.mean(X)
    std = np.std(X)
    return mean, std


class Datagen(object):

    def __init__(self, conf):
        hdf_path = os.path.join(conf.path.train_feat, 'train.h5')
        hdf_train = h5py.File(hdf_path, 'r+')
        self.x = hdf_train['features'][:]
        self.labels = [s.decode() for s in hdf_train['labels'][:]]

        class_set = set(self.labels)

        self.y = class_to_int(self.labels,class_set)
        self.x,self.y = balance_class_distribution(self.x,self.y)
        array_train = np.arange(len(self.x))
        _,_,_,_,train_array,valid_array = train_test_split(self.x,self.y,array_train,random_state=42,stratify=self.y)
        self.train_index = train_array
        self.valid_index = valid_array
        self.mean,self.std = norm_params(self.x[train_array])

    def feature_scale(self,X):
        return (X-self.mean)/self.std

    def generate_train(self):
        ''' Returns normalized training and validation features.
        Args:
        -conf - Configuration object
        Out:
        - X_train: Training features
        - X_val: Validation features
        - Y_train: Training labels
        - Y_Val: Validation labels
        '''

        train_array = sorted(self.train_index)
        valid_array = sorted(self.valid_index)
        X_train = self.x[train_array]
        Y_train = self.y[train_array]
        X_val = self.x[valid_array]
        Y_val = self.y[valid_array]
        X_train = self.feature_scale(X_train)
        X_val = self.feature_scale(X_val)
        return X_train,Y_train,X_val,Y_val

class Datagen_test(Datagen):
    def __init__(self,hf,conf):
        super(Datagen_test, self).__init__(conf=conf)
        self.x_pos = hf['feat_positive'][:]
        self.x_neg = hf['feat_negative'][:]
        self.x_query = hf['feat_query'][:]

    def generate_eval(self):
        '''Returns normalized test features
        Output:
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
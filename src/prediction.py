import torch
import logging
import numpy as np

from tqdm import tqdm

from data.datagenerator import Datagen_test
from utils import EpisodicBatchSampler, euclidean_dist
from protonet import Protonet

def eval_prototypes(conf=None,hdf_eval=None,strt_index_query=None):

    """ Run the evaluation
    Args:
     - conf: config object
     - hdf_eval: Features from the audio file
     - device:  cuda/cpu
     - str_index_query : start frame of the query set w.r.t to the original file

     Out:
     - onset: Onset array predicted by the model
     - offset: Offset array predicted by the model
      """
    if conf.set.gpus > 0:
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    hop_seg = int(conf.features.hop_seg * conf.features.sample_rate // conf.features.hop)

    gen_eval = Datagen_test(hdf_eval,conf)
    X_pos, X_neg,X_query = gen_eval.generate_eval()

    X_pos = torch.tensor(X_pos)
    Y_pos = torch.LongTensor(np.zeros(X_pos.shape[0]))
    X_neg = torch.tensor(X_neg)
    Y_neg = torch.LongTensor(np.zeros(X_neg.shape[0]))
    X_query = torch.tensor(X_query)
    Y_query = torch.LongTensor(np.zeros(X_query.shape[0]))

    num_batch_query = len(Y_query) // conf.eval.query_batch_size

    query_dataset = torch.utils.data.TensorDataset(X_query, Y_query)
    q_loader = torch.utils.data.DataLoader(dataset=query_dataset, batch_sampler=None,batch_size=conf.eval.query_batch_size,shuffle=False)
    q_set_size = X_query.shape[1] * X_query.shape[2]
    #breakpoint()
    query_set_feat = torch.zeros(0, 3968).cpu()


    Model = Protonet(conf)

    Model.load_from_checkpoint(conf.path.best_model, conf=conf)

    Model.to(device)
    Model.eval()

    'List for storing the combined probability across all iterations'
    prob_comb = []

    iterations = conf.eval.iterations
    for i in range(iterations):
        prob_pos_iter = []
        neg_indices = torch.randperm(len(X_neg))[:conf.eval.samples_neg]
        X_neg = X_neg[neg_indices]
        Y_neg = Y_neg[neg_indices]
        batch_size_neg = conf.eval.negative_set_batch_size
        neg_dataset = torch.utils.data.TensorDataset(X_neg, Y_neg)
        negative_loader = torch.utils.data.DataLoader(dataset=neg_dataset, batch_sampler=None, batch_size=batch_size_neg)

        batch_samplr_pos = EpisodicBatchSampler(Y_pos, num_batch_query + 1, 1, conf.train.n_shot)
        pos_dataset = torch.utils.data.TensorDataset(X_pos, Y_pos)
        pos_loader = torch.utils.data.DataLoader(dataset=pos_dataset, batch_sampler=batch_samplr_pos)

        neg_iterator = iter(negative_loader)
        pos_iterator = iter(pos_loader)
        q_iterator = iter(q_loader)

        print("Iteration number {}".format(i))

        for batch in tqdm(neg_iterator):
            x_neg, y_neg = batch
            x_neg = x_neg.to(device)
            feat_neg = Model(x_neg)
            feat_neg = feat_neg.detach().cpu()
            query_set_feat = torch.cat((query_set_feat, feat_neg), dim=0)
        
        neg_prototype = query_set_feat.mean(dim=0)
        neg_prototype = neg_prototype.to(device)

        for batch in tqdm(q_iterator):
            x_q, y_q = batch
            x_q = x_q.to(device)
            x_pos, y_pos = next(pos_iterator)
            x_pos = x_pos.to(device)
            x_pos = Model(x_pos)
            x_query = Model(x_q)
            probability_pos = get_probability(x_pos, neg_prototype, x_query)
            prob_pos_iter.extend(probability_pos)

        prob_comb.append(prob_pos_iter)
    
    prob_final = np.mean(np.array(prob_comb),axis=0)

    #breakpoint()

    krn = np.array([1, 0 -1])
    prob_thresh = np.where(prob_final > conf.eval.threshold, 1, 0)

    prob_pos_final = prob_final * prob_thresh
    changes = np.convolve(krn, prob_thresh)

    onset_frames = np.where(changes == 1)[0]
    offset_frames = np.where(changes == -1)[0]

    str_time_query = strt_index_query * conf.features.hop / conf.features.sample_rate

    onset = (onset_frames + 1) * (hop_seg) * conf.features.hop / conf.features.sample_rate
    onset = onset + str_time_query

    offset = (offset_frames + 1) * (hop_seg) * conf.features.hop / conf.features.sample_rate
    offset = offset + str_time_query

    assert len(onset) == len(offset)
    return onset, offset



def get_probability(x_pos,neg_proto,query_set_out):


    """
    Calculates the  probability of each query point belonging to either the positive or negative class
    
    Args:
     - x_pos : Model output for the positive class
     - neg_proto : Negative class prototype calculated from randomly chosed 100 segments across the audio file
     - query_set_out:  Model output for the first 8 samples of the query set
    
    Returns:
     - Probabiility array for the positive class
     """

    pos_prototype = x_pos.mean(0)
    prototypes = torch.stack([pos_prototype,neg_proto])
    dists = euclidean_dist(query_set_out,prototypes)
    '''  Taking inverse distance for converting distance to probabilities'''
    inverse_dist = torch.div(1.0, dists)
    prob = torch.softmax(inverse_dist, dim=1)
    '''  Probability array for positive class'''
    prob_pos = prob[:,0]
    
    return prob_pos.detach().cpu().tolist()

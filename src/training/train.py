import torch
import numpy as np
from tqdm import tqdm
from .loss_functions import prototypical_loss_fn as loss_fn

def train(model, train_loader, val_loader, num_batches_train, num_batches_val, conf):
    if conf.train.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    optimiser = torch.optim.SGD(model.parameters(), 
                                lr=conf.train.lr, 
                                momentum=conf.train.momentum)
    epochs = conf.train.epochs

    train_loss_list = np.array([])
    train_acc_list = np.array([])
    val_loss_list = np.array([])
    val_acc_list = np.array([])
    model.to(device)

    for epoch in range(epochs + 1):
        train_iterator = iter(train_loader)
        train_batches = tqdm(train_iterator, desc='Epoch: {}'.format(epoch))
        for batch in train_batches:
            optimiser.zero_grad()
            model.train()
            X, Y = batch
            X = X.to(device)
            Y = Y.to(device)
            
            Y_out = model(X)
            breakpoint()
            train_loss, train_acc = loss_fn(Y_out, Y, conf)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            
            optimiser.step()
            train_batches.update(1)
            train_batches.set_postfix_str(s='Train Loss: {}\tTrain Acc: {}'.format(train_loss, train_acc))

        avg_loss = np.mean(train_loss_list[-num_batches_train:])
        avg_acc = np.mean(train_acc_list[-num_batches_train:])
        print('Average train loss: {}\tAverage training accuracy: {}'.format(avg_loss, avg_acc))

        model.eval()
        val_iterator = iter(val_loader)
        for batch in val_iterator:
            X, Y = batch
            X = X.to(device)
            Y = Y.to(device)
            Y_out = model(X)
            val_loss, val_acc = loss_fn(Y_out, Y, conf)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

        avg_loss = np.mean(val_loss_list[-num_batches_val:])
        avg_acc = np.mean(val_acc_list[-num_batches_val:])
        print('Average validation loss: {}\tAverage validation accuracy: {}'.format(avg_loss, avg_acc))
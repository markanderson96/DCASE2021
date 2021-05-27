import torch
import logging
import numpy as np
from tqdm import tqdm
from .loss_functions import prototypical_loss_fn as loss_fn

log_fmt = '%(asctime)s - %(module)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

def train(model, train_loader, val_loader, num_batches_train, num_batches_val, conf):
    if conf.train.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    optimiser = torch.optim.SGD(model.parameters(), 
                                lr=conf.train.lr, 
                                momentum=conf.train.momentum)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimiser,
                                                   gamma=conf.train.scheduler_gamma,
                                                   step_size=conf.train.scheduler_step_size)
    epochs = conf.train.epochs

    best_model_path = conf.path.best_model
    last_model_path = conf.path.last_model

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    
    best_val_acc = 0.0
    
    model.to(device)

    for epoch in range(epochs):
        train_iterator = iter(train_loader)
        train_batches = tqdm(train_iterator, desc='Epoch: {}'.format(epoch + 1))
        for batch in train_batches:
            optimiser.zero_grad()
            model.train()
            X, Y = batch
            X = X.to(device)
            Y = Y.to(device)
            
            Y_out = model(X)
            #breakpoint()
            train_loss, train_acc = loss_fn(Y_out, Y, conf)
            train_loss_list.append(train_loss.item())
            train_acc_list.append(train_acc.item())
            
            train_loss.backward()
            optimiser.step()
            train_batches.update(1)
            train_batches.set_postfix_str(s='Train Loss: {:.4f}  Train Acc: {:.4f}'.format(train_loss, train_acc))

        avg_loss = np.mean(train_loss_list[-num_batches_train:])
        avg_acc = np.mean(train_acc_list[-num_batches_train:])
        logger.info('Epoch: {}  Average train loss: {:.4f}  Average training accuracy: {:.4f}'.format(epoch + 1, avg_loss, avg_acc))

        lr_scheduler.step()
        model.eval()
        val_iterator = iter(val_loader)
        for batch in val_iterator:
            X, Y = batch
            X = X.to(device)
            Y = Y.to(device)
            Y_out = model(X)
            val_loss, val_acc = loss_fn(Y_out, Y, conf)
            val_loss_list.append(val_loss.item())
            val_acc_list.append(val_acc.item())

        avg_loss = np.mean(val_loss_list[-num_batches_val:])
        avg_acc = np.mean(val_acc_list[-num_batches_val:])
        logger.info('Epoch: {}  Average validation loss: {:.4f}  Average validation accuracy: {:.4f}'.format(epoch + 1, avg_loss, avg_acc))

        if avg_acc > best_val_acc:
            logger.info("Saving the best model with valdation accuracy {:.4f}".format(avg_acc))
            best_val_acc = avg_acc
            best_state = model.state_dict()
            torch.save(model.state_dict(),best_model_path)
    
    torch.save(model.state_dict(),last_model_path)

    return best_val_acc
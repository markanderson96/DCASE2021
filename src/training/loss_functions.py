import torch
import torch.nn.functional as F 
from utils import euclidean_dist

def prototypical_loss_fn(Y_in, Y_target, conf):
    def support_idxs(c):
        return Y_target.eq(c).nonzero()[:n_support].squeeze(1)

    device = conf.train.device
    n_support = conf.train.n_shot

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
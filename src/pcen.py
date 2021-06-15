import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pcen(x, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, last_state=None, empty=True):
    frames = x.split(1, -2)
    m_frames = []
    if empty:
        last_state = None
    for frame in frames:
        if last_state is None:
            last_state = frame
            m_frames.append(frame)
            continue
        m_frame = (1 - s) * last_state + s * frame
        last_state = m_frame
        m_frames.append(m_frame)
    M = torch.cat(m_frames, 1)
    pcen_ = x.div_(M.add_(eps).pow_(alpha)).add_(delta).pow_(r).sub_(delta ** r)
    return pcen_, last_state


class PCENTransform(nn.Module):

    def __init__(self, eps=1E-6, s=0.025, alpha=0.98, delta=2, r=0.5, conf=None):
        super().__init__()
        self.s = s
        self.alpha = alpha
        self.delta = delta
        self.r = r
        self.eps = eps
        self.conf = conf
        self.register_buffer("last_state", torch.zeros(conf.features.n_mel))
        self.reset()

    def reset(self):
        self.empty = True

    def forward(self, x):
        x, ls = pcen(x, self.eps, self.s, self.alpha, self.delta, self.r, self.last_state, self.empty)
        self.last_state = ls.detach()
        self.empty = False
        return x
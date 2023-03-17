from bisect import bisect_right
from collections import Counter

import torch


class MultiStepLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]


class ExponentialLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, decay_epochs, gamma=0.1, last_epoch=-1):
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch / self.decay_epochs)
                for base_lr in self.base_lrs]


class ExponentialLR_two_part(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, decay_epochs1, decay_epochs2, gamma=0.1, cfg_lr=0, last_epoch=-1):
        self.decay_epochs1 = decay_epochs1
        self.decay_epochs2 = decay_epochs2
        self.gamma = gamma
        self.cfg_lr = cfg_lr
        super(ExponentialLR_two_part, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch / self.decay_epochs1) if base_lr == self.cfg_lr \
          else base_lr * self.gamma ** (self.last_epoch / self.decay_epochs2)
                for base_lr in self.base_lrs]

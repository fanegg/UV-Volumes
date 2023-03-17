from collections import Counter
from lib.utils.optimizer.lr_scheduler import MultiStepLR, ExponentialLR, ExponentialLR_two_part


def make_lr_scheduler(cfg, optimizer):
    cfg_scheduler = cfg.train.scheduler
    if cfg_scheduler.type == 'multi_step':
        scheduler = MultiStepLR(optimizer,
                                milestones=cfg_scheduler.milestones,
                                gamma=cfg_scheduler.gamma)
    elif cfg_scheduler.type == 'exponential':
        scheduler = ExponentialLR(optimizer,
                                  decay_epochs=cfg_scheduler.decay_epochs1,
                                  gamma=cfg_scheduler.gamma)
    elif cfg_scheduler.type == 'exponential_two_part':
        scheduler = ExponentialLR_two_part(optimizer,
                                  decay_epochs1=cfg_scheduler.decay_epochs1,
                                  decay_epochs2=cfg_scheduler.decay_epochs2,
                                  gamma=cfg_scheduler.gamma,
                                  cfg_lr=cfg.train.lr)
    return scheduler


def set_lr_scheduler(cfg, scheduler):
    cfg_scheduler = cfg.train.scheduler
    if cfg_scheduler.type == 'multi_step':
        scheduler.milestones = Counter(cfg_scheduler.milestones)
    elif cfg_scheduler.type == 'exponential':
        scheduler.decay_epochs = cfg_scheduler.decay_epochs1
    scheduler.gamma = cfg_scheduler.gamma

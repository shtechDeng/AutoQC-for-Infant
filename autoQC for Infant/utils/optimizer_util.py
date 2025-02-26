# encoding:utf-8
import sys
import torch.optim as optim


def optimizers(config, model, logger):
    if config['optimizer'] == 'sgd':
        return optim.SGD(model.parameters(),
                         lr=config['learning_rate'], momentum=config['momentum'],
                         weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'asgd':
        return optim.ASGD(model.parameters(),
                          lr=config['learning_rate'], momentum=config['momentum'],
                          weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'rms':
        return optim.RMSprop(model.parameters(),
                             lr=config['learning_rate'], momentum=config['momentum'],
                             weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adam':
        return optim.Adam(model.parameters(),
                          lr=config['learning_rate'],
                          weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        return optim.AdamW(model.parameters(),
                           lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])
    else:
        logger.error('The optimizer name: {} is invalid'
                     .format(config['optimizer']))
        sys.exit()


def adjust_learning_rate(config, optimizer, logger):
    if config['scheduler_mode'] == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['step_size'],
            gamma=config['gamma'],
            last_epoch=config['last_epoch']
        )
        return lr_scheduler
    elif config['scheduler_mode'] == 'multi':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config['milestones'],
            gamma=config['gamma'],
            last_epoch=config['last_epoch']
        )
        return lr_scheduler
    elif config['scheduler_mode'] == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config['gamma'],
            last_epoch=config['last_epoch']
        )
        return lr_scheduler
    elif config['scheduler_mode'] == 'cos':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['T_max'],
            eta_min=config['eta_min'],
            last_epoch=config['last_epoch']
        )
        return lr_scheduler
    elif config['scheduler_mode'] == 'auto':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config['mode']
        )
        return lr_scheduler
    else:
        logger.error('The lr scheduler mode: {} is invalid'
                     .format(config['scheduler_mode']))
        sys.exit()

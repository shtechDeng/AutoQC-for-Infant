# encoding:utf-8
import logging


def loggers(cfg):
    log = logging.getLogger()
    log.setLevel(logging.INFO)

    fh = logging.FileHandler(cfg['log_filename_train_test'])
    sh = logging.StreamHandler()

    fm = logging.Formatter('%(asctime)s - %(filename)s [line:%(lineno)d] - %(levelname)s: %(message)s')

    fh.setFormatter(fm)
    sh.setFormatter(fm)

    log.addHandler(fh)
    log.addHandler(sh)

    output_config(cfg, log)

    return log


def output_config(cfg, log):
    for key, val in cfg.items():
        log.info(f'{key}:{val}')

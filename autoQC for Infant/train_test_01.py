import sys
import time
import torch
import json
from utils.dataset_util import train_test_dataloader
from utils.device_util import devices
from utils.logger_util import loggers
from utils.loss_util import loss_functions
from utils.model_util import models
from utils.optimizer_util import optimizers, adjust_learning_rate
from utils.save_util import save_result_image_loss, save_model
from utils.visualizer_util import Visualizers


def train_01(m, e, trld, opt, cri1, log, dev, cfg):
    m.train()
    total_loss = 0
    samples = 0
    idx = 0
    for idx, (data, target, nifti_path) in enumerate(trld):
        data = data.type(torch.float32).to(dev, non_blocking=True)
        target = target.to(dev, non_blocking=True)

        opt.zero_grad()
        output = m(data)
        loss = cri1(output, target.long())

        loss.backward()
        opt.step()

        total_loss += loss.data.item()
        samples += data.shape[0]

        outputstring = 'Train epoch: {} batch: [{}~{}/{}], learn_rate: {:.8f}, loss-{}: {:.8f}' \
            .format(e, samples - data.shape[0] + 1, samples, len(trld.dataset), opt.param_groups[0]['lr'],
                    type(cri1).__name__, loss.data.item())
        log.info(outputstring)

    avg_loss = total_loss / (idx + 1)

    outputstring = 'Train epoch: {}, average {}: {:.8f}'.format(e, type(cri1).__name__, avg_loss)
    log.info(outputstring)

    with open('train_loss.txt', 'a') as fl:
        fl.write('{}:{}\n'.format(e, avg_loss))

    return avg_loss


def test_01(m, e, tsld, eva1, log, dev, cfg):
    m.eval()
    total_loss = 0
    samples = 0
    acc = 0
    sum = len(tsld)
    for idx, (data, target, nifti_path) in enumerate(tsld):
        data = data.type(torch.float32).to(dev, non_blocking=True)
        target = target.to(dev, non_blocking=True)

        output = m(data)
        result = output.argmax()
        if result.int() == target[0].int():
            acc += 1

        loss = eva1(output, target.long())

        total_loss += loss.data.item()
        samples += data.shape[0]
        outputstring = 'Test epoch: {} batch: [{}~{}/{}], loss-{}: {:.8f}' \
            .format(e, samples - data.shape[0] + 1, samples, len(tsld.dataset),
                    type(eva1).__name__, loss.data.item())
        log.info(outputstring)

    acc_rate = acc / sum
    avg_loss = total_loss / (idx + 1)

    outputstring = 'Test epoch: {}, average {}: {:.8f} acc:{}'.format(e, type(eva1).__name__, avg_loss,acc_rate)
    log.info(outputstring)

    with open('test_loss.txt', 'a') as fl:
        fl.write('{}:{}  acc:{}\n'.format(e, avg_loss, acc_rate))

    return avg_loss


if __name__ == '__main__':
    """初始化环境与模型"""
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    logger = loggers(config)

    # visual = Visualizers(config['visualization_train_test'])

    device = devices(config, logger)

    trainloader, testloader = train_test_dataloader(config)

    general_config = config['general_model']
    model = models(general_config, device, logger)
    # model = torch.nn.DataParallel(model)
    model.to(device)

    criterion_1 = loss_functions(config['train_loss_function'], logger)
    best_train_loss = float('Inf')

    evaluation_1 = loss_functions(config['test_loss_function'], logger)
    best_test_loss = float('Inf')

    optimizer = optimizers(general_config, model, logger)

    scheduler = adjust_learning_rate(general_config, optimizer, logger)

    start_time = time.time()
    """模型周期迭代优化"""
    for epoch in range(int(config['epochs'])):
        # 训练模型
        train_avg_loss = train_01(model, epoch, trainloader,
                                  optimizer, criterion_1,
                                  logger, device, config)

        # 测试模型
        test_avg_loss = test_01(model, epoch, testloader,
                                evaluation_1, logger, device, config)

        # 保存模型
        if test_avg_loss < best_test_loss:
            best_test_loss = test_avg_loss
            save_model(config, model, optimizer, epoch, best_test_loss)

        # 调整学习率
        if general_config['scheduler_mode'] != 'auto':
            scheduler.step()
        elif general_config['scheduler_mode'] == 'auto':
            scheduler.step(metrics=test_avg_loss)
        else:
            print("The lr scheduler mode is invalid")
            sys.exit()

    end_time = time.time()
    print(f'Training total cost time: {(end_time - start_time):.2f} second')

# encoding:utf-8
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


class DiceLoss(nn.Module):
    def __init__(self, eps=1, threshold=0.5, ignore_channels=None):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.threshold = threshold
        self.ignore_channels = ignore_channels

    def forward(self, probs, targets):
        assert probs.shape[0] == targets.shape[0]

        probs = _threshold(probs, threshold=self.threshold)
        pr, gt = _take_channels(probs, targets, ignore_channels=self.ignore_channels)

        tp = torch.sum(gt * pr)
        fp = torch.sum(pr) - tp
        fn = torch.sum(gt) - tp
        score = (2 * tp + self.eps) / (2 * tp + fn + fp + self.eps)

        return score


class IouLoss(nn.Module):
    def __init__(self, eps=1, threshold=0.5, ignore_channels=None):
        super().__init__()
        self.eps = eps
        self.threshold = threshold
        self.ignore_channels = ignore_channels

    def forward(self, probs, targets):
        probs = _threshold(probs, threshold=self.threshold)
        pr, gt = _take_channels(probs, targets, ignore_channels=self.ignore_channels)

        intersection = torch.sum(gt * pr)
        union = torch.sum(gt) + torch.sum(pr) - intersection + self.eps
        score = (intersection + self.eps) / union

        return score


class NpccLoss(nn.Module):
    def __init__(self, reduction=True):
        super(NpccLoss, self).__init__()
        self.reduce = reduction

    def forward(self, preds, targets):
        targets = targets.view(targets.size(0), targets.size(1), -1)
        preds = preds.view(preds.size(0), preds.size(1), -1)

        pr = preds - torch.mean(preds, dim=-1).unsqueeze(-1)
        gt = targets - torch.mean(targets, dim=-1).unsqueeze(-1)

        score = - torch.sum(pr * gt, dim=2) / (torch.sqrt(torch.sum(pr ** 2, dim=2)) * torch.sqrt(torch.sum(gt ** 2,
                                                                                                            dim=2)))
        if self.reduce is True:
            return score.mean()

        return score


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])

    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data5d.type() == img1.data5d.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def loss_functions(loss_name, logger):
    if loss_name == 'mae':
        return nn.L1Loss()
    elif loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'bce':
        return nn.BCELoss()
    elif loss_name == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_name == 'dice':
        return DiceLoss()
    elif loss_name == 'iou':
        return IouLoss()
    elif loss_name == 'npcc':
        return NpccLoss()
    elif loss_name == 'ssim':
        return SSIM()
    else:
        logger.error('The loss function name: {} is invalid'.format(loss_name))
        sys.exit()

# encoding:utf-8
import os
import time
import torch
import numpy as np
from torchvision import utils as vutils
from scipy.io import savemat
from PIL import Image
import PIL.ImageOps
import sys
import cv2


def save_model(config, model, optimizer, epoch, loss=None, mode=None):
    save_path = config['model_save_dir']
    aim = config['network_aim']
    mn = type(model).__name__
    lr = optimizer.param_groups[0]['lr']
    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 文件命名逻辑-[任务名称]_[模型名称]_[平均损失]_[周期]_[学习率]_[时间戳]
    if loss is not None:
        if mode is not None:
            torch.save(model.state_dict(), os.path.join(save_path, f'{aim}_{mn}_{loss:.6f}_{epoch}_{lr}_{timestamp}.pth'))
        else:
            torch.save(model, os.path.join(save_path, f'{aim}_{mn}_{loss:.6f}_{epoch}_{lr}_{timestamp}.pth'))
    # 文件命名逻辑-[任务名称]_[模型名称]_[周期]_[学习率]_[时间戳]
    else:
        if mode is not None:
            torch.save(model.state_dict(), os.path.join(save_path, f'{aim}_{mn}_{epoch}_{lr}_{timestamp}.pth'))
        else:
            torch.save(model, os.path.join(save_path, f'{aim}_{mn}_{epoch}_{lr}_{timestamp}.pth'))


def create_eval_dir(save_dir):
    if not os.path.exists(save_dir):
        save_dir = os.path.join(save_dir, '0')
        os.makedirs(save_dir)
    else:
        fn_list = list(map(int, os.listdir(save_dir)))
        if len(fn_list) == 0:
            save_dir = os.path.join(save_dir, '0')
        else:
            save_dir = os.path.join(save_dir, str(max(fn_list) + 1))
        os.makedirs(save_dir)
    return save_dir


def save_loss_in_text(save_dir, save_filename, fn_list, loss_list):
    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
    save_strings = f'{timestamp}\n'
    for fn_file, loss in zip(fn_list, loss_list):
        fn = ''
        for idx in range(len(fn_file)):
            fn += f'{fn_file[idx]}'
        save_strings += f'{fn} {loss}\n'

    with open(os.path.join(save_dir, save_filename), 'w') as f:
        f.write(save_strings)


def save_img(img, img_path, norm=True, to_gray=False, color=False):
    # img 可以是 tensor 类型 [b c h w]/[c h w]/[h w] 的形状
    grid = vutils.make_grid(img, nrow=8, padding=2, normalize=norm, range=None, scale_each=False, pad_value=0)
    if norm:
        ndarray = grid.mul(255).permute(1, 2, 0).to(dtype=torch.uint8).numpy()
    else:
        # 加 0.5 使其四舍五入到 [0,255] 中最接近的整数
        ndarray = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(dtype=torch.uint8).numpy()
    img = Image.fromarray(ndarray)

    if to_gray:
        # Gray = 0.29900 * R + 0.58700 * G + 0.11400 * B
        img.convert('L')

    if color:
        img = cv2.applyColorMap(np.asarray(img), cv2.COLORMAP_HSV)
        img = Image.fromarray(img)

    img.save(img_path)



def save_result_image_loss(base_path, epoch, fn_list, tensor_img, loss=None, aim=None, save_raw=False, ab_pattern_label=None, color=False):
    timestamp = time.strftime('%m%d%H%M', time.localtime())
    save_dir = os.path.join(base_path, f'epoch_{epoch}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 先切断数据联系再切断计算图联系
    if tensor_img.device != 'cpu':
        data = tensor_img.clone().detach().cpu()
    else:
        data = tensor_img.clone().detach()

    if ab_pattern_label!=None:
        if ab_pattern_label.device != 'cpu':
            pattern_label = ab_pattern_label.clone().detach().cpu()
        else:
            pattern_label= ab_pattern_label.clone().detach()

        # s_img = torch.from_numpy(np.vstack((data,pattern_label)))
        s_img = torch.from_numpy(np.append(data, pattern_label, axis=2))

    if loss is not None:
        loss = round(loss, 6)

    # 文件命名逻辑-[任务名称]_[平均损失]_[文件名]
    for idx, fn in enumerate(fn_list):
        filename, ext = os.path.splitext(fn)
        if ext == '.mat':
            if save_raw:
                savemat(os.path.join(save_dir, f'{aim}_{loss}_{fn}'), {f'{filename}': data[idx].numpy()})

        elif ext == '.npy':
            if save_raw:
                np.save(os.path.join(save_dir, f'{aim}_{loss}_{fn}'), data[idx].numpy())

        if len(data.shape) == 3:
            save_dir_sub = os.path.join(save_dir, f'{aim}_{loss}_{filename}_{timestamp}')
            if not os.path.exists(save_dir_sub):
                os.makedirs(save_dir_sub)
            for k, img in enumerate(data):
                if ab_pattern_label != None:
                    save_img(s_img, os.path.join(save_dir_sub, f'{k}.png'), norm=True, to_gray=True)
                    if color:
                        save_img(s_img, os.path.join(save_dir_sub, f'color.png'), norm=True, to_gray=True, color=True)
                else:
                    save_img(img, os.path.join(save_dir_sub, f'{k}.png'), norm=True, to_gray=True)
        elif len(data.shape) == 2:
            save_img(data, os.path.join(save_dir, f'{aim}_{loss}_{filename}.png'), norm=False, to_gray=True)
        else:
            print("ndarray shape error")
            sys.exit()

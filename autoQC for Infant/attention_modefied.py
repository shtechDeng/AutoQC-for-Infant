import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
# import seaborn as sns
import cv2


# sns.set_theme()

from IPython import display
import pydicom
import dicom2nifti
import nibabel as nib
from tqdm import tqdm
from sklearn.utils import shuffle
from collections import Counter
from skimage.exposure import rescale_intensity
from skimage.transform import rescale, rotate
from torchvision.transforms import Compose

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import GuidedGradCam
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

import utils
heat_map_dir = '/hpc/data/home/bme/yubw/taotl/model8/heat_map'

data_nifti_dir='/hpc/data/home/bme/yubw/taotl/model8/jiyue_nii_selected/'
train_csv_path = '/hpc/data/home/bme/yubw/taotl/model8/jiyue_try.csv'

def auto_select_box(nib_data):
    nib_data[nib_data<30] = 0
    
    dim_1_sum = nib_data.sum(axis=(1,2)).astype(np.int32)
    dim_2_sum = nib_data.sum(axis=(0,2)).astype(np.int32)
    dim_3_sum = nib_data.sum(axis=(0,1)).astype(np.int32)

    sum_d1 = 0
    max_sum_d1 = 0
    for i in range(192):
        sum_d1 += dim_1_sum[i]
    for i in range(192, len(dim_1_sum)):
        sum_d1 = sum_d1 + dim_1_sum[i] - dim_1_sum[i-192]
        if sum_d1 > max_sum_d1:
            max_sum_d1 = sum_d1
            x = i

    sum_d2 = 0
    max_sum_d2 = 0
    for i in range(256):
        sum_d2 += dim_2_sum[i]
    for i in range(256, len(dim_2_sum)):
        sum_d2 = sum_d2 + dim_2_sum[i] - dim_2_sum[i-256]
        if sum_d2 > max_sum_d2:
            max_sum_d2 = sum_d2
            y = i
    
    max_grad = 0
    flag = False
    for i in range(224, len(dim_3_sum)-1):
        l_dis = 8
        r_dis = min(12, len(dim_3_sum) - i - 1)
        
        l = (dim_3_sum[i-l_dis] - dim_3_sum[i])/l_dis
        r = (dim_3_sum[i+r_dis] - dim_3_sum[i])/r_dis
        
        if l>0 and r > -6000:
            flag = True
        if not flag or (l>0 and l+r > max_grad):
            max_grad = l+r
            z = i
        
    return x, y, z

class MRI_Data(Dataset):
    def __init__(
            self,
            data_path,
            transform=None
    ):
        data_list = []
        self.data_path = data_path
        self.data_len = len(data_path)
        self.transform = transform
        with open(data_path) as f:
            for row in f.readlines():
                data_path , label = row.split(' ')
                data_list.append((data_path, label))

        self.data_list = data_list

    def __getitem__(self, index):
        nifti_path , label = self.data_list[index]
        nib_data = nib.load(nifti_path).get_fdata()
        x, y, z = nib_data.shape



        if self.mode == 'train' or self.mode == 'test':
            if self.transform is not None:
                # 实例化自定义数据集时加入transform转换
                nib_data = self.transform(nib_data)

            if np.random.rand() < 0.5:
                # a, b, c = _select_box(nib_data)
                # a = int(np.random.choice(list(range(max(a - 4, 191), min(a + 5, x - 1))), 1))
                # b = int(np.random.choice(list(range(max(b - 4, 223), min(b + 5, y - 1))), 1))
                # c = int(np.random.choice(list(range(max(c - 4, 223), min(c + 5, z - 1))), 1))
                # nib_data = nib_data[a - 191:a + 1, b - 223:b + 1, c - 223:c + 1]

                # new_index = np.random.choice(self.data_len)
                # while abs(self.Age[new_index] - self.Age[index]) > 10:
                #     new_index = np.random.choice(self.data_len)
                # new_nifti_path = os.path.join(self.data_nifti_dir, self.data_path[new_index])
                # new_nib_data = nib.load(new_nifti_path).get_fdata()
                # x, y, z = new_nib_data.shape

                # if self.transform is not None:
                #     new_nib_data = self.transform(new_nib_data)

                # a, b, c = _select_box(new_nib_data)
                # a = int(np.random.choice(list(range(max(a - 4, 191), min(a + 5, x - 1))), 1))
                # b = int(np.random.choice(list(range(max(b - 4, 223), min(b + 5, y - 1))), 1))
                # c = int(np.random.choice(list(range(max(c - 4, 223), min(c + 5, z - 1))), 1))
                # nib_data = nib_data[a - 191:a + 1, b - 223:b + 1, c - 223:c + 1]
                # print(nib_data.shape)
                # if np.random.rand() < 0.5:
                #     nib_data = np.concatenate((nib_data[:96, :, :], new_nib_data[96:, :, :]), axis=0)
                # else:
                #     nib_data = np.concatenate((new_nib_data[:96, :, :], nib_data[96:, :, :]), axis=0)
                # label = torch.tensor((self.Age[index] + self.Age[new_index]) / 2, dtype=torch.float32)
                label = torch.tensor(label, dtype=torch.float32)

            else:
                # a, b, c = _select_box(nib_data)
                # a = int(np.random.choice(list(range(max(a - 4, 191), min(a + 5, x - 1))), 1))
                # b = int(np.random.choice(list(range(max(b - 4, 223), min(b + 5, y - 1))), 1))
                # c = int(np.random.choice(list(range(max(c - 4, 223), min(c + 5, z - 1))), 1))
                # nib_data = nib_data[a - 191:a + 1, b - 223:b + 1, c - 223:c + 1]
                # print(nib_data.shape)

                label = torch.tensor(label, dtype=torch.float32)

            p10 = np.percentile(nib_data, 10)
            p99 = np.percentile(nib_data, 99)
            nib_data = rescale_intensity(nib_data, in_range=(p10, p99), out_range=(0, 1))

            m = np.mean(nib_data, axis=(0, 1, 2))
            s = np.std(nib_data, axis=(0, 1, 2))
            nib_data = (nib_data - m) / s

            nib_data = torch.tensor(nib_data, dtype=torch.float32)
        elif self.mode == 'valid':
            # a, b, c = _select_box(nib_data)

            # nib_data = nib_data[a - 191:a + 1, b - 223:b + 1, c - 223:c + 1]
            p10 = np.percentile(nib_data, 10)
            p99 = np.percentile(nib_data, 99)
            nib_data = rescale_intensity(nib_data, in_range=(p10, p99), out_range=(0, 1))
            m = np.mean(nib_data, axis=(0, 1, 2))
            s = np.std(nib_data, axis=(0, 1, 2))
            nib_data = (nib_data - m) / s

            nib_data = torch.tensor(nib_data, dtype=torch.float32)
            label = torch.tensor(self.Age[index], dtype=torch.float32)
        else:
            print("mode must in [train, valid, 'test']")

        # if self.mode == 'train':
        #     if self.transform is not None:
        #         # 实例化自定义数据集时加入transform转换
        #         nib_data = self.transform(nib_data)

        #     if np.random.rand() < 0.5:
        #         a, b, c = auto_select_box(nib_data)
        #         a = int(np.random.choice(list(range(max(a - 4, 191), min(a + 5, x - 1))), 1))
        #         b = int(np.random.choice(list(range(max(b - 4, 255), min(b + 5, y - 1))), 1))
        #         c = int(np.random.choice(list(range(max(c - 4, 223), min(c + 5, z - 1))), 1))
        #         nib_data = nib_data[a - 191:a + 1, b - 255:b + 1, c - 223:c + 1]

        #         new_index = np.random.choice(self.data_len)
        #         while abs(self.Age[new_index] - self.Age[index]) > 10:
        #             new_index = np.random.choice(self.data_len)
        #         new_nifti_path = os.path.join(self.data_nifti_dir, self.data_path[new_index])
        #         new_nib_data = nib.load(new_nifti_path).get_fdata()
        #         x, y, z = new_nib_data.shape

        #         if self.transform is not None:
        #             new_nib_data = self.transform(new_nib_data)

        #         a, b, c = auto_select_box(new_nib_data)
        #         a = int(np.random.choice(list(range(max(a - 4, 191), min(a + 5, x - 1))), 1))
        #         b = int(np.random.choice(list(range(max(b - 4, 255), min(b + 5, y - 1))), 1))
        #         c = int(np.random.choice(list(range(max(c - 4, 223), min(c + 5, z - 1))), 1))
        #         new_nib_data = new_nib_data[a - 191:a + 1, b - 255:b + 1, c - 223:c + 1]

        #         if np.random.rand() < 0.5:
        #             nib_data = np.concatenate((nib_data[:96, :, :], new_nib_data[96:, :, :]), axis=0)
        #         else:
        #             nib_data = np.concatenate((new_nib_data[:96, :, :], nib_data[96:, :, :]), axis=0)
        #         label = torch.tensor((self.Age[index] + self.Age[new_index]) / 2, dtype=torch.float32)
        #     else:
        #         a, b, c = auto_select_box(nib_data)
        #         a = int(np.random.choice(list(range(max(a - 4, 191), min(a + 5, x - 1))), 1))
        #         b = int(np.random.choice(list(range(max(b - 4, 255), min(b + 5, y - 1))), 1))
        #         c = int(np.random.choice(list(range(max(c - 4, 223), min(c + 5, z - 1))), 1))
        #         nib_data = nib_data[a - 191:a + 1, b - 255:b + 1, c - 223:c + 1]

        #         label = torch.tensor(self.Age[index], dtype=torch.float32)

        #     p10 = np.percentile(nib_data, 10)
        #     p99 = np.percentile(nib_data, 99)
        #     nib_data = rescale_intensity(nib_data, in_range=(p10, p99), out_range=(0, 1))

        #     m = np.mean(nib_data, axis=(0, 1, 2))
        #     s = np.std(nib_data, axis=(0, 1, 2))
        #     nib_data = (nib_data - m) / s

        #     nib_data = torch.tensor(nib_data, dtype=torch.float32)
        # elif self.mode == 'valid':
        #     a, b, c = auto_select_box(nib_data)

        #     nib_data = nib_data[a - 191:a + 1, b - 255:b + 1, c - 223:c + 1]
        #     p10 = np.percentile(nib_data, 10)
        #     p99 = np.percentile(nib_data, 99)
        #     nib_data = rescale_intensity(nib_data, in_range=(p10, p99), out_range=(0, 1))
        #     m = np.mean(nib_data, axis=(0, 1, 2))
        #     s = np.std(nib_data, axis=(0, 1, 2))
        #     nib_data = (nib_data - m) / s

        #     nib_data = torch.tensor(nib_data, dtype=torch.float32)
        #     label = torch.tensor(self.Age[index], dtype=torch.float32)
        # else:
        #     print("mode must in [train, valid, 'test']")
        return nib_data.unsqueeze(0), label

    def __len__(self):
        return len(self.data_path)

def transforms(scale=None, angle=None, flip_prob=None):
    transform_list = []

    if scale is not None:
        transform_list.append(Scale(scale))
    if angle is not None:
        transform_list.append(Rotate(angle))
    if flip_prob is not None:
        transform_list.append(HorizontalFlip(flip_prob))
    # 实例化transforms对象，用Compose整合多个操作
    return Compose(transform_list)


# 自定义transform
class Scale(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        image = sample

        img_size = image.shape

        scale = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)

        image = rescale(
            image,
            scale,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )

        if scale < 1.0:
            diff1 = (img_size[0] - image.shape[0]) / 2.0
            diff2 = (img_size[1] - image.shape[1]) / 2.0
            diff3 = (img_size[2] - image.shape[2]) / 2.0

            padding = ((int(np.floor(diff1)), int(np.ceil(diff1))), (int(np.floor(diff2)), int(np.ceil(diff2))),
                       (int(np.floor(diff3)), int(np.ceil(diff3))))
            image = np.pad(image, padding, mode="constant", constant_values=0)
        else:
            x_min = (image.shape[0] - img_size[0]) // 2
            x_max = x_min + img_size[0]
            y_min = (image.shape[1] - img_size[1]) // 2
            y_max = y_min + img_size[1]
            z_min = (image.shape[2] - img_size[2]) // 2
            z_max = z_min + img_size[2]

            image = image[x_min:x_max, y_min:y_max, z_min:z_max]

        return image


class Rotate(object):

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image = sample

        angle = np.random.uniform(low=-self.angle, high=self.angle)
        image = rotate(image, angle, resize=False, preserve_range=True, mode="constant")

        return image


class HorizontalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image = sample

        if np.random.rand() > self.flip_prob:
            return image

        image = image[::-1, :, :]

        return image.copy()

def resdual_block(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3), stride=1, padding=1),
        nn.InstanceNorm3d(out_channel, affine=True),
        nn.ReLU(),
        nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 3), stride=1, padding=1),
        nn.InstanceNorm3d(out_channel, affine=True),
        nn.ReLU(),
        nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 3), stride=1, padding=1),
        nn.InstanceNorm3d(out_channel, affine=True),
        nn.ReLU()
    )


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=(7, 7, 7), stride=1, padding=2),
            nn.InstanceNorm3d(4, affine=True),
            nn.ReLU(),
            nn.Conv3d(4, 4, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.InstanceNorm3d(4, affine=True),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.InstanceNorm3d(16, affine=True),
            nn.ReLU(),
            nn.Conv3d(16, 16, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.InstanceNorm3d(16, affine=True),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2)
        )

        self.block_3_1 = resdual_block(16, 64)
        self.block_3_1_supple = nn.Conv3d(16, 64, kernel_size=1)
        self.block_3_2 = resdual_block(64, 64)
        self.block_3_pool = nn.MaxPool3d(2, stride=2)

        self.block_4_1 = resdual_block(64, 128)
        self.block_4_1_supple = nn.Conv3d(64, 128, kernel_size=1)
        self.block_4_2 = resdual_block(128, 128)
        self.block_4_3 = resdual_block(128, 128)
        self.block_4_pool = nn.MaxPool3d(2, stride=2)

        self.block_5_1 = resdual_block(128, 128)
        self.block_5_2 = resdual_block(128, 128)
        self.block_5_3 = resdual_block(128, 128)
        self.block_5_pool = nn.MaxPool3d(2, stride=2)

        self.block_6 = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),

                nn.Linear(128, 128),
                nn.Linear(128, 64),
                nn.Linear(64, 1)
            )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)

        x = self.block_3_1(x) + self.block_3_1_supple(x)
        x = self.block_3_2(x) + x
        x = self.block_3_pool(x)

        x = self.block_4_1(x) + self.block_4_1_supple(x)
        x = self.block_4_2(x) + x
        x = self.block_4_3(x) + x
        x = self.block_4_pool(x)

        x = self.block_5_1(x) + x
        x = self.block_5_2(x) + x
        x = self.block_5_3(x) + x
        x = self.block_5_pool(x)

        x = self.block_6(x)

        return x

def k_fold_cv(k, random_state=42):
    data_info_select = pd.read_csv(train_csv_path)
    data_info_select = shuffle(data_info_select, random_state=random_state)

    num_data = len(data_info_select)
    num_data_one_fold = int(np.ceil(num_data/k))
    for begin_idx in range(0, num_data, num_data_one_fold):
        val_path = list(data_info_select['data_path'][begin_idx:min(
            begin_idx+num_data_one_fold, num_data)])
        val_label = list(data_info_select['Patient_Age'][begin_idx:min(
            begin_idx+num_data_one_fold, num_data)])

        val_dataset = MRI_Data(val_path, val_label, mode='valid')

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        yield val_loader, val_path

device = torch.device("cuda:0")
model = Model()
criterion = nn.MSELoss()

for fold, (val_loader, val_path) in enumerate(k_fold_cv(5)):
    print("---->  fold: ", fold+1)
    model.load_state_dict(torch.load(
        '/hpc/data/home/bme/yubw/taotl/model8/model_save/fold_1_epoch_50.pth'))
    model.to(device)
    model.eval()
    dl = DeepLift(model)
    # ig = IntegratedGradients(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    for index, datas in enumerate(val_loader):
        data, label = datas
        label = label.unsqueeze(1)
        data = data.to(device)
        label = label.to(device)
        data.requires_grad = True
        optimizer.zero_grad()

        #deeplift
        attribution = dl.attribute(data)

        # ig
        # baseline = torch.zeros_like(data)
        # attribution = ig.attribute(data, baseline)

        original_data = nib.load(os.path.join(data_nifti_dir, val_path[index]))
        original_affine = original_data.affine
        original_data = original_data.get_fdata()
        # print(original_data.shape)
        # bound = auto_select_box(original_data)
        bound = (224,256,256)

        attribution = attribution[0].cpu().detach().numpy()[0]
        # print(attribution.shape)
        attribution = attribution/0.0001
        # 恢复原图大小和affine得到的attrbution
        # restored_attrbution = utils.restore_size(
        #     attribution, bound, nib_affine=original_affine)

        # show
        # utils.show_heatmap(attribution)
        # utils.show_mri_and_heatmap(fold, index,  original_data,
        #                         restored_attrbution.get_fdata())
        # utils.show_mri_and_heatmap_2(fold, index,  original_data,
        #                             restored_attrbution.get_fdata())
        
        # 存储热图，进行下一步配准
        dir_name = val_path[index].split('.')[0]
        print(os.path.join(
            heat_map_dir, dir_name, dir_name+'.nii.gz'))
        if not os.path.exists(os.path.join(heat_map_dir, dir_name)):
            os.mkdir(os.path.join(heat_map_dir, dir_name))
        nib.Nifti1Image(attribution, original_affine).to_filename(os.path.join(
            heat_map_dir, dir_name, dir_name+'.nii.gz'))
        # attribution.to_filename(os.path.join(
        #     heat_map_dir, dir_name, dir_name+'.nii.gz'))

        print('Finished index: ',index)

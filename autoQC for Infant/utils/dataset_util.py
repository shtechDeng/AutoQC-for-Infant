# encoding:utf-8
import os
import random

import torch
import numpy as np
from scipy.signal import convolve
from torch.utils.data import Dataset
from torchvision import transforms
from utils.load_util import file_loader, data_to_one
from utils.simulation_util import ZernikeWavefront, PsfGenerator3D
import nibabel as nib
from tqdm import tqdm
from collections import Counter
from skimage.exposure import rescale_intensity
from skimage.transform import rescale, rotate
from torchvision.transforms import Compose
from skimage.transform import resize


class MyDataSet1(Dataset):
    """
    one data to one label
    """
    def __init__(self, data_root, data_file, data_size, label_size, config):
        self.data_root = data_root
        self.data_size = data_size
        self.label_size = label_size
        self.loader = file_loader(config)

        datafn_list = []
        with open(data_file, 'r') as f:
            for line in f:
                temp = line.strip().split(' ')
                datafn_list.append((temp[0], temp[1]))
        self.datafn_list = datafn_list

    def __getitem__(self, index):
        datafn, labelfn = self.datafn_list[index]
        data = self.loader(os.path.join(self.data_root, datafn), 0)
        data = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((self.data_size[-2],
                                                      self.data_size[-1]))])(data)
        label = self.loader(os.path.join(self.data_root, labelfn), 1)
        label = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((self.label_size[-2],
                                                       self.label_size[-1]))])(label)

        if datafn.split('\\')[1:]:
            temp = datafn.split('\\')[1:]
            fn = temp[0] + '_' + temp[1]
        else:
            temp = datafn.split('/')[1:]
            fn = temp[0] + '_' + temp[1]

        return data, label, fn

    def __len__(self):
        return len(self.datafn_list)


class MyDataSet2(Dataset):
    """
    one data to two label
    """
    def __init__(self, data_root, data_file, data_size, label_size, otherLabel_size, config):
        self.data_root = data_root
        self.data_size = data_size
        self.label_size = label_size
        self.otherLabel_size = otherLabel_size
        self.loader = file_loader(config)

        datafn_list = []
        with open(data_file, 'r') as f:
            for line in f:
                temp = line.strip().split(' ')
                datafn_list.append((temp[0], temp[2], temp[1]))

        seed = 51
        random.seed(seed)
        random.shuffle(datafn_list)
        self.datafn_list = datafn_list[:6000]

    def __getitem__(self, index):
        datafn, labelfn, otherLabelfn = self.datafn_list[index]
        input_data = self.loader(os.path.join(self.data_root, datafn), 0)
        input_data = torch.from_numpy(input_data)
        input_data = transforms.Compose([transforms.Resize((self.data_size[-2],
                                                            self.data_size[-1]))])(input_data)
        label_pattern = self.loader(os.path.join(self.data_root, labelfn), 1)
        label_pattern = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((self.label_size[-2],
                                                               self.label_size[-1]))])(label_pattern)
        label_index = np.load(os.path.join(self.data_root, otherLabelfn), allow_pickle=True)
        label_index = torch.from_numpy(np.array(list(label_index.item().values())))

        if datafn.split('\\')[1:]:
            temp = datafn.split('\\')[1:]
            fn = temp[0] + '_' + temp[1]
        else:
            temp = datafn.split('/')[1:]
            fn = temp[0] + '_' + temp[1]

        return input_data.float(), [label_pattern.float(), label_index.float()], fn

    def __len__(self):
        return len(self.datafn_list)


class MyDataSet3(Dataset):
    """
    one data to two label one of which is simulation psf with aberration
    """
    def __init__(self, data_root, data_file, data_size, label_size, otherLabel_size, config):
        self.data_root = data_root
        self.data_size = data_size
        self.label_size = label_size
        self.aberration_size = otherLabel_size
        self.loader = file_loader(config)

        datafn_list = []
        with open(data_file, 'r') as f:
            for line in f:
                temp = line.strip().split(' ')
                datafn_list.append((temp[0], temp[1], temp[2]))
        self.datafn_list = datafn_list

        datafn_aberration_list = []
        for datafn in datafn_list:
            aberration_dict = np.load(os.path.join(data_root, datafn[-1]), allow_pickle=True)
            aberration_list = list(aberration_dict.item().items())
            for aberration in aberration_list:
                datafn_aberration_list.append([datafn[:-1], aberration])
        self.datafn_aberration_list = datafn_aberration_list

        self.pixel_size = (0.1, 0.0313, 0.0313)
        self.detection_na = 1.3
        self.wave_length = 0.488
        self.n = 1.518

    def __getitem__(self, index):
        datafn_labelfn, aberration_index = self.datafn_aberration_list[index]
        data = self.loader(os.path.join(self.data_root, datafn_labelfn[0]), 0)
        data = data * np.ones(self.data_size[1:], dtype=np.float32)

        # data = data.dot(np.ones(self.data_size[1:], dtype=np.float32))
        label = self.loader(os.path.join(self.data_root, datafn_labelfn[1]), 1)

        aberration_key = aberration_index[0]
        aberration_value = aberration_index[1]
        zwf = ZernikeWavefront(aberration_value, order='ansi')
        psf = PsfGenerator3D(psf_shape=self.data_size[1:],
                             units=self.pixel_size,
                             na_detection=self.detection_na,
                             lam_detection=self.wave_length,
                             n=self.n)

        aberration = np.float32(zwf.polynomial(self.aberration_size[-1], normed=True, outside=0))
        psf_zxy = np.float32(psf.incoherent_psf_intensity(zwf, normed=True))

        # psf_data = []
        # for idx in np.arange(self.data_size[1]):
        #     psf_data.append(convolve(data[idx], psf_zxy[idx], 'same'))

        # 空域卷积定理实现的严格操作顺序是，先进行傅立叶变换 fft，接着进行频域点乘，然后进行傅立叶反变换 ifft，最后进行平移 ifftshift
        psf_data = np.array(np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(data) * np.fft.fft2(psf_zxy)), axes=[-2, -1])), dtype=np.float32)

        psf_data = torch.from_numpy(np.asarray(psf_data))
        psf_data = transforms.Compose([transforms.Resize((self.data_size[-2],
                                                      self.data_size[-1]))])(psf_data)
        label = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((self.label_size[-2],
                                                       self.label_size[-1]))])(label)
        aberration = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((self.aberration_size[-2],
                                                            self.aberration_size[-1]))])(aberration)

        if datafn_labelfn[0].split('\\')[2:]:
            temp = datafn_labelfn[0].split('\\')[2:]
            fn = temp[0] + '_' f'{aberration_key[1]:6f}' + '_' + temp[1]
        else:
            temp = datafn_labelfn[0].split('/')[2:]
            fn = temp[0] + '_' f'{aberration_key[1]:6f}' + '_' + temp[1]

        return data, psf_data, [label, aberration, torch.tensor(list(aberration_value.values()), dtype=torch.float32)], fn

    def __len__(self):
        return len(self.datafn_aberration_list)


class MyDataSet4(Dataset):
    """
    one data to two label one of which is simulation psf with aberration
    """
    def __init__(self, data_root, data_file, data_size, label_size, otherLabel_size, config):
        self.data_root = data_root
        self.data_size = data_size
        self.label_size = label_size
        self.aberration_size = otherLabel_size
        self.loader = file_loader(config)

        datafn_list = []
        with open(data_file, 'r') as f:
            for line in f:
                temp = line.strip().split(' ')
                datafn_list.append((temp[0], temp[1], temp[2]))
        self.datafn_list = datafn_list

        datafn_aberration_list = []
        for datafn in datafn_list:
            aberration_dict = np.load(os.path.join(data_root, datafn[-1]), allow_pickle=True)
            aberration_list = list(aberration_dict.item().items())
            for aberration in aberration_list:
                datafn_aberration_list.append([datafn[:-1], aberration])
        self.datafn_aberration_list = datafn_aberration_list

        self.pixel_size = (0.1, 0.0313, 0.0313)
        self.detection_na = 1.3
        self.wave_length = 0.488
        self.n = 1.518

    def __getitem__(self, index):
        datafn_labelfn, aberration_index = self.datafn_aberration_list[index]
        data = self.loader(os.path.join(self.data_root, datafn_labelfn[0]), 0)
        data = data * np.ones(self.data_size[1:], dtype=np.float32)

        aberration_key = aberration_index[0]
        aberration_value = aberration_index[1]
        zwf = ZernikeWavefront(aberration_value, order='ansi')
        psf = PsfGenerator3D(psf_shape=self.data_size[1:],
                             units=self.pixel_size,
                             na_detection=self.detection_na,
                             lam_detection=self.wave_length,
                             n=self.n)

        aberration_pattern = np.float32(zwf.polynomial(self.aberration_size[-1], normed=True, outside=np.nan))
        aberration_min = np.nanmin(aberration_pattern, axis=(-2, -1), keepdims=True)
        aberration_max = np.nanmax(aberration_pattern, axis=(-2, -1), keepdims=True)
        aberration_pattern = (aberration_pattern - aberration_min) / (aberration_max - aberration_min)
        aberration_pattern[np.isnan(aberration_pattern)] = 0

        psf_zxy = np.float32(np.abs(psf.incoherent_psf(zwf, normed=True)) ** 2)
        psf_data = []
        for idx in np.arange(self.data_size[1]):
            psf_data.append(convolve(data[idx], psf_zxy[idx], 'same'))

        data = torch.from_numpy(np.asarray(psf_data))
        data = transforms.Compose([transforms.Resize((self.data_size[-2],
                                                      self.data_size[-1]))])(data)

        label_index = torch.tensor(list(aberration_value.values())).float()
        label_pattern = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((self.aberration_size[-2],
                                                               self.aberration_size[-1]))])(aberration_pattern)

        # input_data = torch.nn.AvgPool2d(8)(torch.tensor(psf_data)).numpy()
        # label_pattern_data = aberration_pattern
        # label_index_data = aberration_value

        if datafn_labelfn[0].split('\\')[2:]:
            temp = datafn_labelfn[0].split('\\')[2:]
            fn = temp[0] + '_' f'{aberration_key[1]:6f}' + '_' + temp[1]
        else:
            temp = datafn_labelfn[0].split('/')[2:]
            fn = temp[0] + '_' f'{aberration_key[1]:6f}' + '_' + temp[1]

        # path_input = r"A:\SR&AB\dataset\0826\input_data"
        # path_label_pattern = r"A:\SR&AB\dataset\0826\label_pattern_data"
        # path_label_index = r"A:\SR&AB\dataset\0826\label_index_data"
        # np.save(os.path.join(path_input, fn.replace(".tif", ".npy")), input_data)
        # np.save(os.path.join(path_label_pattern, fn.replace(".tif", ".npy")), label_pattern_data)
        # np.save(os.path.join(path_label_index, fn.replace(".tif", ".npy")), label_index_data)
        # return fn

        return data, [label_index, label_pattern], fn

    def __len__(self):
        return len(self.datafn_aberration_list)

class MRI_Data(Dataset):
    def __init__(
            self,
            data_path,
            mode,
            # transform = transforms.Compose([transforms.Resize(208, 300, 320)])
            transform = None
    ):
        self.mode = mode
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
        label = label.replace("\n","")
        nib_data = nib.load(nifti_path).get_fdata()
        if(len(nib_data.shape)) == 4:
            nib_data = np.squeeze(nib_data)
            nib_data = np.swapaxes(np.swapaxes(nib_data, 0, 2), 1, 2)
        
        nib_data = resize(nib_data, (208, 304, 320))

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
                label = torch.tensor(int(float(label)), dtype=torch.float32)

            else:
                # a, b, c = _select_box(nib_data)
                # a = int(np.random.choice(list(range(max(a - 4, 191), min(a + 5, x - 1))), 1))
                # b = int(np.random.choice(list(range(max(b - 4, 223), min(b + 5, y - 1))), 1))
                # c = int(np.random.choice(list(range(max(c - 4, 223), min(c + 5, z - 1))), 1))
                # nib_data = nib_data[a - 191:a + 1, b - 223:b + 1, c - 223:c + 1]
                # print(nib_data.shape)

                label = torch.tensor(int(float(label)), dtype=torch.float32)

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
        return nib_data.unsqueeze(0), label/2, nifti_path

    def __len__(self):
        return len(self.data_list)


def train_test_dataloader(config):
    trainset = MRI_Data(config["train_database"],"train")

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config['train_loader_batch'], shuffle=True,
                                              num_workers=0, pin_memory=False)

    testset = MRI_Data(config["test_database"],"test")

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=config['test_loader_batch'], shuffle=False,
                                             num_workers=0, pin_memory=False)

    return trainloader, testloader


def verify_dataloader(config):
    verifyset = MyDataSet4(data_root=config['database_root'], data_file=config['verify_database'],
                           data_size=config['data_size'], label_size=config['label_size'], otherLabel_size=config['otherLabel_size'],
                           config=config)

    verifyloader = torch.utils.data.DataLoader(verifyset, batch_size=config['verify_loader_batch'], shuffle=False,
                                               num_workers=0, pin_memory=True)

    return verifyloader


if __name__ == '__main__':
    import json
    with open(r'../config.json', 'r') as f1:
        config1 = json.load(f1)

    trainset1 = MRI_Data("../data/BCPpass_fail.txt","train" )

    trainloader1 = torch.utils.data.DataLoader(dataset=trainset1, batch_size=config1['train_loader_batch'], shuffle=False,
                                               num_workers=2, pin_memory=True)
    for idx, (data, target, nifti_path) in enumerate(trainloader1):
        shape = data.shape
        # if shape[2] != 208 or shape[3] != 300 or shape[4] != 320:
        print(shape)
        print(nifti_path[0]+"\n")














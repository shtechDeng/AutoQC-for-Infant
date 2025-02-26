import sys
import numpy as np
import scipy.io as sio
from PIL import Image
import cv2
import glob
import tifffile


def image_load(path):
    return cv2.imread(path)


def image_gray_load(path):
    return Image.open(path).convert('L')


def tif_load(path):
    return tifffile.imread(path)


def npy_load(path):
    return np.load(path, allow_pickle=True)


def mat_load(path):
    return sio.loadmat(path)


def count_means_stdevs(
        path='./*/*',
        data_size=(256, 256),
        pixel_limit=255
):
    data_path = glob(path)
    datas = []
    for i in range(len(data_path)):
        if os.path.splitext(data_path[i])[-1] in [".png", ".bmp", ".tif", ".jpg"]:
            data_temp = cv2.imdecode(np.fromfile(data_path[i], dtype=np.uint8), -1)
        else:
            data_temp = None
        assert len(data_temp.shape) == 3
        datas.append(np.expand_dims(cv2.resize(data_temp, data_size, interpolation=cv2.INTER_CUBIC), -1))

    datas = np.concatenate(datas, axis=-1)  # size [h w c b]
    if datas.shape[-2] == 1:
        datas = datas.repeat(3, -2)
    means = []
    stdevs = []

    for j in range(datas.shape[-2]):
        pixels = datas[..., j, :].ravel() / pixel_limit
        means.append([np.mean(pixels)])
        stdevs.append([np.std(pixels)])

    return means, stdevs


def data_to_normalize(inputs, mean, std, max_pixel_value=255.0):
    mean = np.asarray(mean, dtype=np.float32)
    assert len(mean.shape) == 2
    if len(inputs.shape) == 3:
        mean = mean[..., np.newaxis]
    elif len(inputs.shape) == 4:
        mean = mean[np.newaxis, ..., np.newaxis]
    mean *= max_pixel_value
    mean[mean == 0] = 1

    std = np.asarray(std, dtype=np.float32)
    assert len(std.shape) == 2
    if len(inputs.shape) == 3:
        std = std[..., np.newaxis]
    elif len(inputs.shape) == 4:
        std = std[np.newaxis, ..., np.newaxis]
    std *= max_pixel_value
    std[std == 0] = 1

    outputs = np.asarray(inputs, dtype=np.float32)
    outputs -= mean
    outputs /= std

    return outputs


def data_to_one(inputs):
    assert not np.any(np.isnan(inputs))
    data_min = np.min(inputs, axis=(-2, -1), keepdims=True)
    data_max = np.max(inputs, axis=(-2, -1), keepdims=True)
    outputs = (inputs - data_min) / (data_max - data_min)

    return outputs


class ImageLoader:
    def __init__(self, config):
        self.transform_mode = config['transform_mode']
        if self.transform_mode == "one":
            one_input = config['one_input_range']
            one_label = config['one_label_range']
            self.min_max_list = [one_input, one_label]
        elif self.transform_mode == "norm":
            norm_input = config['norm_input_mean_std']
            if len(norm_input) == 0 or None in norm_input:
                input_means, input_stds = batch_means_stdevs(path=f"{config['database_root']}/data/*/*.png",
                                                             data_size=config['data_size'][2:], pixel_limit=255)
            else:
                input_means, input_stds = norm_input[0], norm_input[1]
            norm_label = config['norm_label_mean_std']
            if len(norm_label) == 0 or None in norm_label:
                label_means, label_stds = batch_means_stdevs(path=f"{config['database_root']}/label/*/*.png",
                                                             data_size=config['label_size'][2:], pixel_limit=255)
            else:
                label_means, label_stds = norm_label[0], norm_label[1]
            self.mean_std_list = [(input_means, input_stds), (label_means, label_stds)]
        else:
            print("data transform not change anything")

    def data_transform(self, inputs, index):
        if self.transform_mode == "one":
            min_max = self.min_max_list[index]
            if None in min_max:
                outputs = data_to_one(inputs)
            elif np.all([np.isscalar(elem) for elem in min_max]):
                assert min_max[0] < min_max[1]
                outputs = (inputs - min_max[0]) / (min_max[1] - min_max[0])
            else:
                outputs = inputs

        elif self.transform_mode == "norm":
            outputs = data_to_normalize(inputs, *self.mean_std_list[index])
        else:
            outputs = inputs

        return outputs

    def __call__(self, path, index, **kwargs):
        result = image_load(path)
        return self.data_transform(np.asarray(result, dtype=np.float32), index)


class TifLoader:
    def __init__(self, config):
        self.transform_mode = config['transform_mode']
        if self.transform_mode == "one":
            one_input = config['one_input_range']
            one_label = config['one_label_range']
            self.min_max_list = [one_input, one_label]
        elif self.transform_mode == "norm":
            norm_input = config['norm_input_mean_std']
            if len(norm_input) == 0 or None in norm_input:
                input_means, input_stds = batch_means_stdevs(path=f"{config['database_root']}/data/*/*.png",
                                                             data_size=config['data_size'][2:], pixel_limit=255)
            else:
                input_means, input_stds = norm_input[0], norm_input[1]
            norm_label = config['norm_label_mean_std']
            if len(norm_label) == 0 or None in norm_label:
                label_means, label_stds = batch_means_stdevs(path=f"{config['database_root']}/label/*/*.png",
                                                             data_size=config['label_size'][2:], pixel_limit=255)
            else:
                label_means, label_stds = norm_label[0], norm_label[1]
            self.mean_std_list = [(input_means, input_stds), (label_means, label_stds)]
        else:
            print("data transform not change anything")

    def data_transform(self, inputs, index):
        if self.transform_mode == "one":
            min_max = self.min_max_list[index]
            if None in min_max:
                outputs = data_to_one(inputs)
            elif np.all([np.isscalar(elem) for elem in min_max]):
                assert min_max[0] < min_max[1]
                outputs = (inputs - min_max[0]) / (min_max[1] - min_max[0])
            else:
                outputs = inputs

        elif self.transform_mode == "norm":
            outputs = data_to_normalize(inputs, *self.mean_std_list[index])
        else:
            outputs = inputs

        return outputs

    def __call__(self, path, index, **kwargs):
        result = tif_load(path)
        return self.data_transform(np.asarray(result, dtype=np.float32), index)


class NpyLoader:
    def __init__(self, config):
        self.transform_mode = config['transform_mode']
        if self.transform_mode == "one":
            one_input = config['one_input_range']
            one_label = config['one_label_range']
            self.min_max_list = [one_input, one_label]
        elif self.transform_mode == "norm":
            norm_input = config['norm_input_mean_std']
            if len(norm_input) == 0 or None in norm_input:
                input_means, input_stds = batch_means_stdevs(path=f"{config['database_root']}/data/*/*.png",
                                                             data_size=config['data_size'][2:], pixel_limit=255)
            else:
                input_means, input_stds = norm_input[0], norm_input[1]
            norm_label = config['norm_label_mean_std']
            if len(norm_label) == 0 or None in norm_label:
                label_means, label_stds = batch_means_stdevs(path=f"{config['database_root']}/label/*/*.png",
                                                             data_size=config['label_size'][2:], pixel_limit=255)
            else:
                label_means, label_stds = norm_label[0], norm_label[1]
            self.mean_std_list = [(input_means, input_stds), (label_means, label_stds)]
        else:
            print("data transform not change anything")

    def data_transform(self, inputs, index):
        if self.transform_mode == "one":
            min_max = self.min_max_list[index]
            if None in min_max:
                outputs = data_to_one(inputs)
            elif np.all([np.isscalar(elem) for elem in min_max]):
                assert min_max[0] < min_max[1]
                outputs = (inputs - min_max[0]) / (min_max[1] - min_max[0])
            else:
                outputs = inputs

        elif self.transform_mode == "norm":
            outputs = data_to_normalize(inputs, *self.mean_std_list[index])
        else:
            outputs = inputs

        return outputs

    def __call__(self, path, index, **kwargs):
        result = npy_load(path)
        return self.data_transform(np.asarray(result, dtype=np.float32), index)


class MatLoader:
    def __init__(self, config):
        self.transform_mode = config['transform_mode']
        if self.transform_mode == "one":
            one_input = config['one_input_range']
            one_label = config['one_label_range']
            self.min_max_list = [one_input, one_label]
        elif self.transform_mode == "norm":
            norm_input = config['norm_input_mean_std']
            if len(norm_input) == 0 or None in norm_input:
                input_means, input_stds = batch_means_stdevs(path=f"{config['database_root']}/data/*/*.png",
                                                             data_size=config['data_size'][2:], pixel_limit=255)
            else:
                input_means, input_stds = norm_input[0], norm_input[1]
            norm_label = config['norm_label_mean_std']
            if len(norm_label) == 0 or None in norm_label:
                label_means, label_stds = batch_means_stdevs(path=f"{config['database_root']}/label/*/*.png",
                                                             data_size=config['label_size'][2:], pixel_limit=255)
            else:
                label_means, label_stds = norm_label[0], norm_label[1]
            self.mean_std_list = [(input_means, input_stds), (label_means, label_stds)]
        else:
            print("data transform not change anything")

    def data_transform(self, inputs, index):
        if self.transform_mode == "one":
            min_max = self.min_max_list[index]
            if None in min_max:
                outputs = data_to_one(inputs)
            elif np.all([np.isscalar(elem) for elem in min_max]):
                assert min_max[0] < min_max[1]
                outputs = (inputs - min_max[0]) / (min_max[1] - min_max[0])
            else:
                outputs = inputs

        elif self.transform_mode == "norm":
            outputs = data_to_normalize(inputs, *self.mean_std_list[index])
        else:
            outputs = inputs

        return outputs

    def __call__(self, path, index, **kwargs):
        result = mat_load(path)
        for item in result.values():
            if isinstance(item, np.ndarray):
                return self.data_transform(np.asarray(item, dtype=np.float32), index)


def file_loader(config):
    extension = config['data_ext']
    if extension in ['.png', '.bmp', '.jpg', '.jpeg', '.raw']:
        return ImageLoader(config)
    elif extension in ['.tif', '.tiff']:
        return TifLoader(config)
    elif extension == '.npy':
        return NpyLoader(config)
    elif extension == '.mat':
        return MatLoader(config)
    else:
        print(f"当前数据格式{extension}无法读取")
        sys.exit()

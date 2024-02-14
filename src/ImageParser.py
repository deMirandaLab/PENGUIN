from apeer_ometiff_library import io
# import tensorflow as tf
import numpy as np
from src.file_specs import FileSpecifics

# class ImageParser():

# def parse_image(img_path: str) -> dict:
def parse_image(img_path: str) -> np.array:
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    CHANNELS
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    np.array
        array of image
    """
    try:
        (img_apeer, omexml) = io.read_ometiff(img_path)
        if img_path.__contains__('MB'):#  (1, 50, 1, 1, 494, 464) instead of   (1, 1, 52, 586, 617)
            img = img_apeer[0, :, 0, 0, :, :]
        else:
            img = img_apeer[0, 0, :, :, :]
        img = np.moveaxis(img, 0, -1)
        # print(img.shape)
            # return {'image': img, 'img_meta': omexml, 'filename': img_path}
        return img
    except Exception as e:
        print('did not read image', img_path)
        print(e)
        return None

from tifffile import tifffile
def parse_image_pages(img_path):
    """
    For tiff files with ages and not fullstack
    :param img_path:
    :return:
    """
    im = []
    with tifffile.TiffFile(img_path) as tif:
        for page in tif.pages:
            image = page.asarray()
            im.append(image)
    im = np.array(im)
    img = np.moveaxis(im, 0, -1)
    return img

def parse_image_pages_namesCH(img_path):
    """
    For tiff files with ages and not fullstack
    :param img_path:
    :return:
    """
    im = []
    ch_names = []
    with tifffile.TiffFile(img_path) as tif:
        for page in tif.pages:
            image = page.asarray()
            im.append(image)
            if 'PageName' in page.tags:
                ch_names.append(page.tags['PageName'].value)
            elif 'ImageDescription' in page.tags:
                ch_names.append(page.tags['ImageDescription'].value)
            else:
                pass

    im = np.array(im)
    img = np.moveaxis(im, 0, -1)
    return img, ch_names

def parse_image_with_meta(img_path: str) -> dict:
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    CHANNELS
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """
    img_apeer, omexml = io.read_ometiff(img_path)
    ch_names = []
    if img_path.__contains__('MB'):#  (1, 50, 1, 1, 494, 464) instead of   (1, 1, 52, 586, 617)
        img = img_apeer[0, :, 0, 0, :, :]
    else:
        img = img_apeer[0, 0, :, :, :]
    img = np.moveaxis(img, 0, -1)  # 'image': img,
    return {'img_meta': omexml, 'filename': img_path, 'shape_img': img.shape}


def parse_image_with_ch_name(img_path: str):
    """
    """
    img_apeer, omexml = io.read_ometiff(img_path)
    if img_path.__contains__('MB'):#  (1, 50, 1, 1, 494, 464) instead of   (1, 1, 52, 586, 617)
        img = img_apeer[0, :, 0, 0, :, :]
    else:
        img = img_apeer[0, 0, :, :, :]
    img = np.moveaxis(img, 0, -1)  # 'image': img,

    return img, omexml


def simple_reduce_channels(img: np.array, channeltokeep: list) -> np.array:
    """
    SELECT UNTIL THAT CHANNEL
    Parameters
    ----------
    img
    channels

    Returns
    -------

    """
    new_img = img[:, :, channeltokeep]
    return new_img


def simple_select_one_channel(img: np.array, channels: int) -> np.array:
    """
    SELECT UNTIL THAT CHANNEL
    Parameters
    ----------
    img
    channels

    Returns
    -------

    """
    new_img = img[:, :, channels]
    new_img = new_img.reshape(new_img.shape[0], new_img.shape[1], 1)
    return new_img


def remove_outliers(img: np.array, up_limit=99, down_limit=1) -> np.array:
    # To facilitate pixel annotation, the images are sharpened.
    # More specifically, outliers are removed through saturation of all pixels with
    # values lower than the 1st and higher than the 99th percentile.
    # Means that per each image and per each channel you find the values of the 1st and 99th percentile
    # and all the values below 1st percentile and above 99th percentile are set o zero right?
    imOutlier = img
    #     per channel
    for i in range(img.shape[2]):
        print(i)
        ch = img[:, :, i]  # imOutlier[:, :, i]= np.log(img[:, :, i]+0.5).round(4)
        p_99 = np.percentile(ch, up_limit)  # return 50th percentile, e.g median.
        print(p_99)
        p_01 = np.percentile(ch, down_limit)  # return 50th percentile, e.g median.
        print(p_01)
        # np.where Where True, yield x, otherwise yield y

        ch = np.where(ch > p_99, p_99, ch) # instead of substitube by 0
        ch = np.where(ch < p_01, 0, ch)
        imOutlier[:, :, i] = ch

    n_pixels_changed = np.sum(imOutlier != img)
    # n_pixels_changed = sum(map(lambda x, y: bool(x - y), imOutlier, img))
    # print('set {} pixels to zero (above {} and below {} percentile threshold per channel out of {}'.
    #       format(n_pixels_changed, up_limit,down_limit, img.shape[0]*img.shape[1]*img.shape[2]))
    # print((n_pixels_changed/(img.shape[0]*img.shape[1]*img.shape[2]))*100, 'pixels changed')
    return imOutlier


def normalize_by_channel(img: np.array) -> np.array:
    # Assuming you're working with image data of shape (W, H, 3), you should probably ' \
    #             'normalize over each channel (axis=2) separately, as mentioned in the other answer.
    # keepdims makes the result shape (1, 1, 3) instead of (3,). This doesn't matter here, but
    # would matter if you wanted to normalize over a different axis.
    v = img
    v_min = v.min(axis=(0, 1), keepdims=True)
    v_max = v.max(axis=(0, 1), keepdims=True)
    # print(v_min)
    # print(v_max)
    new_img = (v - v_min) / (v_max - v_min)
    # new_img = new_img * (v_max - v_min) + v_min
    # new_img = (v_max - v_min)/(v_max - v_min)*(v-v_max)+v_max
    return new_img


# get maximum and minimum per channel

def get_maximum_minimum_per_channel_over_dataset(dataset):
    vmin = []
    vmax = []
    for v in dataset:
        v_min = v.min(axis=(0, 1), keepdims=True)  # minimum value per channel in that image
        v_max = v.max(axis=(0, 1), keepdims=True)
        vmin.append(v_min)
        vmax.append(v_max)
    #     print(v_min)
    #     print(v_max)
    # print(len(vmin))
    # print(vmax)
    v_min = np.array(vmin).min(axis=(0, 1), keepdims=True)[0]
    v_max = np.array(vmax).max(axis=(0, 1), keepdims=True)[0]
    return v_min, v_max


def get_mean_std_per_channel_over_dataset(dataset):
    mean = []
    std = []
    for v in dataset:
        v_mean = v.mean(axis=(0, 1), keepdims=True)  # minimum value per channel in that image
        v_std = v.max(axis=(0, 1), keepdims=True)
        mean.append(v_mean)
        std.append(v_std)
    v_mean = np.array(mean).min(axis=(0, 1), keepdims=True)[0]
    v_std = np.array(std).std(axis=(0, 1), keepdims=True)[0]
    return v_mean, v_std

def normalize_dataset_channel(data: np.array):
    data_min = np.min(data, axis=(1,2), keepdims=True)
    data_max = np.max(data, axis=(1,2), keepdims=True)

    scaled_data = (data - data_min) / (data_max - data_min)
    return scaled_data


def normalize_by_channel_based_on_dataset(img: np.array, v_min, v_max) -> np.array:
    # Assuming you're working with image data of shape (W, H, 3), you should probably ' \
    #             'normalize over each channel (axis=2) separately, as mentioned in the other answer.
    # keepdims makes the result shape (1, 1, 3) instead of (3,). This doesn't matter here, but
    # would matter if you wanted to normalize over a different axis.
    v = img
    new_img = (v - v_min) / (v_max - v_min)
    # new_img = v / v_max
    return new_img

# todo   https://github.com/BodenmillerGroup/ImcPluginsCP/blob/a53bb7e1dea60b859d57677ea9a15281fa84d493/plugins/smoothmultichannel.py#L340
# important
# https://github.com/BodenmillerGroup/ImcSegmentationPipeline/tree/56ce18cfa570770eba169c7a3fb02ac492cc6d4b
def filter_hot_pixelsBodenmiller(img: np.ndarray, thres: float) -> np.ndarray:
    # https://github.com/BodenmillerGroup/ImcSegmentationPipeline/blob/56ce18cfa570770eba169c7a3fb02ac492cc6d4b/src/imcsegpipe/utils.py#L10
    from scipy.ndimage import maximum_filter
    kernel = np.ones((1, 3, 3), dtype=bool)
    kernel[0, 1, 1] = False
    max_neighbor_img = maximum_filter(img, footprint=kernel, mode="mirror")
    return np.where(img - max_neighbor_img > thres, max_neighbor_img, img)

def filter_hot_pixelsMean(img: np.ndarray) -> np.ndarray:
    # https://github.com/BodenmillerGroup/ImcSegmentationPipeline/blob/56ce18cfa570770eba169c7a3fb02ac492cc6d4b/src/imcsegpipe/utils.py#L10
    from skimage.filters.rank import geometric_mean
    kernel = np.ones((1, 3, 3), dtype=bool)
    kernel[0, 1, 1] = False
    mean_neighbor_img = geometric_mean(img, footprint=kernel)
    return mean_neighbor_img


def get_mean_std_per_channel(dataset):
    means = dataset.mean(axis=(0, 1), keepdims=True)  # Take the mean over the N,H,W axes
    print(means.shape)  # => will evaluate to (C,)
    std = dataset.mean(axis=(0, 1), keepdims=True)
    print(std.shape)


def standardize_by_channel(dataset):
    x = dataset
    z = (x - x.mean(axis=(0, 1), keepdims=True)) / x.std(axis=(0, 1), keepdims=True)
    return z


def logaritmic_image(img: np.array) -> np.array:
    imLog = img
    for i in range(img.shape[2]):
        imLog[:, :, i] = np.log(img[:, :, i]).round(4)
    return imLog


def logaritmic_image2(img: np.array) -> np.array:
    imLog = img
    for i in range(img.shape[2]):
        imLog[:, :, i] = np.log(img[:, :, i] + 0.5).round(4)
    return imLog


def mibi_noise(img: np.array) -> np.array:
    # For all other  channels, positive  pixels in the background channel  were subtracted
    # by two counts.Following the subtraction, negative values were converted to zeros
    imLog = img - 2
    result = np.where(imLog < 0, 0, imLog)
    print(result.shape)
    return result


# # check better ways to resize???
# def resize_dataset(image, INP_SIZE):
#     # image = tf.image.resize(image, (INP_SIZE[0], INP_SIZE[1]))
#
#     image = tf.image.resize_with_pad(image, INP_SIZE[0], INP_SIZE[1])
#     # https://www.tensorflow.org/api_docs/python/tf/image/resize
#
#     return image


def average_channel_per_pixel(img: np.array) -> np.array:
    pass


# see channels names. GEt ImageParser as class to store meta and stuff


def reshape_img_flatdim_channels(img, INP_SIZE, channels=52):
    new_img = np.array(img)
    new_img = new_img.reshape((INP_SIZE[0] * INP_SIZE[1], channels))
    return new_img


def reshape_img_flatdim(img):
    new_img = np.array(img)
    new_img = new_img.flatten()
    return new_img

def reduce_size_by_max_pool():
    pass


# def get_max_values(image):
#     m = max(image.flatten())
#     return m
#
# maximum = list(map(get_max_values, result))
# print(maximum)
# print(max(maximum))
#
#

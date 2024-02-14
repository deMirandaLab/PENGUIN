
import os
import numpy as np
import src.ImagePreprocessFilters as IPrep
import src.ImageParser as IP
from matplotlib import pyplot as plt
from IPython.display import set_matplotlib_formats
from glob import glob
import random
from IPython.display import display
from ipywidgets import interact
import ipywidgets as widgets
import skimage
import pathlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px




def set_params(res=300):
    plt.rcParams['figure.dpi'] = res
    plt.rcParams['savefig.dpi'] = res



def preprocess_images(files, up_limit = 99, down_limit = 1):
    # imgs can be one or muliple channels
    # PARSE images to get numpy arrays into shape (, ,n_channels)
    images = map(IP.parse_image, files)
    images = list(images)
    # PERCENTILE SATURATION OUTLIERS
    imgsOut = map(lambda p: IPrep.remove_outliers(p, up_limit, down_limit), images)
    # NORMALIZE PER CHANNEL with function from OpenCV
    imgs_norm = map(IPrep.normalize_channel_cv2_minmax, imgsOut)
    imgs_norm = list(imgs_norm)  # to be used more again
    return images, imgs_norm

def preprocess_images_pages(files, up_limit = 99, down_limit = 1):
    # imgs can be one or muliple channels
    # PARSE images to get numpy arrays into shape (, ,n_channels)
    images = map(IP.parse_image_pages, files)
    images = list(images)

    # PERCENTILE SATURATION OUTLIERS
    imgsOut = map(lambda p: IPrep.remove_outliers(p, up_limit, down_limit), images)
    # NORMALIZE PER CHANNEL with function from OpenCV
    imgs_norm = map(IPrep.normalize_channel_cv2_minmax, imgsOut)
    imgs_norm = list(imgs_norm)  # to be used more again
    return images, imgs_norm


# PERCENTILE FILTERS PER CHANNEL
def calculate_per_channel_percentile(images_channel,PERCENTILE):
    imgs_filtered = map(lambda p: IPrep.percentile_filter(p, window_size=3, percentile=PERCENTILE, transf_bool=True), images_channel)
    imgs_filtered = list(imgs_filtered)
    return imgs_filtered

def calculate_per_channel_consecutivepercentile(images_channel,PERCENTILE):
    imgs_filtered = map(lambda p: IPrep.percentile_filter(p, window_size=3, percentile=PERCENTILE, transf_bool=True), images_channel)
    imgs_filtered = map(lambda p: IPrep.percentile_filter(p, window_size=3, percentile=PERCENTILE, transf_bool=True), imgs_filtered)
    imgs_filtered = list(imgs_filtered)
    return imgs_filtered

def plot_one_channel(images, cmap = plt.cm.gray):
    for idx in range(len(images)):
        if images[0].shape[0]==1:
            images = [images[i][0] for i in range(len(images))]

        plt.imshow(images[idx][...], cmap = cmap)
        plt.show()

def plot_one_channel_side_by_side(images, columns=1, figsize=(20,20), cmap = plt.cm.gray):
    rows = len(images)//columns
    fig, axs = plt.subplots(rows, columns, figsize=figsize)
    if images[0].shape[0]==1:
        images = [images[i][0] for i in range(len(images))]

    for i, filename in enumerate(images):
        axs.flat[i].imshow(filename, cmap=cmap)
        axs.flat[i].axis('off')
    plt.tight_layout()
    plt.show()



def plot_compare_images(images1, images2, high_contrast, cmap = plt.cm.gray):
    if len(images1) == 1:
        fig, axs = plt.subplots(1, 2, figsize=(50, 50))
        axs = [axs]  # Convert to list to handle single-row case
    else:
        figsize = (50, 50 * len(images1))
        fig, axs = plt.subplots(len(images1), 2, figsize=figsize)
        plt.subplots_adjust(hspace=0.0)  # Adjust the vertical space between rows


    # if images1[0].shape[0]==1:
    #     images1 = [images1[i][0] for i in range(len(images1))]
    # if images2[0].shape[0]==1:
    #     images2 = [images2[i][0] for i in range(len(images2))]

    # fig, axs = plt.subplots(len(images1), 2, figsize = figsize)
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        if high_contrast == 'True':
            axs[i][0].imshow(image1, cmap=cmap, vmin=0, vmax=1)
            axs[i][0].axis('off')
            axs[i][1].imshow(image2, cmap=cmap, vmin=0, vmax=1)
            axs[i][1].axis('off')
        else:
            axs[i][0].imshow(image1, cmap=cmap)
            axs[i][0].axis('off')
            axs[i][1].imshow(image2, cmap=cmap)
            axs[i][1].axis('off')
    plt.tight_layout()
    plt.show()




def plot_compare_images_plotly(images1, images2, high_contrast, cmap='gray'):
    # Create subplots with two columns
    # fig = make_subplots(rows=1, cols=2, subplot_titles=[f'Image {i}' for i in range(1, len(images1) + 1)])
    fig = None
    # Add traces for each pair of images
    for i, (image1, image2) in enumerate(zip(images1, images2), start=1):
        # to imporve readibility
        image1 = np.squeeze(image1)
        min_val = np.min(image1)
        max_val = np.max(image1)
        image1 = (image1 - min_val) / (max_val - min_val)

        img_sequence = [np.squeeze(image1), np.squeeze(image2)]
        # if high_contrast == 'True':
        fig = px.imshow(np.array(img_sequence), facet_col=0, binary_string=True, zmin=0, zmax=1) #
        # else:
        #     fig = px.imshow(np.array(img_sequence), facet_col=0, binary_string=True)  #
        fig.update_layout(width=1000, height=1000)
        fig.show()
    # fig.update_layout(width=800, height=600 * len(images1), showlegend=False)

    # Show the plot



# https://datacarpentry.org/image-processing/05-creating-histograms/
# Do histogram of a cohort per channel
def histogram_one_channel_one_img(img):
    histogram, bin_edges = np.histogram(img, bins=256, range=(0, 1))
    # configure and draw the histogram figure
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.xlim([0.0, 1.0])  # <- named arguments do not work here

    plt.plot(bin_edges[0:-1], histogram)  # <- or here


def histogram_all_channel_one_img(img):
    from random import randint
    CH = img.shape[2]
    # tuple to select colors of each channel line
    colors = []
    for i in range(CH):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    # create the histogram plot, with three lines, one for
    # each color
    plt.figure()
    for channel_id, color in enumerate(colors):
        histogram, bin_edges = np.histogram(
            img[:, :, channel_id], bins=256, range=(0, 1)
        )
        plt.plot(bin_edges[0:-1], histogram, color=color)
#     plt.hist(img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k') #calculating histogram


def histogram_one_channel_all_img(images,linewidth=0.5):

    fig, ax = plt.subplots(figsize=(15, 7))
    for image in images:
        hist, bin_edges = np.histogram(image, bins=100, range=(0, 1))
        ax.plot(bin_edges[0:-1], hist, linewidth=linewidth)
    plt.ylim([0.0,75000])
    plt.show()

def calculate_psnr_snr(image1_true, image2_test):
    # Assumes image1 and image2 are numpy arrays with the same shape and dtype
    # do this by channel
    mse = np.mean((image1_true - image2_test) ** 2, axis=(0, 1))
    snr = np.mean(image1_true ** 2, axis=(0, 1)) / mse
    psnr = 10 * np.log10(np.amax(image1_true) ** 2 / mse)
    print(psnr)
    return psnr


def plot_psnr(imgs_channel,norm_imgs_channel, imgs_filtered):
    psnr = list(map(lambda p, f: skimage.metrics.peak_signal_noise_ratio(p, f, data_range = 1),
                    norm_imgs_channel, imgs_filtered))
    print(psnr)
    print('original', np.max(imgs_channel[0]), np.min(imgs_channel[0]))
    print('norm', np.max(norm_imgs_channel[0]), np.min(norm_imgs_channel[0]))
    print('filtered', np.max(imgs_filtered[0]), np.min(imgs_filtered[0]))
    plt.plot(psnr)


def calculus_fun(files_channel, PERCENTILE, TH, sample_images,sample_images_number):
    # defining the sample images. It will only run on the number of images
    if sample_images == 'random':
        randomlist = random.sample(range(0, len(files_channel)),sample_images_number)
        files_channel = [files_channel[i] for i in randomlist]

    elif sample_images == 'top':
        files_channel = files_channel[:sample_images_number]

    elif sample_images == 'bottom':
        files_channel = files_channel[-sample_images_number:]
    else: # all
        files_channel = files_channel


    images_original, imgs_norm = preprocess_images(files_channel)
    imgs_channel = [images_original[i] for i in range(len(images_original))]
    norm_imgs_channel = [imgs_norm[i] for i in range(len(imgs_norm))]

    # thresholding
    if isinstance(TH, float):
        imgs_filtered = list(map(lambda p: IPrep.out_ratio2(p, th=TH), norm_imgs_channel))

    elif TH is not None:
        if TH == 'otsu':
            imgs_filtered = list(map(lambda p: IPrep.th_otsu(p), norm_imgs_channel))
        elif TH == 'isodata':
            imgs_filtered = list(map(lambda p: IPrep.th_isodata(p), norm_imgs_channel))
        elif TH == 'Li':
            imgs_filtered = list(map(lambda p: IPrep.th_li(p), norm_imgs_channel))
        elif TH == 'Yen':
            imgs_filtered = list(map(lambda p: IPrep.th_yen(p), norm_imgs_channel))
        elif TH =='triangle':
            imgs_filtered = list(map(lambda p: IPrep.th_triangle(p), norm_imgs_channel))
        elif TH =='mean':
            imgs_filtered = list(map(lambda p: IPrep.th_mean(p), norm_imgs_channel))
        elif TH == 'local':
            imgs_filtered = list(map(lambda p: IPrep.th_local(p, block_size=3, method='gaussian'), norm_imgs_channel))
    else:
        imgs_filtered = norm_imgs_channel


    if PERCENTILE is None:
        imgs_filtered = imgs_filtered
    elif PERCENTILE == 'p50consecutive':
        imgs_filtered = calculate_per_channel_consecutivepercentile(imgs_filtered,PERCENTILE=50)
    else:
        imgs_filtered = calculate_per_channel_percentile(imgs_filtered,PERCENTILE)

    return imgs_channel,norm_imgs_channel,imgs_filtered




def defining_files(files_channel,sample_images_number, sample_images):
    # defining the sample images. It will only run on the number of images
    if sample_images == 'random':
        randomlist = random.sample(range(0, len(files_channel)),sample_images_number)
        files_channel = [files_channel[i] for i in randomlist]

    elif sample_images == 'top':
        files_channel = files_channel[:sample_images_number]

    elif sample_images == 'bottom':
        files_channel = files_channel[-sample_images_number:]
    else: # all
        files_channel = files_channel
    return files_channel

def calculus_multitiff(imgs_ch, PERCENTILE, TH):
    norm_imgs_channel = imgs_ch
    # thresholding
    if isinstance(TH, float):
        imgs_filtered = list(map(lambda p: IPrep.out_ratio2(p, th=TH), norm_imgs_channel))

    elif TH is not None:
        if TH == 'otsu':
            imgs_filtered = list(map(lambda p: IPrep.th_otsu(p), norm_imgs_channel))
        elif TH == 'isodata':
            imgs_filtered = list(map(lambda p: IPrep.th_isodata(p), norm_imgs_channel))
        elif TH == 'Li':
            imgs_filtered = list(map(lambda p: IPrep.th_li(p), norm_imgs_channel))
        elif TH == 'Yen':
            imgs_filtered = list(map(lambda p: IPrep.th_yen(p), norm_imgs_channel))
        elif TH =='triangle':
            imgs_filtered = list(map(lambda p: IPrep.th_triangle(p), norm_imgs_channel))
        elif TH =='mean':
            imgs_filtered = list(map(lambda p: IPrep.th_mean(p), norm_imgs_channel))
        elif TH == 'local':
            imgs_filtered = list(map(lambda p: IPrep.th_local(p, block_size=3, method='gaussian'), norm_imgs_channel))
    else:
        imgs_filtered = norm_imgs_channel


    if PERCENTILE is None:
        imgs_filtered = imgs_filtered
    elif PERCENTILE == 'p50consecutive':
        imgs_filtered = calculate_per_channel_consecutivepercentile(imgs_filtered,PERCENTILE=50)
    else:
        imgs_filtered = calculate_per_channel_percentile(imgs_filtered,PERCENTILE)

    return norm_imgs_channel,imgs_filtered




def calculus_multitiff_lists(img, th_list, p_list):
    filtered_img = np.empty(img.shape)
    for ch in range(img.shape[2]):
        img_ch = img[:,:,ch]

        # thresholding
        TH = th_list[ch]

        new_img = img_ch
        if isinstance(TH, float):
            new_img = np.where(img_ch >= TH, img_ch, 0)
        elif TH is None:
            new_img = img_ch
        elif TH is not None:
            if TH == 'otsu':
                new_img = IPrep.th_otsu(img_ch)
            elif TH == 'isodata':
                new_img = IPrep.th_isodata(img_ch)
            elif TH == 'Li':
                new_img = IPrep.th_li(img_ch)
            elif TH == 'Yen':
                new_img = IPrep.th_yen(img_ch)
            elif TH =='triangle':
                new_img = IPrep.th_triangle(img_ch)
            elif TH =='mean':
                new_img = IPrep.th_mean(img_ch)
            elif TH == 'local':
                new_img = IPrep.th_local(img_ch, block_size=3, method='gaussian')
        else:
            print('threshold not recognized, passing')
            new_img = img_ch

        filtered_img[:,:,ch] = new_img

        img_ch = filtered_img[:,:,ch]
        # percentile
        PERCENTILE = p_list[ch]

        if PERCENTILE is None:
            new_img = img_ch
        elif PERCENTILE == 'p50consecutive':
            img_ch = img_ch[...,np.newaxis]
            new_img = IPrep.percentile_filter(img_ch,PERCENTILE=50)
            new_img = IPrep.percentile_filter(new_img,PERCENTILE=50)
            new_img = new_img.squeeze()
        else:
            img_ch = img_ch[...,np.newaxis]
            new_img = IPrep.percentile_filter(img_ch, window_size=3, percentile=PERCENTILE, transf_bool=True)
            new_img = new_img.squeeze()
        filtered_img[:,:,ch] = new_img

    return filtered_img


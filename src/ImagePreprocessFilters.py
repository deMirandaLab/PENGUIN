import copy

import matplotlib.pyplot as plt
import numpy as np

# import ImageParser as IP
from scipy import ndimage as nd
import cv2



# todo equal from ImageParser  can work even if one channel
def remove_outliers(img: np.array, up_limit=99, down_limit=1) -> np.array:
    # To facilitate pixel annotation, the images are sharpened.
    # More specifically, outliers are removed through saturation of all pixels with
    # values lower than the 1st and higher than the 99th percentile.
    # Means that per each image and per each channel you find the values of the 1st and 99th percentile
    # and all the values below 1st percentile and above 99th percentile are set o zero right?
    imOutlier = img
    for i in range(img.shape[2]):
        ch = img[:, :, i]  # imOutlier[:, :, i]= np.log(img[:, :, i]+0.5).round(4)
        p_99 = np.percentile(ch, up_limit)  # return 50th percentile, e.g median.
        p_01 = np.percentile(ch, down_limit)  # return 50th percentile, e.g median.
        # np.where Where True, yield x, otherwise yield y

        ch = np.where(ch > p_99, p_99, ch) # instead of substitube by 0
        ch = np.where(ch < p_01, 0, ch)
        imOutlier[:, :, i] = ch

    n_pixels_changed = np.sum(imOutlier != img)
    # n_pixels_changed = sum(map(lambda x, y: bool(x - y), imOutlier, img))
    # print('set {} pixels to zero (above {} and below {} percentile threshold per channel out of {}'.
    #       format(n_pixels_changed, up_limit,down_limit, img.shape[0]*img.shape[1]*img.shape[2]))
    # print((n_pixels_changed/(img.shape[0]*img.shape[1]*img.shape[2]))*100, 'pixels changed in saturation')
    return imOutlier

# def normalize_channel_cv2_minmax(img: np.array)-> np.array:
#     # https://www.pythonpool.com/cv2-normalize/
# sem ser por canal faz tudo junto. e como os canais têm valores diferentes faz asneira
#     out = np.zeros(img.shape, np.double)
#     normalized = cv2.normalize(img, out, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
#     return normalized

def normalize_channel_cv2_minmax(img: np.array)-> np.array:
    # https://www.pythonpool.com/cv2-normalize/
    normalized = np.zeros(img.shape)
    for ch in range(img.shape[2]):
        out_img = img[:,:,ch]
        out = np.zeros(out_img.shape, np.double)
        normalized_ch = cv2.normalize(out_img, out, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        normalized[:,:, ch] = normalized_ch
    return normalized

def out_ratio(img, th=0.9):
    if out_ratio:
        # get the binary. and deletes al the faded
        new_img = np.where(img >= th, 1, 0)
        return new_img

def out_ratio2(img, th=0.1):
    new_img = np.where(img >= th, img, 0)
    return new_img

import skimage
def th_otsu(img):
    th = skimage.filters.threshold_otsu(img)
    new_img = np.where(img > th, img, 0)
    return new_img

def th_isodata(img):
    th = skimage.filters.threshold_isodata(img)
    new_img = np.where(img > th, img, 0)
    return new_img

def th_li(img):
    th = skimage.filters.threshold_li(img)
    new_img = np.where(img > th, img, 0)
    return new_img

def th_yen(img):
    th = skimage.filters.threshold_yen(img)
    new_img = np.where(img > th, img, 0)
    return new_img

def th_triangle(img):
    th = skimage.filters.threshold_triangle(img)
    new_img = np.where(img > th, img, 0)
    return new_img

def th_mean(img):
    th = skimage.filters.threshold_mean(img)
    new_img = np.where(img > th, img, 0)
    return new_img

def th_local(img, block_size=3, method='gaussian'):
    th = skimage.filters.threshold_local(img, block_size=block_size, method=method)
    new_img = np.where(img > th, img, 0)
    return new_img
###########################################################################
##### Filters

def percentile_filter(img: np.array, window_size:int = 3, percentile:int=50, transf_bool = True, out_ratio = False )-> np.array:
    '''
    :param img:
    :param window_size:
    :param percentile:
    :param transf_bool:
    :return:
    '''
    from scipy.ndimage import percentile_filter
    kernel = np.ones((window_size, window_size, 1))
    # todo check if disk is better
    # from skimage.morphology import disk
    # kernel = disk(window_size) # disk(3) will create a window 5 *5 plus points on edges
    if transf_bool:
        # will transform to bool. apply the filter and map back the values
        # wi not change any value from the image. will so identify the noise and set those to zero
        # transform to bool (if it has values or not)
        img_to_apply = np.where(img > 0, 1, 0)

    else:
        # will apply the filter directly to the image
        # the positive values from the image will be different
        img_to_apply = copy.deepcopy(img)

    nzero = np.count_nonzero(img_to_apply)

    percentile_blur = percentile_filter(img_to_apply,
                                        percentile=percentile,
                                        footprint=kernel)
    nzero_filter = np.count_nonzero(percentile_blur)


    if transf_bool:
        bl = percentile_blur.astype(bool) # False is Zero
        percentile_blur = np.where(bl == False, 0, img)
        # or
        # data[~bl] = 0

    # get the number of pixel changed. Can be that they are not zeros. or pixels changed that become non zero
    pixel_changed = nzero - nzero_filter
    total_pixel = img.shape[0] * img.shape[1] * img.shape[2]
    percentage_changed = np.round(pixel_changed/total_pixel*100,3)
    # print('total number of pixel changed: {} of {} \n'
    #       'percentage of pixels changed: {}'.format(pixel_changed, total_pixel,percentage_changed))

    return percentile_blur

def percentile_filter_changedpercentiles(img: np.array, window_size:int = 3, percentiles:list=[], transf_bool = True )-> np.array:
    '''

    :param img:
    :param window_size:
    :param percentiles:
    :param transf_bool:
    :return:
    Percentile different from channel

    '''
    from scipy.ndimage import percentile_filter
    kernel = np.ones((window_size, window_size))
    # todo check if disk is better
    # from skimage.morphology import disk
    # kernel = disk(window_size) # disk(3) will create a window 5 *5 plus points on edges

    # make sure that the percentiles input is a list of n channel elements
    if len(percentiles)!= img.shape[-1]:
        raise ValueError(f"Percentiles must have the same number of elements as "
                         f"the number of channels in the image, "
                         f"expected {img.shape[-1]} but got {len(percentiles)}")
    if transf_bool:
        img_to_apply = np.where(img > 0, 1, 0)
    else:
        img_to_apply = copy.deepcopy(img)

    nzero = np.count_nonzero(img_to_apply)

    percentile_blur = np.empty(img.shape)
    for ch in range(img_to_apply.shape[2]):
        img_ch = img_to_apply[:,:,ch]
        med = percentile_filter(img_ch,
                                        percentile=percentiles[ch],
                                        footprint=kernel)
        percentile_blur[:,:,ch] = med

    nzero_filter = np.count_nonzero(percentile_blur)

    if transf_bool:
        bl = percentile_blur.astype(bool) # False is Zero
        percentile_blur = np.where(bl == False, 0, img)
        # or
        # data[~bl] = 0
    # print(np.unique(percentile_blur))
    # # get the number of pixel changed. Can be that they are not zeros. or pixels changed that become non zero
    # pixel_changed = nzero - nzero_filter
    # total_pixel = img.shape[0] * img.shape[1] * img.shape[2]
    # percentage_changed = np.round(pixel_changed/total_pixel*100,3)
    # print('total number of pixel changed: {} of {} \n'
    #       'percentage of pixels changed: {}'.format(pixel_changed, total_pixel,percentage_changed))

    return percentile_blur


def out_ratio_changedth(img: np.array, th:list=[])-> np.array:
    '''

    :param img:
    :param window_size:
    :param percentiles:
    :param transf_bool:
    :return:
    Percentile different from channel

    '''

    th_img = np.empty(img.shape)
    for ch in range(img.shape[2]):
        img_ch = img[:,:,ch]
        new_img = np.where(img_ch >= th, img, 0)
        th_img[:,:,ch] = new_img
    return th_img






def x_shaped_kernel(size):
    # kernel = [[0] * size for _ in range(size)]
    # kernel = np.zeros((size, size,1))
    kernel = np.zeros((size, size)) # withfor loop

    for i in range(size):
        kernel[i][i] = 1
        kernel[i][size - 1 - i] = 1
    return kernel

def plus_shaped_kernel(size):
    # kernel = [[False] * size for _ in range(size)]
    # kernel = np.zeros((size, size,1))
    kernel = np.zeros((size, size)) # withfor loop

    for i in range(size):
        kernel[size//2][i] = True
        kernel[i][size//2] = True
    return kernel

def center_pixel_kernel(size):
    # kernel = [[False] * size for _ in range(size)]
    # kernel = np.zeros((size, size,1))
    kernel = np.zeros((size, size)) # withfor loop

    kernel[size//2][size//2] = True
    return kernel

def hybrid_median_filter(img: np.array, window_size:int = 3, percentile:int=50, transf_bool = True )-> np.array:
    # https://github.com/shurtado/NoiseSuppress/blob/master/imenh_lib.py
    # didnot follow this github. did my own implementation
    from scipy.ndimage import percentile_filter

    if transf_bool:
        img_to_apply = np.where(img > 0, 1, 0)
    else:
        img_to_apply = copy.deepcopy(img)

    kernel_cross = x_shaped_kernel(window_size)
    kernel_plus = plus_shaped_kernel(window_size)
    kernel_center = center_pixel_kernel(window_size)

    median_stack = np.empty(img_to_apply.shape)
    for ch in range(img.shape[2]):
        # get median of kernel_cross and + shape
        img_med = img_to_apply[:,:,ch]
        median_cross = percentile_filter(img_med,percentile=percentile, footprint=kernel_cross)
        median_plus = percentile_filter(img_med,percentile=percentile, footprint=kernel_plus)
        median_pixel = percentile_filter(img_med,percentile=percentile, footprint=kernel_center)

        img_stack = np.dstack((median_cross, median_plus, median_pixel))

        hybrid_median = np.percentile(img_stack, q=percentile, axis=-1)
        median_stack[:,:,ch] = hybrid_median

    if transf_bool:
        bl = median_stack.astype(bool) # False is Zero
        median_stack = np.where(bl == False, 0, img)
    return median_stack

# # todo check
# def mean_filter(img: np.array, window_size:int = 3)-> np.array:
#     from skimage.filters.rank import mean
#     # https://scikit-image.org/docs/stable/auto_examples/filters/plot_rank_mean.html#sphx-glr-auto-examples-filters-plot-rank-mean-py
#     kernel = np.ones((window_size, window_size, 1))
#     normal_result = mean(img, footprint=kernel)
#     return normal_result
#
# # def hybrid_median_image_HMM(img: np.array, max_kernel_size: int = 7 )-> np.array:
# #     import HMM
# # #image: Corresponds to the noise image.
# # # max_kernel_size: The maximun dimension of the kernel, this number must be odd.
# # # figures: Allow to show the original/noise image (named as image) and the denoising image after applied the hybrid median-mean approach; Figures has two options: True for displaying both images or False for not displaying.
# # # plots: Allow to select a square region to measure/quantify the speckle contrast and plot the speckle contrast vs number of iterations; Plots has two options True or False.
# # # save_image: Allow to save the final denoising image after applying the hybrid median-mean method; Save_image has two options True or False.
# # # https://oirl.github.io/Speckle-Hybrid-median-mean/
# # # https://www.spiedigitallibrary.org/journals/optical-engineering/volume-60/issue-12/123107/Speckle-noise-reduction-in-coherent-imaging-systems-via-hybrid-medianmean/10.1117/1.OE.60.12.123107.full?SSO=1
# #     img_HMM = HMM.HybridMedianMean(img, max_kernel_size=max_kernel_size, figures='False', plots ='False', save_image='False')
# #     return img_HMM
# # not appply on multidimensional images may change butnot worth it maybe?
#
#
#
# I think Cv2 does not have
# def adaptive_median_filter(img:np.ndarray, max_size: int = 7, transf_bool = True)-> np.ndarray:
#     import cv2
#     if transf_bool:
#         img_to_apply = np.where(img > 0, 1, 0)
#     else:
#         img_to_apply = copy.deepcopy(img)
#
#     median_stack = np.empty(img_to_apply.shape)
#     for ch in range(img.shape[2]):
#         img_med = img_to_apply[:,:,ch]
#         adaptive_median = cv2.adaptiveMedianBlur(img_med, max_size)
#         median_stack[:,:,ch] = adaptive_median
#
#     if transf_bool:
#         bl = median_stack.astype(bool) # False is Zero
#         median_stack = np.where(bl == False, 0, img)
#
#     return median_stack



# def adaptive_median_filter_multi_channel_CHATGPT(image:np.ndarray, window_size:int=3)-> np.ndarray:
#     # from scipy.signal import medfilt2d
#
#     # written in CHATGPT
#     # the function calculates the median for each channel separately and also the median difference
#     # for each channel separately. It also checks whether the difference between the current pixel and
#     # the median is greater than the median difference for all channels, if so, it replaces the current pixel with the median.
#     # As before, you should experiment with different window sizes to see what works best for your specific image
#     # and level of noise.
#
#     # Create a copy of the image to avoid modifying the original
#     filtered_image = np.copy(image)
#
#     # Define the size of the window to use for median filtering
#     if window_size % 2 == 0:
#         raise ValueError("Window size must be odd")
#
#     # Define the size of the padding for the image
#     padding = (window_size - 1) // 2
#
#     # Pad the image with a mirrored version of itself
#     padded_image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode="reflect")
#
#     # Iterate over each pixel in the image
#     for i in range(padding, padded_image.shape[0] - padding):
#         for j in range(padding, padded_image.shape[1] - padding):
#             # Get the window of pixels around the current pixel
#             window = padded_image[i - padding:i + padding + 1, j - padding:j + padding + 1,:]
#
#             # Calculate the median of the pixels in the window for each channel
#             median = [np.median(window[:,:,c]) for c in range(window.shape[-1])]
#
#             # Calculate the absolute difference of each pixel in the window from the median for each channel
#             abs_diff = np.abs(window - median)
#
#             # Calculate the median of the absolute differences for each channel
#             median_diff = [np.median(abs_diff[:,:,c]) for c in range(abs_diff.shape[-1])]
#
#             # If the difference between the current pixel and the median is greater than the median difference, set the pixel to the median
#             if np.all(np.abs(padded_image[i,j]-median) > median_diff):
#                 filtered_image[i - padding, j - padding,:] = median
#
#     return filtered_image
#


def morphological_filter(image:np.ndarray, structuring_element_size:int=3)-> np.ndarray:
    """

    :param image:
    :param structuring_element_size:
    :return:

    This function uses morphological closing and opening to remove salt and pepper noise from an image.
    The structuring element is a matrix (default is 3x3) that defines the neighborhood of each pixel
    that is considered during the morphological operations. The structuring element is used as a filter
     to determine whether a pixel should be considered for erosion and dilation.
    Morphological closing is a dilation followed by erosion operation, it is used to fill the small white or black
    regions(noise) in the image.
    Morphological opening is an erosion followed by dilation operation, it is used to remove small white or black
    regions(noise) in the image.
    You should experiment with different structuring element sizes to see what works best for your specific image
     and level of noise.
    """

    # write in ChatGPT

    # Create a copy of the image to avoid modifying the original
    filtered_image = np.copy(image)

    # Define the structuring element to use for morphological operations
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (structuring_element_size, structuring_element_size))

    # Perform morphological closing to fill in small white or black regions
    filtered_image = cv2.morphologyEx(filtered_image, cv2.MORPH_CLOSE, structuring_element)

    # Perform morphological opening to remove small white or black regions
    filtered_image = cv2.morphologyEx(filtered_image, cv2.MORPH_OPEN, structuring_element)

    return filtered_image


def filter_hot_pixelsBodenmiller(img: np.ndarray, thres: float) -> np.ndarray:
    '''

    :param img:
    :param thres:
    :return:
    # https://github.com/BodenmillerGroup/ImcSegmentationPipeline/blob/56ce18cfa570770eba169c7a3fb02ac492cc6d4b/src/imcsegpipe/utils.py#L10

    '''
    from scipy.ndimage import maximum_filter
    kernel = np.ones((1, 3, 3), dtype=bool)
    kernel[0, 1, 1] = False
    # array([[[ True,  True,  True],
    #         [ True, False,  True],
    #         [ True,  True,  True]]])
    max_neighbor_img = maximum_filter(img, footprint=kernel, mode="mirror")
    return np.where(img - max_neighbor_img > thres, max_neighbor_img, img)

def modified_hot_pixelsBodenmiller(img: np.ndarray, thres: float, window_size:int = 3) -> np.ndarray:
    """

    :param img:
    :param thres:
    :param window_size:
    :return:
    """
    # changed for channels last and accept window
    # https://bodenmillergroup.github.io/steinbock/latest/cli/preprocessing/
    # https://github.com/BodenmillerGroup/ImcSegmentationPipeline/blob/56ce18cfa570770eba169c7a3fb02ac492cc6d4b/src/imcsegpipe/utils.py#L10
    from scipy.ndimage import maximum_filter
    kernel = np.ones((window_size, window_size,1), dtype=bool)
    line = window_size//2
    kernel[line, line,0] = False  # cneter pixel
    max_neighbor_img = maximum_filter(img, footprint=kernel, mode="mirror")
    return np.where(img - max_neighbor_img > thres, max_neighbor_img,0 ) #img



def gaussian_filter(img: np.array, sigma = 0.2)-> np.array:
    """

    :param img:
    :param sigma:
    :return:
    """
    from scipy.ndimage import gaussian_filter
    # in skimage and scipy is defined by a sigmavalue . in open CVis a kernel window
    # open CV may be faster
    # gaussian_blur = cv.GaussianBlur(img,(5,5),0))  # https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

    gaus = np.empty(img.shape)
    for ch in range(img.shape[2]):
        gaus_ch = img[:,:,ch]
        denoise_gaus = gaussian_filter(gaus_ch, sigma=sigma)
        gaus[:,:,ch] = denoise_gaus
    return gaus


def non_local_means_filter(img: np.ndarray, patch_size:int = 5, patch_distance: int = 11,
                           sigma:float = 0.2) -> np.ndarray:
    """

    :param img:
    :param patch_size:
    :param patch_distance:
    :param sigma:
    :return:
    """
    # they can accept by channel but maybe, it is more suited to do by channel?
    # https://scikit-image.org/docs/stable/auto_examples/filters/plot_nonlocal_means.html
    from skimage.restoration import denoise_nl_means, estimate_sigma
    # estimate the noise standard deviation from the noisy image
    # sigma_est = np.mean(estimate_sigma(img, channel_axis=-1))
    # patch_kw = dict(patch_size=5,      # 5x5 patches 7 by default
    #                 patch_distance=6,  # 13x13 search area 11 by distance
    #                 channel_axis=-1)
    # # If win_size is not specified, it is calculated as max(5, 2 * ceil(3 * sigma_spatial) + 1).
    #
    # denoise = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True,
    #                        **patch_kw)
    nlm = np.empty(img.shape)
    patch_kw = dict(patch_size=patch_size,      # 5 - 5x5 patches    7 by default
                    patch_distance=patch_distance  #6 -  13x13 search area 11 by distance
                    )
    if sigma is None:
        sigma_est = np.mean(estimate_sigma(img, channel_axis=-1))
        h = 1.15 * sigma_est
    else:
        h = sigma
    for ch in range(img.shape[2]):
        nlm_ch = img[:,:,ch]
        denoise_nlm = denoise_nl_means(nlm_ch, h=h, fast_mode=True,
                           **patch_kw)
        nlm[:,:,ch] = denoise_nlm
    return nlm



def bilateral_filter(img: np.array, **params)-> np.array:
    """

    :param img:
    :param sigma:
    :return:
    """
    # is slow
    from skimage.restoration import denoise_bilateral
    bil = np.empty(img.shape)
    for ch in range(img.shape[2]):
        bil_ch = img[:,:,ch]
        denoise_bil = denoise_bilateral(bil_ch, **params)
        bil[:,:,ch] = denoise_bil
    return bil


def total_variation_filter(img: np.array, weight:float = 0.3,**params)-> np.array:
    """

    :param img:
    :param weight:
    :param params:
    :return:
    """
    from skimage.restoration import denoise_tv_chambolle
    # check ifmultichannel needs to be like bilateral   Apply total-variation denoising separately for each channel.
    denoise_TV = denoise_tv_chambolle(img, weight=weight, multichannel=True, **params)
    return denoise_TV


def wavelet_filter(img: np.array)-> np.array:
    """

    :param img:
    :return:
    """
    from skimage.restoration import denoise_wavelet
    wavelet = np.empty(img.shape)
    for ch in range(img.shape[2]):
        img_ch = img[:,:,ch]
        wav = denoise_wavelet(img_ch, multichannel=False,method='BayesShrink',
                              mode='soft',rescale_sigma=True)
        wavelet[:,:,ch] = wav
    # multichannel should be ok
    # sigma if None is the standard deviation
    return wavelet








def anisodiff(img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),sigma=0, option=1,ploton=False):
    import scipy.ndimage.filters as flt
    import warnings
    """
    Anisotropic diffusion.

    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration

    Returns:
            imgout   - diffused image.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference: 
    P. Perona and J. Malik. 
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi  
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.       
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    
    copied from https://www.kaggle.com/code/kmader/anisotropic-diffusion-example/notebook
    """

    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep

        fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
        ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)

        ax1.imshow(img,interpolation='nearest')
        ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in np.arange(1,niter):

        # calculate the diffs
        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)

        if 0<sigma:
            deltaSf=flt.gaussian_filter(deltaS,sigma);
            deltaEf=flt.gaussian_filter(deltaE,sigma);
        else:
            deltaSf=deltaS;
            deltaEf=deltaE;

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaSf/kappa)**2.)/step[0]
            gE = np.exp(-(deltaEf/kappa)**2.)/step[1]
        elif option == 2:
            gS = 1./(1.+(deltaSf/kappa)**2.)/step[0]
            gE = 1./(1.+(deltaEf/kappa)**2.)/step[1]

        # update matrices
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]

        # update the image
        imgout += gamma*(NS+EW)

        if ploton:
            iterstring = "Iteration %i" %(ii+1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()
        # sleep(0.01)

    return imgout





def anisotropic_filtering(img: np.array, niter:int = 1, kappa: int = 50,
                          gamma: float = 0.2, option:int = 1)-> np.array:
    """

    :param img:
    :param niter:
    :param kappa:
    :param gamma:
    :param option:
    :return:
    """
    # from medpy.filter.smoothing import anisotropic_diffusion
    # https://loli.github.io/medpy/generated/medpy.filter.smoothing.anisotropic_diffusion.html
    # niter= number of iterations
    #kappa = Conduction coefficient (20 to 100)
    #gamma = speed of diffusion (<=0.25)
    #Option: Perona Malik equation 1 or 2. A value of 3 is for Turkey's biweight function
    # Equation 1 favours high contrast edges over low contrast ones, while equation 2 favours wide regions over smaller ones. See [R9] for details. Equation 3 preserves sharper boundaries than previous formulations and improves the automatic stopping of the diffusion.
    af = np.empty(img.shape)
    for ch in range(img.shape[2]):
        af_ch = img[:,:,ch]

        # img_aniso_filtered = anisotropic_diffusion(af_ch, niter=niter, kappa=kappa, gamma=gamma, option=option)
        img_aniso_filtered = anisodiff(af_ch,niter=niter,kappa=kappa,gamma=gamma,step=(1.,1.),sigma=0, option=option,ploton=False)
        af[:,:,ch] = img_aniso_filtered
    return af







def bm3d_filter(img: np.array, sigma_psd:float = 0.2)-> np.array:
    """

    :param img:
    :param sigma_psd:
    :return:
    """
    import bm3d
    if sigma_psd is None:
        from skimage.restoration import estimate_sigma
        sigma_psd = np.mean(estimate_sigma(img, multichannel=True))
        print(sigma_psd)

    # for each marker in image
    img_bm3d = np.empty(img.shape)
    for ch in range(img.shape[2]):
        bm3d_ch = img[:,:,ch]
        BM3D_denoised_image = bm3d.bm3d(bm3d_ch, sigma_psd= sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES) # more slow but more powerful
        #BM3D_denoised_image = bm3d.bm3d(noisy_img, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
        img_bm3d[:,:,ch] = BM3D_denoised_image
    return img_bm3d










# todo mixed pipelines


# todo make function that applys differeent thresholds for specific channels
#  in percentile filters

# todo metrics for analysis of noise
#
# from skimage.metrics import peak_signal_noise_ratio
# noise_psnr = peak_signal_noise_ratio(ref_img, noisy_img)
# https://github.com/bnsreenu/python_for_microscopists/blob/master/094_denoising_MRI.py

def save_images(img: np.array, name:str, ch_last:bool = True)-> np.array:
    import tifffile
    img_save = np.float32(img)
    if ch_last == True: # channel is the last axis
        img_save = np.moveaxis(img_save, -1, 0)

    tifffile.imwrite(name, img_save,photometric="minisblack")
    return img

def save_images_ch_names(img: np.array, name:str, ch_last:bool = True, channel_names:list=None)-> np.array:
    import tifffile
    img_save = np.float32(img)
    if ch_last == True: # channel is the last axis
        img_save = np.moveaxis(img_save, -1, 0)

    tifffile.imwrite(name, img_save,photometric="minisblack", metadata={'Channel': {'Name': channel_names}})
    return img

def save_img_ch_names_pages(img: np.array, name:str, ch_last:bool = True, channel_names:list=None)-> np.array:
    import tifffile
    img_save = np.float32(img)
    if ch_last == True: # channel is the last axis
        img_save = np.moveaxis(img_save, -1, 0) # put the channel on first axis

    with tifffile.TiffWriter(name, bigtiff=True) as tiff:
        for i, page in enumerate(img_save):
            tiff.save(page,description=channel_names[i],
                      extratags = [(285,2,None,channel_names[i], False)]) #, metadata=tags #  description=channel_names[i],
    return img_save

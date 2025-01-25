Functions usage
===============

Parse Image
-----------
To parse tiffiles into numpy arrays you can use the ``ImageParser.parse_image()`` function:

.. autofunction:: ImageParser.parse_image

If your tiffiles are not stacks but page based tiffs:

.. autofunction:: ImageParser.parse_image_pages

Lastly, if you want to extract the channel names from the pages of TIFF use ``ImageParser.parse_image_pages_namesCH()`` function.

.. autofunction:: ImageParser.parse_image_pages_namesCH

Preprocessing Image
-------------------
In this pipeline there are two main preprocessing functions: saturation of outliers and the normalization.

To saturate outliers you can use:
.. autofunction:: ImagePreprocessFilters.remove_outliers

To normalize, PENGUIN uses:

.. autofunction:: ImagePreprocessFilters.normalize_channel_cv2_minmax

Thresholding
------------
Thresholding allows to discard background signals, essentially removing signals of low intensity (already normalized).

To do this, the most straightforward approach is thresholding based on the pixel value, where pixel values below this threshold are set to 0.

.. autofunction:: ImagePreprocessFilters.out_ratio2

Other thresholding techniques are also available:

.. autofunction:: ImagePreprocessFilters.th_otsu
.. autofunction:: ImagePreprocessFilters.th_isodata
.. autofunction:: ImagePreprocessFilters.th_li
.. autofunction:: ImagePreprocessFilters.th_yen
.. autofunction:: ImagePreprocessFilters.th_triangle
.. autofunction:: ImagePreprocessFilters.th_mean
.. autofunction:: ImagePreprocessFilters.th_local

Percentile Filter
-----------------
In median filters, the center pixel is substituted with the median of the ranked values from its surrounding pixels. They excel in dealing with impulse noise, as such noise usually ranks at the extreme ends of the brightness scale. Percentile filters, akin to median filters, adjust pixel values based on a range of percentiles rather than solely the median (50th percentile). Different markers may benefit from different values of noise reduction, as they may display more or less shot noise.

To apply percentile filter to each channel:

.. autofunction:: ImagePreprocessFilters.percentile_filter

If you want to apply the hybrid median filter, you can check this implementation:

.. autofunction:: ImagePreprocessFilters.hybrid_median_filter

Save Images
-----------
Lastly, to save the denoised images one can use ``ImagePreprocessFilters.save_images()`` to multitiffs:

.. autofunction:: ImagePreprocessFilters.save_images

To save as multipage tiffs with page names as metadata:

.. autofunction:: ImagePreprocessFilters.save_images_ch_names

And to save the channel names as page names use:

.. autofunction:: ImagePreprocessFilters.save_img_ch_names_pages
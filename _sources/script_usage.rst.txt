
Script usage
============


If you want to process your images directly without notebooks, there are two example pipelines to apply to images
with stacks of channels, or with each channel in a different file.

In this case, you will not be able to interactively check which thresholdings and percentiles best apply to each channel.

The scripts apply the pipeline:
    - saturation of outliers
    - channel normalization
    - thresholding
    - percentile filtering
    - save

The following code is only a snapshot, please check the full script.

For stacks of channels, and with all the parameters defined, the general idea would be as follow:


.. code-block:: console

    images_original = list(map(IP.parse_image_pages, files))

    imgs_out = map(lambda p: IPrep.remove_outliers(p, up_limit, down_limit), images_original)
    imgs_norm = map(IPrep.normalize_channel_cv2_minmax, imgs_out)
    filtered_images = map(lambda i: preprocess_image(i, thresholds, percentiles), imgs_norm)
    imgs_filtered = list(filtered_images)

    # save with channel names
    images_final = map(
        lambda p, f: IPrep.save_img_ch_names_pages(p, f, ch_last=True, channel_names=channel_names),
        imgs_filtered, names_save)

preprocess_image is a function defined in the example and applies thresholding and percentile
per channel.


For channels defined by a single file and organized in patient folders, and with all the parameters defined,
the general idea would be as follow:

.. code-block:: console

    for channel, th, perc in zip(channel_names, thresholds, percentiles):
        file_paths = [file for file in files if str(channel + '.ome.tiff') in str(file)]
        images_original = list(map(IP.parse_image, file_paths))
        imgs_out = map(lambda p: IPrep.remove_outliers(p, up_limit, down_limit), images_original)
        imgs_norm = map(IPrep.normalize_channel_cv2_minmax, imgs_out)
        if isinstance(threshold, float):
            imgs_filtered = list(map(lambda p: IPrep.out_ratio2(p, th=threshold), imgs_norm))
        if percentile is not None:
            imgs_filtered = map(
                lambda p: IPrep.percentile_filter(p, window_size=3, percentile=percentile, transf_bool=True),
                imgs_filtered)

        map(lambda p, f: IPrep.save_images(p, f, ch_last=True), imgs_filtered, names_save)


Please check the scripts for additional parameters. Feel free to adjust all the code.

# import os
# import sys
# from glob import glob
# import numpy as np
#
# try:
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     parent_dir = os.path.dirname(current_dir)
#     sys.path.append(parent_dir)
#     from src.file_specs import FileSpecifics
#     import src.ImagePreprocessFilters as IPrep
#     import src.ImageParser as IP
# except ImportError:
#     print("Error: Could not import modules from the 'src' directory.")
#     print("Please ensure your directory structure is correct and that the 'src' directory exists.")
#     sys.exit(1)
#
#
# def preprocess_image(file_paths, up_limit=99, down_limit=1, threshold=None, percentile=50,
#                      binary_masks=False):
#     images_original = list(map(IP.parse_image, file_paths))
#
#     if len(images_original[0].shape) == 2:  # Check if the shape is 2D
#         images_original = [np.expand_dims(img, axis=-1) for img in images_original]
#
#     # PERCENTILE SATURATION OUTLIERS
#     imgs_out = map(lambda p: IPrep.remove_outliers(p, up_limit, down_limit), images_original)
#
#     # NORMALIZE PER CHANNEL with function from OpenCV
#     imgs_norm = map(IPrep.normalize_channel_cv2_minmax, imgs_out)
#
#     # THRESHOLDING
#     if isinstance(threshold, float):
#         imgs_filtered = list(map(lambda p: IPrep.out_ratio2(p, th=threshold), imgs_norm))
#     elif threshold is None:
#         imgs_filtered = imgs_norm
#     elif threshold in ['otsu', 'isodata', 'Li', 'Yen', 'triangle', 'mean']:
#         threshold_fn = getattr(IPrep, f'th_{threshold}')
#         imgs_filtered = list(map(threshold_fn, imgs_norm))
#     elif threshold == 'local':
#         imgs_filtered = list(map(lambda p: IPrep.th_local(p, block_size=3, method='gaussian'), imgs_norm))
#     else:
#         raise ValueError(f"Invalid threshold type: {threshold}")
#
#     if percentile is not None:
#         imgs_filtered = map(
#             lambda p: IPrep.percentile_filter(p, window_size=3, percentile=percentile, transf_bool=True),
#             imgs_filtered)
#         imgs_filtered = list(imgs_filtered)
#     return imgs_filtered
#     # if binary_masks:
#     #     imgs_filtered = [np.where(a > 0, 1, 0) for a in imgs_filtered]
#     #
#     # names_save = [os.path.join(path_for_results, os.path.basename(os.path.dirname(sub)), os.path.basename(sub)) for
#     #               sub in file_paths]
#     # map(lambda p, f: IPrep.save_images(p, f, ch_last=True), imgs_filtered, names_save)
#     # print('Images saved at ', path_for_results)
#
# def run_test_case_one_file_per_channel_simple():
#     folder_path = 'data_test/one_ch/'
#     path_for_results = 'results_percentile/'
#     # Thresholding
#     threshold = None
#     percentile = 50
#     binary_masks = False
#
#     # load files
#     files = [y for x in os.walk(folder_path) for y in glob(os.path.join(x[0], '*.ome.tiff'))]
#     num_images = len(files)
#     print(f"Number of images identified: {num_images}")
#     if num_images == 0:
#         sys.exit(1)
#
#     channel_names = ['CD45', 'CD68', 'CD31', 'Bcatenin', 'Vimentin']
#     thresholds = [0.1, None, 0.1, 0.1, None]
#     percentiles = [0.5, 0.5, 0.5, 0.5, 0.5]
#
#     for channel, th, perc in zip(channel_names, thresholds, percentiles):
#         files_channel = [file for file in files if str(channel + '.ome.tiff') in str(file)]
#         imgs_filtered = preprocess_image(files_channel,
#                                          99,1,
#                                         th, perc, binary_masks)
#         assert len(imgs_filtered) > 0, "No images after preprocessing."
#         assert len(imgs_filtered) == len(files_channel), "Mismatch in number of images after preprocessing."
#
#         paths_save = [str(path_for_results + os.path.basename(os.path.dirname(sub))) for sub in files_channel]
#         print(paths_save)
#         names_save = [os.path.join(path_for_results, os.path.basename(os.path.dirname(sub)), os.path.basename(sub)) for
#                       sub in file_paths]
#         map(lambda p, f: IPrep.save_images(p, f, ch_last=True), imgs_filtered, names_save)
#         print('Images saved at ', path_for_results)
#         assert len(names_save) == len(imgs_filtered), "Mismatch in number of images saved."
#         assert all(os.exists(names_save)), "Results path does not exist."
#
# def test_other_filters():
#     # This function is a placeholder for testing other filters.
#     # You can implement specific tests for other filters here.
#     pass
#
#
# if __name__ == "__main__":
#     with tempfile.TemporaryDirectory() as temp_results_dir:
#         print(f"Using temporary directory for results: {temp_results_dir}")
#         try:
#             folder_path = 'data_test/one_ch/'
#             path_for_results = temp_results_dir
#             run_test_case_one_file_per_channel_simple()
#             print("\nAll tests passed successfully! 🎉")
#         except AssertionError as e:
#             print(f"Test failed: {e}")
#             sys.exit(1)

import os
import sys
from glob import glob
import numpy as np
import tempfile
import shutil

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from src.file_specs import FileSpecifics
    import src.ImagePreprocessFilters as IPrep
    import src.ImageParser as IP
except ImportError:
    print("Error: Could not import modules from the 'src' directory.")
    print("Please ensure your directory structure is correct and that the 'src' directory exists.")
    sys.exit(1)


def preprocess_image(file_paths, up_limit=99, down_limit=1, threshold=None, percentile=50,
                     ):
    images_original = list(map(IP.parse_image, file_paths))

    if len(images_original[0].shape) == 2:
        images_original = [np.expand_dims(img, axis=-1) for img in images_original]

    imgs_out = map(lambda p: IPrep.remove_outliers(p, up_limit, down_limit), images_original)
    imgs_norm = map(IPrep.normalize_channel_cv2_minmax, imgs_out)

    if isinstance(threshold, float):
        imgs_filtered = list(map(lambda p: IPrep.out_ratio2(p, th=threshold), imgs_norm))
    elif threshold is None:
        imgs_filtered = imgs_norm
    elif threshold in ['otsu', 'isodata', 'li', 'yen', 'triangle', 'mean']:
        threshold_fn = getattr(IPrep, f'th_{threshold}')
        imgs_filtered = list(map(threshold_fn, imgs_norm))
    elif threshold == 'local':
        imgs_filtered = list(map(lambda p: IPrep.th_local(p, block_size=3, method='gaussian'), imgs_norm))
    else:
        raise ValueError(f"Invalid threshold type: {threshold}")

    if percentile is not None:
        imgs_filtered = map(
            lambda p: IPrep.percentile_filter(p, window_size=3, percentile=percentile, transf_bool=True),
            imgs_filtered)
        imgs_filtered = list(imgs_filtered)
    return imgs_filtered


def test_other_filters():
    folder_path = 'data_test/one_ch/'

    # Thresholding parameters
    threshold = None
    percentile = 50

    files = [y for x in os.walk(folder_path) for y in glob(os.path.join(x[0], '*.ome.tiff'))]
    num_images = len(files)
    print(f"Number of images identified: {num_images}")
    if num_images == 0:
        sys.exit(1)

    channel_names = ['CD45', 'CD68', 'CD31', 'Bcatenin', 'Vimentin', 'CD45', 'CD20']
    thresholds = ['otsu', 'isodata', 'li', 'yen', 'triangle', 'mean', 'local']
    percentiles = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5,0.5]

    for channel, th, perc in zip(channel_names, thresholds, percentiles):
        files_channel = [file for file in files if str(channel + '.ome.tiff') in str(file)]
        imgs_filtered = preprocess_image(files_channel,
                                         99, 1,
                                         th, perc)
        assert len(imgs_filtered) > 0, "No images after preprocessing."
        assert len(imgs_filtered) == len(files_channel), "Mismatch in number of images after preprocessing."


def run_test_case_one_file_per_channel_simple():
    folder_path = 'data_test/one_ch/'

    files = [y for x in os.walk(folder_path) for y in glob(os.path.join(x[0], '*.ome.tiff'))]
    num_images = len(files)
    print(f"Number of images identified: {num_images}")
    if num_images == 0:
        sys.exit(1)

    channel_names = ['CD45', 'CD68', 'CD31', 'Bcatenin', 'Vimentin']
    thresholds = [0.1, None, 0.1, 0.1, None]
    percentiles = [0.5, 0.5, 0.5, 0.5, 0.5]

    for channel, th, perc in zip(channel_names, thresholds, percentiles):
        files_channel = [file for file in files if str(channel + '.ome.tiff') in str(file)]
        imgs_filtered = preprocess_image(files_channel,
                                         99, 1,
                                         th, perc)
        assert len(imgs_filtered) > 0, "No images after preprocessing."
        assert len(imgs_filtered) == len(files_channel), "Mismatch in number of images after preprocessing."



def test_save_images(path_for_results):
    folder_path = 'data_test/one_ch/ESD1_01'

    files = [y for x in os.walk(folder_path) for y in glob(os.path.join(x[0], '*.ome.tiff'))]
    num_images = len(files)
    print(f"Number of images identified: {num_images}")
    if num_images == 0:
        sys.exit(1)

    channel_names = ['CD45', 'CD68', 'CD31', 'Bcatenin', 'Vimentin']
    thresholds = [0.1, None, 0.1, 0.1, None]
    percentiles = [0.5, 0.5, 0.5, 0.5, 0.5]

    for channel, th, perc in zip(channel_names, thresholds, percentiles):
        files_channel = [file for file in files if str(channel + '.ome.tiff') in str(file)]
        imgs_filtered = preprocess_image(files_channel,
                                         99, 1,
                                         th, perc)
        assert len(imgs_filtered) > 0, "No images after preprocessing."
        assert len(imgs_filtered) == len(files_channel), "Mismatch in number of images after preprocessing."

        # The path for saving now correctly uses the temporary folder
        names_save = [os.path.join(path_for_results, os.path.basename(os.path.dirname(sub)), os.path.basename(sub)) for
                      sub in files_channel]

        print(names_save)
        if names_save:
            save_dir = os.path.dirname(names_save[0])
            os.makedirs(save_dir, exist_ok=True)
        # We need to wrap the `map` in `list()` to ensure the function is executed
        list(map(lambda p, f: IPrep.save_images(p, f, ch_last=True), imgs_filtered, names_save))
        print('Images saved at ', path_for_results)

        # Correct assertion: check if all saved file paths exist
        assert len(names_save) == len(imgs_filtered), "Mismatch in number of images saved."
        assert all(os.path.exists(name) for name in names_save), "Not all images were saved successfully."


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp_results_dir:
        print(f"Using temporary directory for results: {temp_results_dir}")
        try:
            run_test_case_one_file_per_channel_simple()
            print("\nTest 1 passed successfully!")
        except AssertionError as e:
            print(f"Test failed: {e}")
            sys.exit(1)
        try:
            test_other_filters()
            print("\nTest 2 passed successfully!")
        except AssertionError as e:
            print(f"Test failed: {e}")
            sys.exit(1)

        try:
            test_save_images(temp_results_dir)
            print("\nTest 3 passed successfully!")
        except AssertionError as e:
            print(f"Test failed: {e}")
            sys.exit(1)


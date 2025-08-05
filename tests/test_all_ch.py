import os
import sys
from glob import glob
import numpy as np
import argparse
import tempfile
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


def preprocess_image(img, thresholds, percentiles):
    filtered_img = np.empty(img.shape)
    for ch in range(img.shape[2]):
        img_ch = img[:, :, ch]

        # Thresholding
        th = thresholds[ch]
        if th is not None:
            img_ch = np.where(img_ch >= th, img_ch, 0)

        # Percentile filtering
        perc = percentiles[ch]
        if perc is not None:
            img_ch = img_ch[..., np.newaxis]
            img_ch = IPrep.percentile_filter(img_ch, window_size=3, percentile=perc, transf_bool=True)
            img_ch = img_ch.squeeze()

        filtered_img[:, :, ch] = img_ch
    return filtered_img


folder_path = 'data_test/all_ch/METABRIC22_sample/'
# folder_path = 'data_test/all_ch/stacks_with_names/'
path_for_results = 'data_test/results_percentile/'

def run_test_case_metabric(data_path, results_path):
    # normalization outliers
    up_limit = 99
    down_limit = 1
    binary_masks = False

    # Load files
    files = glob(os.path.join(data_path, '*.tiff'))
    num_images = len(files)
    assert num_images > 0, "No images found in the specified folder."
    assert num_images == 2, "Expected exactly 2 images for testing."

    # Parse image channels
    specs = FileSpecifics(files[0], multitiff=True)
    channel_names = specs.channel_names
    assert specs.channel_names, "Channel names should not be empty."
    num_channels = len(channel_names)
    assert num_channels > 0, "No channels found in the specified image."

    # Calculate thresholds and percentiles
    thresholds = [0.1 for _ in range(num_channels)]
    percentiles = [0.5 for _ in range(num_channels)]

    images_original = list(map(IP.parse_image_pages, files))
    assert len(images_original) == num_images, "Mismatch in number of images parsed."
    assert all(img.shape[2] == num_channels for img in images_original), "All images must have the same number of channels."

    # Preprocessing
    imgs_out = map(lambda p: IPrep.remove_outliers(p, up_limit, down_limit), images_original)
    imgs_norm = map(IPrep.normalize_channel_cv2_minmax, imgs_out)
    filtered_images = map(lambda i: preprocess_image(i, thresholds, percentiles), imgs_norm)
    imgs_filtered = list(filtered_images)

    assert len(imgs_filtered) == num_images, "Mismatch in number of filtered images."
    assert all(img.shape[2] == num_channels for img in imgs_filtered), "All images must have the same number of channels."

    # Save images
    names_save = [os.path.join(results_path, os.path.basename(sub)) for sub in files]
    images_final = map(lambda p, f: IPrep.save_images(p, f, ch_last=True), imgs_filtered, names_save)

    assert len(list(images_final)) == num_images, "Mismatch in number of images saved."
    assert all(os.path.exists(name) for name in names_save), "Not all images were saved successfully."



# # Apply binary masks if needed
# if binary_masks:
#     imgs_filtered = [np.where(a > 0, 1, 0) for a in imgs_filtered]

def run_test_case_with_channel_names(data_path, results_path):
    # normalization outliers
    up_limit = 99
    down_limit = 1
    binary_masks = False

    # Load files
    files = glob(os.path.join(data_path, '*.tiff'))
    num_images = len(files)
    assert num_images > 0, "No images found in the specified folder."

    # Parse image channels
    specs = FileSpecifics(files[0], multitiff=True)
    assert specs.channel_names, "Channel names should not be empty."
    channel_names = specs.channel_names
    num_channels = len(channel_names)
    assert num_channels > 0, "No channels found in the specified image."
    assert all(isinstance(name, str) for name in channel_names), "Channel names should be strings."

    # Calculate thresholds and percentiles
    thresholds = [0.1 for _ in range(num_channels)]
    percentiles = [0.5 for _ in range(num_channels)]

    images_original = list(map(IP.parse_image_pages, files))
    assert len(images_original) == num_images, "Mismatch in number of images parsed."

    # Preprocessing
    imgs_out = map(lambda p: IPrep.remove_outliers(p, up_limit, down_limit), images_original)
    imgs_norm = map(IPrep.normalize_channel_cv2_minmax, imgs_out)
    filtered_images = map(lambda i: preprocess_image(i, thresholds, percentiles), imgs_norm)
    imgs_filtered = list(filtered_images)

    assert len(imgs_filtered) == num_images, "Mismatch in number of filtered images."
    assert all(img.shape[2] == num_channels for img in imgs_filtered), "All images must have the same number of channels."

    # Save images
    # TODO FIX THIS 
    # names_save = [os.path.join(results_path, os.path.basename(sub)) for sub in files]
    # names_save = [os.path.join(results_path, os.path.basename(sub).replace('.ome.tiff', '.tiff')) for sub in files]
    #
    # print(names_save)
    # images_final = map(
    #     lambda p, f: IPrep.save_img_ch_names_pages(p, f, ch_last=True, channel_names=channel_names),
    #     imgs_filtered, names_save)
    # assert all(os.path.exists(name) for name in names_save), "Not all images with channel names were saved successfully."
    #
    # # Check if channel names are correctly saved
    # specs = FileSpecifics(names_save[0], multitiff=True)
    # channel_names = specs.channel_names
    # num_channels = len(channel_names)
    # assert num_channels > 0, "No channels found in the specified image."
    # assert channel_names[0].istype(str), "Channel names should be strings."
    #

if __name__ == "__main__":

    with tempfile.TemporaryDirectory() as temp_results_dir:
        print(f"Using temporary directory for results: {temp_results_dir}")
        try:
            metabric_data_path = 'data_test/all_ch/METABRIC22_sample/'
            run_test_case_metabric(metabric_data_path, temp_results_dir)
            print("\nPass test case for METABRIC data!")

        except AssertionError as e:
            print(f"\nAssertionError: {e}")
            sys.exit(1)

    with tempfile.TemporaryDirectory() as temp_results_dir:
        print(f"Using temporary directory for results: {temp_results_dir}")
        try:
            stacks_data_path = 'data_test/all_ch/stacks_with_names/'
            run_test_case_with_channel_names(stacks_data_path, temp_results_dir)
            print("\nPass test case 2 for stacks with channel names!")
            print("\nAll tests passed successfully! 🎉")

        except AssertionError as e:
            print(f"\nAssertionError: {e}")
            print("Test failed. ❌")
            sys.exit(1)
# del the entire folder
# del path_for_results = 'data_test/results_percentile/'
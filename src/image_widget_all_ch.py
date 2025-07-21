import numpy as np
import pandas as pd
import panel as pn
import os
import ipywidgets as widgets
from IPython.display import display, clear_output
from ipywidgets import SelectMultiple, Layout
from src.file_specs import FileSpecifics
import src.ImagePreprocessFilters as IPrep
import src.ImageParser as IP
import src.jupyter_functions as JN


class ImageWidget:
    def __init__(self, folder_path=None, channel_names=None):
        self.folder_path = folder_path
        self.channel_names = channel_names
        self.files = []

        if folder_path:
            self.load_files_from_folder(folder_path)

        self.output = widgets.Output()
        self.plot_output_histogram = widgets.Output()
        self.plot_output_images = widgets.Output()
        self.plot_output_comparison = widgets.Output()
        self.plot_output_psnr = widgets.Output()
        self.plot_output_ZOOM = widgets.Output()

        self._create_widgets()
        self._attach_observers()

        if self.files:
            self._create_dashboard()

    def _create_widgets(self):
        self.dropdown_ch_name = widgets.Dropdown(description='Channel:')
        self.dropdown_th = widgets.Dropdown(
            options=[None, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 'otsu', 'isodata', 'Li', 'Yen',
                     'triangle', 'mean', 'local'],
            description='Threshold:')
        self.dropdown_percentile = widgets.Dropdown(
            options=[None, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 5, 95, 'p50consecutive'],
            description='Percentile:')
        # self.dropdown_cols = widgets.Dropdown(options=[1, 2, 3, 4, 5], description='Columns to display:')
        self.dropdown_res = widgets.Dropdown(options=[50, 100, 200, 300, 400, 500], description='Resolution:')
        self.dropdown_sample_images = widgets.Dropdown(options=['top', 'bottom', 'random', 'all'],
                                                       description='Subset:')
        self.dropdown_sample_images_number = widgets.Dropdown(options=[1,2,3,4,5,6,7,8],
                                                              description='N img')
        self.dropdown_high_constrast_compare = widgets.Dropdown(options=['False', 'True'],
                                                                description='HighContrast:')


        self.dropdown_image_names = widgets.Dropdown(options=[], description='Img Name:')
        #multiple
        # self.dropdown_image_names = SelectMultiple(
        #     options=['None'],
        #     description='Img Name:',
        #     disabled=False,
        # )



    def _attach_observers(self):
        self.dropdown_ch_name.observe(self._dropdown_eventhandler, names='value')
        self.dropdown_th.observe(self._dropdown_eventhandler, names='value')
        self.dropdown_percentile.observe(self._dropdown_eventhandler, names='value')
        # self.dropdown_cols.observe(self._dropdown_eventhandler, names='value')
        self.dropdown_res.observe(self._dropdown_eventhandler, names='value')
        self.dropdown_sample_images.observe(self._dropdown_eventhandler, names='value')
        self.dropdown_sample_images_number.observe(self._dropdown_eventhandler, names='value')
        self.dropdown_high_constrast_compare.observe(self._dropdown_eventhandler, names='value')


        self.dropdown_image_names.observe(self._dropdown_eventhandler, names='value')


    def _dropdown_eventhandler(self, change):
        self.do_and_plot_percentile()

    def load_files_from_folder(self, folder_path):
        try:
            if not os.path.isdir(folder_path):
                raise NotADirectoryError(f"{folder_path} is not a valid directory.")

            self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tiff')]
            num_images = len(self.files)
            print(f"Number of images identified: {num_images}")

            specs = FileSpecifics(self.files[0], multitiff=True)
            self.channel_names = specs.channel_names

            if self.channel_names:
                if isinstance(self.channel_names, list) and all(
                        isinstance(name, str) for name in self.channel_names):
                    print("Channel names provided.")
                else:
                    print("Using index-based channel selection.")
            else:
                print("No channel names provided. Using index-based channel selection.")

        except Exception as e:
            print(f"Error: {e}")
            self.files = []

    def do_and_plot_percentile(self):
        self.output.clear_output()
        self.plot_output_histogram.clear_output()
        self.plot_output_images.clear_output()
        self.plot_output_comparison.clear_output()
        self.plot_output_psnr.clear_output()
        self.plot_output_ZOOM.clear_output()
        if not self.files:
            self.output.append_stdout("No files to process.")
            return

        # Assuming you have these functions defined elsewhere
        # TODO

        if self.dropdown_image_names.value:
            files_channel = [file for file in self.files if self.dropdown_image_names.value in file]
            # multiple
            # Use the selected files from the dropdown
            # files_channel = [file for file in self.files if
            #                  any(image_name in file for image_name in selected_image_names)]
        else:
            # Use JN.defining_files if no image names are selected
            files_channel = JN.defining_files(self.files, self.dropdown_sample_images_number.value,
                                              self.dropdown_sample_images.value)

        images_original, imgs_norm = JN.preprocess_images_pages(files_channel)

        channel_name = self.dropdown_ch_name.value
        CH = self.channel_names.index(channel_name)
        # imgs_channel = [images_original[i][..., CH] for i in range(len(images_original))]
        # norm_imgs_channel = [imgs_norm[i][..., CH] for i in range(len(imgs_norm))]
        imgs_channel = [images_original[i][np.newaxis,...,CH] for i in range(len(images_original))]
        norm_imgs_channel = [imgs_norm[i][np.newaxis,...,CH] for i in range(len(imgs_norm))]

        def plot_percentile(images, COL_PLOT=2, RES=300):
            JN.set_params(res=RES)
            if COL_PLOT > 1:
                JN.plot_one_channel_side_by_side(images, columns=COL_PLOT, figsize=(50, 50), cmap='gray')
            else:
                JN.plot_one_channel(images, cmap='gray')
            print(files_channel)

        def plot_comparison(images_or, imgs_filtered, res_compare, high_contrast):
            JN.set_params(res=res_compare)
            JN.plot_compare_images(images_or, imgs_filtered, cmap='gray', high_contrast=high_contrast)

        def plot_comparison_zoom(images_or, imgs_filtered, res_compare, high_contrast):
            JN.set_params(res=res_compare)
            JN.plot_compare_images_plotly(images_or, imgs_filtered, cmap='gray', high_contrast=high_contrast)

        # Example values
        PERCENTILE = self.dropdown_percentile.value
        TH = self.dropdown_th.value
        cols = 1
        res = self.dropdown_res.value
        sample_images = self.dropdown_sample_images.value
        sample_images_number = self.dropdown_sample_images_number.value
        high_contrast = self.dropdown_high_constrast_compare.value
        norm_imgs_channel, imgs_filtered = JN.calculus_multitiff(imgs_ch=norm_imgs_channel,
                                                                 PERCENTILE=PERCENTILE,
                                                                 TH=TH)

        with self.plot_output_histogram:
            JN.set_params(res=50)
            JN.histogram_one_channel_all_img(norm_imgs_channel)
            JN.histogram_one_channel_all_img(imgs_filtered)

        with self.plot_output_images:
            plot_percentile(imgs_filtered, cols, res)

        with self.plot_output_comparison:
            plot_comparison([obj[0] for obj in imgs_channel],[obj[0] for obj in imgs_filtered], res, high_contrast)

        with self.plot_output_ZOOM:
            plot_comparison_zoom([obj[0] for obj in imgs_channel],[obj[0] for obj in imgs_filtered], res, high_contrast)

    def _create_dashboard(self):
        self.dropdown_ch_name.options = self.channel_names
        self.dropdown_image_names.options = [os.path.basename(f) for f in self.files]

        # input_widgets = widgets.HBox([self.dropdown_ch_name, self.dropdown_percentile, self.dropdown_th])
        # input_widgets_images = widgets.HBox([self.dropdown_sample_images_number, self.dropdown_sample_images])
        # input_widgets_images2 = widgets.HBox(
        #     [self.dropdown_res, self.dropdown_high_constrast_compare])

        input_widgets = widgets.HBox([self.dropdown_ch_name, self.dropdown_percentile, self.dropdown_th])

        # Create the Display statement
        display_statement = widgets.Label(value='Display:')
        display_statement.layout.margin = '0px 0px 5px 0px'  # Add margin to the bottom

        # Create the dropdown widgets for the number of images subset, high contrast, and resolution
        display_dropdowns = widgets.HBox([self.dropdown_sample_images_number, self.dropdown_sample_images,
                                          self.dropdown_high_constrast_compare, self.dropdown_res,
                                          self.dropdown_image_names])

        # Combine the Display statement and dropdown widgets
        display_layout = widgets.VBox([display_statement, display_dropdowns])

        tab = widgets.Tab([self.plot_output_images, self.plot_output_histogram, self.plot_output_comparison,
                           self.plot_output_ZOOM, self.plot_output_psnr])
        tab.set_title(0, 'Images')
        tab.set_title(1, 'Histogram')
        tab.set_title(2, 'Compare images')
        tab.set_title(3, 'Compare ZOOM')
        tab.set_title(4, 'PSNR')

        dashboard_layout = widgets.Layout(height='auto', overflow='auto', flex='1', max_height='100000px')
        # dashboard = widgets.VBox([input_widgets, input_widgets_images, input_widgets_images2, tab],
        #                          layout=dashboard_layout)
        dashboard = widgets.VBox([input_widgets, display_layout, tab])

        display(dashboard)


    def update_folder_and_channels(self, folder_path):
        self.folder_path = folder_path
        self.load_files_from_folder(folder_path)
        self._create_dashboard()


class SaveWidget:
    def __init__(self):
        self.folder_path = None
        self.channel_names = None
        self.files = []
        self.images_names = []
        self.output_widget = widgets.Output()

        # Define widgets
        self.btn_saving_options = widgets.Button(description='Saving Options')
        self.btn_saving_options.on_click(self._update_output)
        # self.btn_saving_options.on_click(self._saving_options_eventhandler)
        self.PATH = widgets.Text(value='path/to/data', description='Original Path:', disabled=False)
        # self.button = widgets.Button(description="Change Path")
        # self.button.on_click(self._update_output)
        self.edit_table = None
        self.txtbox = None
        self.btn_save = None
        self.btn_save_bin = None
        self.path_for_results = ""

    def _load_files_from_folder(self, folder_path):
        try:
            if not os.path.isdir(folder_path):
                raise NotADirectoryError(f"{folder_path} is not a valid directory.")

            self.images_names = [file for file in os.listdir(folder_path) if "tif" in file]
            self.files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tiff')]
            num_images = len(self.files)
            print(f"Number of images identified: {num_images}")

            specs = FileSpecifics(self.files[0], multitiff=True)
            self.channel_names = specs.channel_names

            if self.channel_names:
                if isinstance(self.channel_names, list) and all(
                        isinstance(name, str) for name in self.channel_names):
                    print("Channel names provided.")
                else:
                    print("Using index-based channel selection.")
            else:
                print("No channel names provided. Using index-based channel selection.")

        except Exception as e:
            print(f"Error: {e}")
            self.files = []

    def _saving_options_eventhandler(self, obj):
        self.generate_saving_widgets()

    def generate_saving_widgets(self):
        clear_output(wait=True)  # Clear the output before displaying the saving options
        if not self.files:
            print("Error: No files found in the specified directory.")
            return
        sel_df = pd.DataFrame(
            {'threshold': [0.1] * len(self.channel_names), 'percentile': [50] * len(self.channel_names)},
            index=self.channel_names)
        tabulator_editors = {'threshold': {'type': 'number', 'max': 1, 'step': 0.1},
                             'percentile': {'type': 'number', 'max': 100, 'step': 5}}
        self.edit_table = pn.widgets.Tabulator(sel_df, editors=tabulator_editors,
                                            configuration={'columnDefaults': {'headerSort': False}})
        self.txtbox = widgets.Text(value='ResultsPercentile/', placeholder='Type something', description='PathSave:',
                                   disabled=False)
        self.btn_save = widgets.Button(description='Save')
        self.btn_save_bin = widgets.Button(description='Save Binary')
        self.btn_save.on_click(self._btn_eventhandler)  # Assign event handler for Save button
        self.btn_save_bin.on_click(self._btn_bin_eventhandler)
        display(self.edit_table)
        display(self.txtbox)
        display(widgets.HBox([self.btn_save, self.btn_save_bin]))

    def _btn_eventhandler(self, obj):
        print('Your images are being saved -np32 format ')
        self.path_for_results = self.txtbox.value
        self.save_imgs(binary_masks=False)

    def _btn_bin_eventhandler(self, obj):
        print('Your Binary images are being saved - np32 format')
        self.path_for_results = self.txtbox.value
        self.save_imgs(binary_masks=True)

    def save_imgs(self, binary_masks=False):
        if not os.path.exists(self.txtbox.value):
            os.makedirs(self.txtbox.value)


        # Process images
        images_original, imgs_norm = JN.preprocess_images_pages(self.files)
        imgs_channel = [images_original[i] for i in range(len(images_original))]
        norm_imgs_channel = [imgs_norm[i] for i in range(len(imgs_norm))]

        # Grab the th_list and the percentile_list
        download_table = self.edit_table
        th_list = list(download_table.value['threshold'])
        percentile_list = list(download_table.value['percentile'])

        # Save parameters to CSV
        csv_path = os.path.join(str(self.path_for_results), 'denoiseParameters.csv')
        df = pd.DataFrame(download_table.value)
        df.to_csv(csv_path, index=True)

        # Filter images
        imgs_filtered = map(lambda i: JN.calculus_multitiff_lists(i, th_list, percentile_list), norm_imgs_channel)
        imgs_filtered = list(imgs_filtered)

        if binary_masks:
            imgs_filtered = [np.where(a > 0, 1, 0) for a in imgs_filtered]

        # Save images individually in their respective folders
        for i, file_name in enumerate(self.files):
            # Create a folder for each file
            file_base_name = os.path.splitext(os.path.basename(file_name))[0]
            folder_path = os.path.join(self.txtbox.value, file_base_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Get the image to save
            img_to_save = imgs_filtered[i]  # Assuming one image per file

            # Define the path to save the image
            save_path = os.path.join(folder_path, f"{file_base_name}.tiff")

            # Save the image
            if isinstance(self.channel_names[0], str):
                IPrep.save_img_ch_names_pages(img_to_save, save_path, ch_last=True, channel_names=self.channel_names)
            else:
                IPrep.save_images(img_to_save, save_path, ch_last=True)
            
            # Create 'tiles.txt' file with a '0' inside
            txt_file_path = os.path.join(folder_path, 'Tiles.txt')
            with open(txt_file_path, 'w') as f:
                f.write('0')

        if self.output_widget:
            self.output_widget.append_stdout(f'Images saved in {self.txtbox.value}\n')
        else:
            print(f'Images saved in {self.txtbox.value}')

    def _update_output(self, button):
        with self.output_widget:
            clear_output(wait=True)
            directory_path = self.PATH.value
            if directory_path:
                print("Path provided:", directory_path)
                if os.path.isdir(directory_path):
                    if os.listdir(directory_path):
                        print("Path provided:", directory_path)
                        self.folder_path = directory_path
                        self._load_files_from_folder(directory_path)
                        self._saving_options_eventhandler(button)  # Call the method here

                    else:
                        print("Error: The specified directory is empty.")
                else:
                    print("Error: The specified path is not a valid directory.")
            else:
                print("Error: No directory path provided.")

    def display_widgets(self):
        display(self.PATH)
        display(self.btn_saving_options)
        display(self.output_widget)
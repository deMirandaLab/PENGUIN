from apeer_ometiff_library import io
import numpy as np
import tifffile

class FileSpecifics:
    """
    Get multipage attributes and channel names for multitiffs.
    """
    def __init__(self, tiff_file_path,multitiff=True):
        self.tiff_file_path = str(tiff_file_path)

        self.tif = tifffile.TiffFile(self.tiff_file_path)

        self.multitiff = multitiff
        self.multipage = False
        if self.multitiff:
            if len(self.tif.pages) > 1: self.multipage = True

        self.channel_names = None
        if self.multipage:
            self.extract_page_names_from_multi_page_tiff()

    def extract_page_names_from_multi_page_tiff(self):
        if 'PageName' in self.tif.pages[0].tags:
            page_names = []
            for page in self.tif.pages:
                if 'PageName' in page.tags:
                    full_page_name = page.tags['PageName']
                    last_part = str(full_page_name).split()[-1]  # Get the last part of the page name
                    page_names.append(last_part)
            self.channel_names = page_names
        else:
            self.channel_names = range(0,len(self.tif.pages))
        return None

    def get_channel_name(self):
        return self.channel_names
    def get_is_mutipage(self):
        return self.multipage



if __name__ == '__main__':
    tiff_page_names = "../data/stacks_with_names/TMA3A_ROI_001_raw.ome.tiff"
    tiff_page_no_names = '../data/METABRIC22_sample/MB0000_64_FullStack.tiff'

    tiff = FileSpecifics(tiff_page_names)
    print('File is multipage:',tiff.multipage )
    print('Channel names:', tiff.channel_names)

    tiff = FileSpecifics(tiff_page_no_names)
    print('File is multipage:',tiff.multipage )
    print('Channel names:', tiff.channel_names)




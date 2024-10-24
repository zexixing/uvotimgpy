import os
import tarfile
import glob
import re
from astropy.table import Table
from utils.file_io import process_astropy_table

class AstroDataOrganizer:
    def __init__(self, source_name, data_root_path=None):
        """
        Initialize the AstroDataOrganizer.

        Parameters:
        source_name (str): Name of the source (e.g., '29p').
        data_root_path (str, optional): Absolute path to the root of the data directory.
        """
        self.source_name = source_name
        self.project_path = os.path.join(data_root_path, source_name)
        self.data_table = Table(names=['obsid', 'exp_no.', 'filter', 'image_data'],
                                dtype=['U11', 'i4', 'U3', 'U5'])

    def organize_data(self):
        """
        Organize the astronomical data.

        Returns:
        astropy.table.Table: Organized data in an Astropy Table.
        """
        self._extract_tars()
        obsid_folders = self._get_obsid_folders()
        for obsid_folder in obsid_folders:
            self._process_obsid_folder(obsid_folder)
        return self.data_table

    def _extract_tars(self, delete_extracted=False):
        """
        Extract all .tar files in the project folder.

        Parameters:
        delete_extracted (bool, optional): If True, delete the .tar files after extraction. Default is False.
        """
        tar_files = glob.glob(os.path.join(self.project_path, '*.tar'))
        for tar_file in tar_files:
            # Use regex to extract the observation ID
            match = re.search(r'sw(\d{11})', os.path.basename(tar_file))
            if match:
                extracted_folder = match.group(1)
            else:
                # If no match, use the tar file name
                extracted_folder = os.path.basename(tar_file)[:-4]

            extracted_path = os.path.join(self.project_path, extracted_folder)
            if not os.path.exists(extracted_path):
                with tarfile.open(tar_file, 'r') as tar:
                    tar.extractall(path=self.project_path)
                print(f"Extracted: {tar_file} to {extracted_path}")
            else:
                print(f"Already extracted: {tar_file}")
            
            if delete_extracted:
                if os.path.exists(extracted_path):
                    os.remove(tar_file)
                    print(f"Deleted: {tar_file}")
                else:
                    print(f"Warning: Extracted folder not found, {tar_file} not deleted.")

    def _get_obsid_folders(self):
        """
        Get a list of observation ID folders.

        Returns:
        list: List of observation ID folder names.
        """
        return [f for f in os.listdir(self.project_path) if f.isdigit() and len(f) == 11]

    def _process_obsid_folder(self, obsid_folder):
        """
        Process a single observation ID folder.

        Parameters:
        obsid_folder (str): Name of the observation ID folder to process.
        """
        uvot_path = os.path.join(self.project_path, obsid_folder, 'uvot')
        image_path = os.path.join(uvot_path, 'image')
        if not (os.path.exists(uvot_path) and os.path.exists(image_path)):
            return None

        sk_files = glob.glob(os.path.join(image_path, f'*{obsid_folder}*_sk.img*'))
        for i, sk_file in enumerate(sk_files, start=1):
            file_name = os.path.basename(sk_file)
            filter_name = file_name[13:16]
            image_data = 'image'

            event_file = glob.glob(os.path.join(uvot_path, 'event', f'*{obsid_folder}{filter_name}*evt*'))
            if event_file:
                image_data = 'event'

            self.data_table.add_row([obsid_folder, i, filter_name, image_data])

    def process_data(self, output_path=None, save_format='csv'):
        """
        Process the data table by either saving it to a file or printing it to console.

        Parameters:
        output_path (str, optional): Path for the output file if saving. Absolute path is recommended.
        format (str, optional): Output file format if saving. Default is 'csv'.
        """
        self.organize_data()
        process_astropy_table(self.data_table, output_path, save_format)


# Usage example
if __name__ == "__main__":
    organizer = AstroDataOrganizer('46P',data_root_path='/Volumes/ZexiWork/data/Swift')
    #organizer.organize_data()
    #organizer.process_data(output_path='1p_uvot_data.csv')
    organizer.process_data()
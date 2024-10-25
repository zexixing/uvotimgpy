import os
import tarfile
import glob
import re
from astropy.table import Table
from utils.file_io import process_astropy_table
from query import StarCoordinateQuery

class AstroDataOrganizer:
    def __init__(self, target_name, data_root_path=None):
        """
        Initialize the AstroDataOrganizer.

        Parameters:
        target_name (str): Name of the target (e.g., '29P').
        data_root_path (str, optional): Absolute path to the root of the data directory.
        """
        self.target_name = target_name
        self.project_path = os.path.join(data_root_path, target_name)
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


class ObservationLogger:
    def __init__(self, target_name, data_root_path, is_motion=True):
        """
        Initialize observation log processor.

        Parameters:
        target_name (str): Name of target object
        data_root_path (str): Root path of data directory
        is_motion (bool, optional): Whether the target is moving. Default is True.
                                  True: Moving target (e.g. comets)
                                  False: Fixed target (e.g. stars)
        """    
        self.target_name = target_name
        self.log_table = None
        self.coordinates = None
        
        # Check if data path exists
        self.project_path = os.path.join(data_root_path, target_name)
        if not os.path.exists(self.project_path):
            raise ValueError(f"Data path does not exist: {self.project_path}")
    
        # Initialize data organizer
        self.organizer = AstroDataOrganizer(target_name, data_root_path)
        self.data_table = self.organizer.organize_data()
        
        # Get coordinates for non-moving target
        self.is_motion = is_motion
        if not self.is_motion:
            query = StarCoordinateQuery()
            coords = query.get_coordinates(target_name)
            if coords is None:
                raise ValueError(f"Cannot find coordinates for target: {target_name}")
            self.coordinates = coords
            self.ra = coords.ra
            self.dec = coords.dec

    def read_fits_header(self, fits_file, extension=0):
        """
        读取单个fits文件的header信息
        返回包含所需header信息的字典
        """
        pass

    def process_all_files(self, file_list):
        """
        处理文件列表中的所有fits文件
        创建初始观测日志表格
        """
        pass

    def calculate_orbit_info(self):
        """
        使用sbpy计算轨道信息
        仅在目标运动时使用
        """
        if not self.is_motion:
            return
        pass

    def calculate_coordinates(self):
        """
        计算目标在图像中的像素坐标
        如果目标运动，使用轨道信息
        如果目标不运动，使用固定坐标
        """
        pass

    def create_output_table(self, selected_columns):
        """
        根据选定的列创建输出表格
        """
        pass

    def save_log(self, output_path):
        """
        将观测日志保存到文件
        """
        pass
        
    def create_observation_log(self, output_path, selected_columns=None):
        """
        创建完整的观测日志
        
        Parameters:
        -----------
        output_path : str
            输出文件的路径
        selected_columns : list, optional
            要包含在输出中的列名列表。如果为None，将包含所有列。
        
        Returns:
        --------
        astropy.table.Table
            生成的观测日志表格
        """
        # 获取文件列表
        file_list = self.organizer.get_file_list()
        
        # 处理所有文件
        self.process_all_files(file_list)
        
        # 根据目标类型计算位置信息
        if self.is_motion:
            self.calculate_orbit_info()
        self.calculate_coordinates()
        
        # 创建并保存输出
        self.create_output_table(selected_columns)
        self.save_log(output_path)
        
        return self.log_table    

def test_observation_logger():
    # 只需要提供目标名称和数据路径
    logger = ObservationLogger(
        target_name="29P",
        data_path="/path/to/data"
    )
    
    # 创建日志
    logger.create_observation_log(
        output_path="output_log.csv",
        selected_columns=['DATE-OBS', 'EXPOSURE', 'FILTER', ...]  # 可选
    )
    
# Usage example
if __name__ == "__main__":
    #organizer = AstroDataOrganizer('46P',data_root_path='/Volumes/ZexiWork/data/Swift')
    #organizer.organize_data()
    #organizer.process_data(output_path='1p_uvot_data.csv')
    #organizer.process_data()
    logger = ObservationLogger('1P', '/Volumes/ZexiWork/data/Swift')
    print(logger.data_table)
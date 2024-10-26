import os
import tarfile
import glob
import re
import numpy as np
from astropy.table import Table, hstack
from astropy.wcs import WCS
from astropy import units as u
from astropy.time import Time
from astropy.io import fits
from sbpy.data import Ephem
from query import StarCoordinateQuery
from utils.file_io import process_astropy_table
from utils.filters import normalize_filter_name

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
        self.data_table = Table(names=['obsid', 'exp_no.', 'filter', 'image_type'],
                                dtype=['U11', 'i4', 'U8', 'U5'])

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
        self.data_table['filter'] = normalize_filter_name(self.data_table['filter'], output_format='display')
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
            
            #filter_name = normalize_filter_name(filter_name, output_format='display')
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

    def read_hdu_header(self, hdu):
        header = hdu.header
            
        # Calculate MIDTIME if DATE-OBS and DATE-END exist
        start_time = Time(header.get('DATE-OBS', ''))
        end_time = Time(header.get('DATE-END', ''))
        mid_time = start_time + (end_time - start_time) / 2

        # Convert T/F strings to boolean
        def str_to_bool(value):
            if isinstance(value, str):
                return value.upper() == 'T'
            return bool(value)
        # Define header keys with their data types and units
        header_info = {
            'DATE-OBS': {'value': header.get('DATE-OBS', ''), 'dtype': 'U23', 'unit': None},
            'DATE-END': {'value': header.get('DATE-END', ''), 'dtype': 'U23', 'unit': None},
            'MIDTIME': {'value': mid_time.isot, 'dtype': 'U23', 'unit': None},
            'EXPOSURE': {'value': header.get('EXPOSURE', 0.0), 'dtype': 'f8', 'unit': u.second},
            'WHEELPOS': {'value': header.get('WHEELPOS', 0), 'dtype': 'i4', 'unit': None},
            'RA_PNT': {'value': header.get('RA_PNT', 0.0), 'dtype': 'f8', 'unit': u.degree},
            'DEC_PNT': {'value': header.get('DEC_PNT', 0.0), 'dtype': 'f8', 'unit': u.degree},
            'PA_PNT': {'value': header.get('PA_PNT', 0.0), 'dtype': 'f8', 'unit': u.degree},
            'ASPCORR': {'value': header.get('ASPCORR', 'None'), 'dtype': 'U6', 'unit': None},
            'MOD8CORR': {'value': str_to_bool(header.get('MOD8CORR', False)), 'dtype': 'bool', 'unit': None},
            'FLATCORR': {'value': str_to_bool(header.get('FLATCORR', False)), 'dtype': 'bool', 'unit': None},
            'CLOCKAPP': {'value': str_to_bool(header.get('CLOCKAPP', False)), 'dtype': 'bool', 'unit': None},
            'WCS': {'value': WCS(header), 'dtype': 'wcs', 'unit': None}
        }

        # Convert values to appropriate data types
        for key, info in header_info.items():
            try:
                if key == 'WCS':
                    continue
                elif info['dtype'] == 'bool':
                    continue  # Already converted by str_to_bool
                elif info['dtype'].startswith('U'):
                    header_info[key]['value'] = str(info['value'])
                elif info['dtype'].startswith('f'):
                    header_info[key]['value'] = float(info['value'])
                elif info['dtype'].startswith('i'):
                    header_info[key]['value'] = int(info['value'])
            except (ValueError, TypeError):
                header_info[key]['value'] = None

        return header_info

    def update_table_info(self):
        """
        Add FITS header information to data_table and return a dictionary containing WCS information
        """
        # Get the first file's path
        obsid = self.data_table[0]['obsid']
        filter_name = self.data_table[0]['filter']
        normalized_filter = normalize_filter_name(filter_name)
        base_filename = f"sw{obsid}{normalized_filter}_sk.img"

        # Try both .gz and non-gz versions
        first_fits = f"{self.project_path}/{obsid}/uvot/image/{base_filename}.gz"
        if not os.path.exists(first_fits):
            first_fits = f"{self.project_path}/{obsid}/uvot/image/{base_filename}"
            if not os.path.exists(first_fits):
                raise FileNotFoundError(f"Cannot find FITS file for obsid {obsid} with filter {filter_name}")

        header_info = None
        hdul = None
        try:
            hdul = fits.open(first_fits)
            for ext_no in range(1, len(hdul)):
                if isinstance(hdul[ext_no], fits.hdu.image.ImageHDU):
                    header_info = self.read_hdu_header(hdul[ext_no])
                    break
        except Exception as e:
            print(f"Error reading first file {first_fits}: {str(e)}")
            if hdul is not None:
                hdul.close()
            return {}
        finally:
            if hdul is not None:
                hdul.close()

        if header_info is None:
            print(f"No valid ImageHDU found in {first_fits}")
            return {}

        # 获取基础列名和数据类型
        column_names = list(self.data_table.colnames)
        dtypes = [self.data_table[col].dtype for col in column_names]

        # 添加ext_no列
        column_names.append('ext_no')
        dtypes.append('i2')

        # 添加header信息的列（排除WCS）
        for key, info in header_info.items():
            if key != 'WCS':
                column_names.append(key)
                if info['dtype'] == 'wcs':
                    continue
                dtypes.append(info['dtype'])

        # 添加像素坐标列
        pixel_columns = ['x_pixel', 'y_pixel']
        column_names.extend(pixel_columns)
        dtypes.extend(['f8'] * len(pixel_columns))

        # 处理所有数据
        final_table, wcs_dict = self.process_all_data(column_names, dtypes) # TODO: 去掉wcs_dict

        # 更新data_table
        self.data_table = final_table

    def process_all_motion_data(self, column_names, dtypes):
        """
        处理所有FITS文件数据，包括头文件信息、轨道信息和像素坐标

        Parameters:
        -----------
        header_info_template : dict
            header信息的模板，用于确定需要提取的键值
        column_names : list
            表格的列名列表
        dtypes : list
            对应的数据类型列表

        Returns:
        --------
        tuple : (astropy.table.Table, dict)
            返回最终合并后的数据表和WCS字典
        """
        new_table = Table(names=column_names, dtype=dtypes)
        wcs_dict = {}
        midtimes = []  # 存储所有的midtime

        for row in self.data_table:
            fits_path = row['image_type'] # TODO: 更改路径

            fits_path = os.path.join(self.project_path, row['obsid'], 'uvot/image')
            hdul = None
            try:
                hdul = fits.open(fits_path)
                for ext_no in range(1, len(hdul)):
                    if isinstance(hdul[ext_no], fits.hdu.image.ImageHDU):
                        # 读取header信息
                        header_info = self.read_hdu_header(hdul[ext_no])

                        # 创建新行
                        new_row = []

                        # 添加原data_table的列值
                        for col in self.data_table.colnames:
                            new_row.append(row[col])

                        # 添加ext_no
                        new_row.append(ext_no)

                        # 添加header信息
                        for key in header_info:
                            if key != 'WCS':
                                new_row.append(header_info[key]['value'])

                        # 计算像素坐标
                        try: # TODO: 检查计算像素坐标的方法
                            x_pixel, y_pixel = self.calculate_pixel_coordinates(
                                header_info['WCS']['value'],
                                header_info['RA_PNT']['value'],
                                header_info['DEC_PNT']['value']
                            )
                            new_row.extend([x_pixel, y_pixel])
                        except Exception as e:
                            print(f"Error calculating pixel coordinates for {fits_path}: {str(e)}")
                            new_row.extend([np.nan, np.nan])

                        # 添加新行到表格
                        new_table.add_row(new_row)

                        # 存储WCS信息
                        wcs_dict[f"{fits_path}_{ext_no}"] = header_info['WCS']['value']

                        # 存储midtime用于后续计算轨道信息
                        midtimes.append(header_info['MIDTIME']['value'])

            except Exception as e:
                print(f"Error processing file {fits_path}: {str(e)}")
            finally:
                if hdul is not None:
                    hdul.close()

        # 计算轨道信息并获取sbpy.ephem对象
        try:
            orbit_ephem = self.calculate_orbit_info(midtimes)
            # 合并new_table和orbit_ephem
            final_table = self.merge_tables(new_table, orbit_ephem)
        except Exception as e:
            print(f"Error calculating orbit information: {str(e)}")
            final_table = new_table

        return final_table

    def calculate_orbit_info(self, times, target_name):
        """
        根据时间列表和目标天体名称计算轨道信息
        
        Parameters:
        -----------
        times : astropy.Time
            观测时间列表
        target_name : str
            目标天体的名称（如彗星或小行星的名字/编号）
        
        Returns:
        --------
        sbpy.data.Ephem
            包含轨道信息的Ephem对象，包含ra、dec、rh、delta等信息
        """
        try:
            # 直接从Horizons获取星历
            eph = Ephem.from_horizons(
                target_name,
            epochs=times,
                quantities=['ra', 'dec', 'delta', 'r', 'elongation', 'phase', 'V']
            )
        
            return eph
        
        except Exception as e:
            print(f"Error calculating orbit information for {target_name}: {str(e)}")
            raise

    def merge_tables(self, fits_table, orbit_ephem):
        """
        合并fits信息表和轨道信息表
        
        Parameters:
        -----------
        fits_table : astropy.table.Table
            包含fits信息的表格
        orbit_ephem : sbpy.data.Ephem
            包含轨道信息的Ephem对象
        
        Returns:
        --------
        astropy.table.Table
            合并后的表格
        """
        # 将sbpy.ephem转换为QTable
        orbit_table = orbit_ephem.table
        
        # 使用hstack合并表格
        merged_table = hstack([fits_table, orbit_table])
        
        return merged_table

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
    organizer = AstroDataOrganizer('1P',data_root_path='/Volumes/ZexiWork/data/Swift')
    #organizer.organize_data()
    #organizer.process_data(output_path='1p_uvot_data.csv')
    #organizer.process_data()
    #test_observation_logger()
    #print(logger.data_table)
import os
import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table, QTable
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from sbpy.data import Ephem
from utils.file_io import process_astropy_table

class HstAstroDataOrganizer:
    def __init__(self, target_name, data_root_path=None):
        """
        Initialize the HstAstroDataOrganizer.
        
        Parameters:
        target_name (str): Name of the target (e.g., '29P').
        data_root_path (str): Absolute path to the root of the data directory.
        """
        self.target_name = target_name
        self.project_path = os.path.join(data_root_path, target_name)
        self.data_table = QTable(names=['date', 'file_name', 'file_path'],
                                dtype=[str, str, str],
                                units=[None, None, None])

    def organize_data(self):
        """
        Organize the HST data.
        
        Returns:
        astropy.table.Table: Organized data in an Astropy Table.
        """
        date_folders = self._get_date_folders()
        for date_folder in date_folders:
            self._process_date_folder(date_folder)
        return self.data_table

    def _get_date_folders(self):
        """
        Get a list of date folders (e.g., 'Nov', 'Sep') under the target folder.
        
        Returns:
        list: List of date folder names.
        """
        target_path = self.project_path
        return [f for f in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, f))]
    
    def _sort_by_filename(self, path):
        filename = os.path.basename(path)
        return int(filename[:-5])
    
    def _process_date_folder(self, date_folder):
        """
        Process a single date folder, adding each FITS file to the data table.
        
        Parameters:
        date_folder (str): Name of the date folder to process.
        """
        date_path = os.path.join(self.project_path, date_folder)
        fits_file_paths = glob.glob(os.path.join(date_path, '*.fits'))
        fits_file_paths = sorted(fits_file_paths, key=self._sort_by_filename)
        for fits_file_path in fits_file_paths:
            file_name = os.path.basename(fits_file_path)
            self.data_table.add_row([date_folder, file_name[:-5], fits_file_path])

class HstObservationLogger:
    """
    A class for logging observations from HST data.
    """
    def __init__(self, target_name, data_root_path, target_alternate=None, location='@hst'):
        """
        Initialize the HstObservationLogger.
        
        Parameters:
        target_name (str): Name of the target (e.g., '29P').
        data_root_path (str): Absolute path to the root of the data directory.
        target_alternate (str, optional): Alternate target name for ephemeris calculation.
        location (str, optional): Observer location for ephemeris calculation.
        """
        self.target_name = target_name
        self.target_alternate = target_alternate
        self.location = location
        self.project_path = os.path.join(data_root_path, target_name)
        if not os.path.exists(self.project_path):
            raise ValueError(f"Data path does not exist: {self.project_path}")
        
        # Initialize data organizer and organize data
        self.organizer = HstAstroDataOrganizer(target_name, data_root_path)
        self.data_table = self.organizer.organize_data()

    def _read_fits_header(self, fits_file_path):
        """
        Read the header of a FITS file and extract necessary information.
        
        Parameters:
        fits_file (str): Path to the FITS file.
        
        Returns:
        dict: Extracted header information.
        """
        with fits.open(fits_file_path) as hdul:
            header_0ext = hdul.header[0]
            header_1ext = hdul.header[1]
            start_time = Time(header_0ext.get('EXPSTART', 0.0), format='mjd')
            end_time = Time(header_0ext.get('EXPEND', 0.0), format='mjd')
            mid_time = start_time + (end_time - start_time) / 2
            header_info = {
                'DATE-OBS': {'value': start_time.isot, 'dtype':str, 'unit': None},
                'DATE-END': {'value': end_time.isot, 'dtype':str, 'unit': None},
                'MIDTIME': {'value': mid_time.isot, 'dtype':str, 'unit': None},
                'EXPTIME': {'value': header_0ext.get('EXPTIME', 0.0), 'dtype':float, 'unit': u.second},
                'FILTER': {'value': header_0ext.get('FILTER', ''), 'dtype':str, 'unit': None},
                'INSTRUME': {'value': header_0ext.get('INSTRUME', ''), 'dtype':str, 'unit': None},
                'REFFRAME': {'value': header_0ext.get('REFFRAME', ''), 'dtype':str, 'unit': None},
                'ORIENTAT': {'value': header_1ext.get('ORIENTAT', ''), 'dtype':float, 'unit': u.degree},
                'WCS': {'value': WCS(header_1ext), 'dtype': 'wcs', 'unit': None},
            }

            # update data types
            for key, info in header_info.items():
                value = info['value']
                if info['dtype'] == bool:
                    value = str(value).upper() == 'T'
                else:
                    try:
                        value = info['dtype'](value)
                        if info['unit'] is not None:
                            value = value * info['unit']
                    except (ValueError, TypeError):
                        value = None
                header_info[key]['value'] = value
        return header_info

    def _calculate_ephemeris(self, times):
        """
        Calculate ephemeris data for the moving target.
        
        Parameters:
        times (list): List of observation times (as astropy Time objects).
        
        Returns:
        sbpy.data.Ephem: Ephemeris data.
        """
        try:
            target = self.target_alternate or self.target_name
            eph = Ephem.from_horizons(target, epochs=times, location=self.location)
            return eph
        except Exception as e:
            print(f"Error calculating ephemeris: {e}")
            return None

    def process_data(self, output_path=None, save_format='csv', selected_columns=None, return_table=False):
        """
        Process observation data and create output table.
        
        Parameters:
        output_path (str, optional): Path to save processed data.
        save_format (str, optional): Format to save data ('csv', 'ascii', etc.).
        selected_columns (list, optional): Columns to include in output.
        return_table (bool): Whether to return the processed table.
        
        Returns:
        astropy.table.Table if return_table is True, None otherwise.
        """
        # Prepare empty table
        column_names = ['date', 'file_name', 'DATE-OBS', 'EXPSTART', 'EXPEND', 'EXPTIME', 
                        'FILTER', 'INSTRUME', 'DETECTOR', 'RA_TARG', 'DEC_TARG',
                        'PROPOSID', 'PROGRAM', 'OBSERVER', 'RA', 'DEC', 'delta', 'r', 'elongation']
        dtypes = [str, str, str, float, float, float, str, str, str, float, float, str, str, str, float, float, float, float, float]
        processed_table = Table(names=column_names, dtype=dtypes)
        
        # Lists to store times for ephemeris calculation
        times = []
        indices = []  # To keep track of the row indices
        
        # Process each FITS file
        for idx, row in enumerate(self.data_table):
            date_folder = row['date']
            file_name = row['file_name']
            fits_path = row['file_path']
            try:
                header_info = self._read_fits_header(fits_path)
                # Parse observation time
                date_obs = header_info['DATE-OBS']
                time_obs = Time(date_obs, format='isot', scale='utc')
                times.append(time_obs)
                indices.append(idx)
                # Prepare row data
                new_row = [
                    date_folder,
                    file_name,
                    header_info['DATE-OBS'],
                    header_info['EXPSTART'],
                    header_info['EXPEND'],
                    header_info['EXPTIME'],
                    header_info['FILTER'],
                    header_info['INSTRUME'],
                    header_info['DETECTOR'],
                    header_info['RA_TARG'],
                    header_info['DEC_TARG'],
                    header_info['PROPOSID'],
                    header_info['PROGRAM'],
                    header_info['OBSERVER'],
                    np.nan,  # Placeholder for RA
                    np.nan,  # Placeholder for DEC
                    np.nan,  # Placeholder for delta
                    np.nan,  # Placeholder for r
                    np.nan,  # Placeholder for elongation
                ]
                processed_table.add_row(new_row)
            except Exception as e:
                print(f"Error processing file {fits_path}: {str(e)}")
        
        # Calculate ephemeris data
        if times:
            eph = self._calculate_ephemeris(times)
            if eph:
                # Update processed_table with ephemeris data
                for idx, eph_row in zip(indices, eph):
                    processed_table['RA'][idx] = eph_row['RA'].value
                    processed_table['DEC'][idx] = eph_row['DEC'].value
                    processed_table['delta'][idx] = eph_row['delta'].to(u.au).value
                    processed_table['r'][idx] = eph_row['r'].to(u.au).value
                    processed_table['elongation'][idx] = eph_row['elongation'].value
        
        # Select columns if specified
        if isinstance(selected_columns, list):
            final_table = processed_table[selected_columns]
        else:
            final_table = processed_table
        
        self.data_table = final_table
        
        # Save or return the table
        if return_table:
            return final_table
        else:
            process_astropy_table(self.data_table, output_path, save_format)
        
# 假设上述类已经定义并导入

def main():
    data_root_path = '/Volumes/ZexiWork/data/HST'  # 替换为您的数据根目录路径
    target_name = '29P'  # 您的目标名称
    organizer = HstAstroDataOrganizer(target_name, data_root_path)
    data_table = organizer.organize_data()
    print(data_table)


    # 创建 HstObservationLogger 实例
    #logger = HstObservationLogger(target_name, data_root_path)

    # 定义输出路径
    #output_path = None  # 替换为您想要保存输出的路径

    # 处理数据并保存结果
    #logger.process_data(output_path=output_path, save_format='csv')

    # 如果需要，返回处理后的数据表
    # processed_table = logger.process_data(return_table=True)

if __name__ == '__main__':
    main()
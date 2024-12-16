import os
import tarfile
import glob
import re
import numpy as np
from astropy.table import hstack, QTable
from astropy.wcs import WCS
from astropy import units as u
from astropy.time import Time
from astropy.io import fits
from sbpy.data import Ephem
from uvotimgpy.query import StarCoordinateQuery
from uvotimgpy.base.file_io import process_astropy_table
from uvotimgpy.utils.filters import normalize_filter_name

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
        self.data_table = QTable(names=['OBSID', 'PNT_NO', 'FILTER', 'DATATYPE'],
                                dtype=[str, int, str, str],
                                units=[None, None, None, None])

    def organize_data(self):
        """
        Organize the astronomical data.

        Returns:
        astropy.table.QTable: Organized data in an Astropy QTable.
        """
        self._extract_tars()
        obsid_folders = self._get_obsid_folders()
        for obsid_folder in obsid_folders:
            self._process_obsid_folder(obsid_folder)
        self.data_table['FILTER'] = normalize_filter_name(self.data_table['FILTER'], output_format='display')
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
    """
    A class for logging observations.
    """
    def __init__(self, target_name, data_root_path, is_motion=True, target_alternate=None, location='@Swift'):
        self.target_name = target_name
        self.target_alternate = target_alternate
        self.location = location
        self.is_motion = is_motion
        
        # Set project path
        self.project_path = os.path.join(data_root_path, target_name)
        if not os.path.exists(self.project_path):
            raise ValueError(f"Data path does not exist: {self.project_path}")
            
        # Initialize parameters
        self.organizer = AstroDataOrganizer(target_name, data_root_path)
        self.data_table = self.organizer.organize_data()
        
        # Initialize header info
        self.header_keys = {
            'DATE-OBS': {'dtype': str, 'unit': None},
            'DATE-END': {'dtype': str, 'unit': None}, 
            'MIDTIME': {'dtype': str, 'unit': None},
            'EXPOSURE': {'dtype': float, 'unit': u.second},
            'WHEELPOS': {'dtype': int, 'unit': None},
            'RA_PNT': {'dtype': float, 'unit': u.degree},
            'DEC_PNT': {'dtype': float, 'unit': u.degree}, 
            'PA_PNT': {'dtype': float, 'unit': u.degree},
            'ASPCORR': {'dtype': str, 'unit': None},
            'MOD8CORR': {'dtype': bool, 'unit': None},
            'FLATCORR': {'dtype': bool, 'unit': None},
            'CLOCKAPP': {'dtype': bool, 'unit': None},
            'WCS': {'dtype': WCS, 'unit': None},
        }

        # Obtain coordinates for fixed targets
        if not is_motion:
            self._set_fixed_coordinates()

    def _set_fixed_coordinates(self):
        """
        Obtain coordinates for fixed target (star) objects.
        
        Raises
        ValueError: If coordinates cannot be found for the target.
        """
        query = StarCoordinateQuery()
        target = self.target_alternate or self.target_name
        coords = query.get_coordinates(target)
        if coords is None:
            raise ValueError(f"Cannot find coordinates for target: {target}")
        self.coordinates = coords
        self.ra = coords.ra
        self.dec = coords.dec

    def _get_fits_path(self, obsid, filter_name):
        """
        Get the path to a FITS file.

        Parameters
        obsid (str): Observation ID.
        filter_name (str): Name of the filter.

        Returns
        str: Path to the FITS file.

        Raises
        FileNotFoundError: If FITS file cannot be found.
        """
        normalized_filter = normalize_filter_name(filter_name, output_format='filename')
        base_filename = f"sw{obsid}{normalized_filter}_sk.img"
        
        # Try file name ended in .gz and not .gz
        for ext in ['.gz', '']:
            path = f"{self.project_path}/{obsid}/uvot/image/{base_filename}{ext}"
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Cannot find FITS file for obsid {obsid} with filter {filter_name}")
    
    def _read_hdu_header(self, hdu):
        """
        A function for processing FITS file header information.

        Parameters
        hdu: FITS HDU object containing header information.

        Returns
        dict: Processed header information.
        """

        # add values and change data types of the values
        header = hdu.header
        header_info = {}
        
        start_time = Time(header.get('DATE-OBS', ''))
        end_time = Time(header.get('DATE-END', ''))
        mid_time = start_time + (end_time - start_time) / 2
        
        for key, info in self.header_keys.items():
            value = header.get(key, '')
            if key == 'MIDTIME':
                value = mid_time.isot
            elif key == 'WCS':
                value = WCS(header)
            elif info['dtype'] == bool:
                value = str(value).upper() == 'T'
            else:
                try:
                    value = info['dtype'](value)
                    if info['unit'] is not None:
                        value = value * info['unit']
                except (ValueError, TypeError):
                    value = None
                    
            header_info[key] = {'value': value, 'dtype': info['dtype'], 'unit': info['unit']}
        return header_info

    def _get_ephemeris_batch(self, times, orbital_keywords, batch_size):
        """
        分批获取历表数据的辅助函数

        Parameters
        ----------
        times : array-like
            时间点列表
        orbital_keywords : list
            需要获取的轨道参数关键字列表
        batch_size : int
            每批处理的时间点数量

        Returns
        -------
        Ephem 
            合并后的历表数据，格式与直接调用eph[orbital_keywords]相同
        """
        results = []  # 存储所有批次的结果

        # 分批处理
        for i in range(0, len(times), batch_size):
            try:
                batch_times = times[i:min(i + batch_size, len(times))]
                target = self.target_alternate or self.target_name
                eph = Ephem.from_horizons(target, location=self.location, epochs=batch_times)
                results.append(eph)
            except Exception as e:
                if "Ambiguous target name" in str(e):
                    print(f"请提供准确的目标ID。错误: {str(e)}")
                else:
                    print(f"错误: {str(e)}")
                raise

        # 如果只有一批数据，直接返回
        if len(results) == 1:
            return results[0][orbital_keywords]

        # 使用sbpy的vstack合并结果
        final_eph = results[0]
        for eph in results[1:]:
            final_eph.vstack(eph)

        return final_eph[orbital_keywords]

    def calculate_orbit_info(self, times, orbital_keywords):
        """
        Calculate orbital information for moving targets with sbpy.

        Parameters
        ----------
        times : array-like
            Time points for orbit calculation.
        orbital_keywords : list
            Keywords for orbital parameters to calculate

        Returns
        -------
        dict
            Orbital ephemeris data.

        Raises
        ------
        Exception
            If target name is ambiguous, or batch calculation fails, or other unexpected errors happen.
        """
        try:
            # 如果时间点数量大于阈值，使用批处理
            batch_size = 50
            if len(times) > batch_size:
                return self._get_ephemeris_batch(times, orbital_keywords, batch_size)
            else:
                # 原始的单批次处理
                target = self.target_alternate or self.target_name
                eph = Ephem.from_horizons(target, location=self.location, epochs=times)
                return eph[orbital_keywords]

        except Exception as e:
            if "Ambiguous target name" in str(e):
                print(f"请提供准确的目标ID。错误: {str(e)}")
            elif "414 Request-URI Too Large" in str(e):
                # 如果URL过长，自动切换到批处理模式
                try:
                    return self._get_ephemeris_batch(times, orbital_keywords)
                except Exception as batch_e:
                    print(f"批处理模式也失败: {str(batch_e)}")
                    raise
            else:
                print(f"错误: {str(e)}")
            raise

    def _prepare_table_structure(self):
        """
        Create table structure based on FITS headers.

        Returns
        -------
        tuple: Column names and data types
        """
        
        # Build columns and data types
        column_names = list(self.data_table.colnames)
        dtypes = [self.data_table[col].dtype for col in column_names]
        units = [self.data_table[col].unit for col in self.data_table.colnames]
        
        # Add extra columns
        extra_columns = [('EXT_NO', int, None)]
        extra_columns.extend([(k, v['dtype'], v['unit']) for k, v in self.header_keys.items() if k != 'WCS'])
        extra_columns.extend([('x_pixel', float, None), ('y_pixel', float, None)])
        
        for col, dtype, unit in extra_columns:
            column_names.append(col)
            dtypes.append(dtype)
            units.append(unit)
        return column_names, dtypes, units

    def _process_fits_file(self, fits_path, row, processed_table, wcs_dict, midtimes):
        """
        Extract data from one FITS file.

        Parameters
        ----------
        fits_path (str): Path to FITS file
        row (Row): Data table row
        processed_table (Table): Output table
        wcs_dict (dict): WCS information storage
        midtimes (list): Observation times
        """
        try:
            with fits.open(fits_path) as hdul:
                for ext_no in range(1, len(hdul)):
                    if isinstance(hdul[ext_no], fits.hdu.image.ImageHDU):
                        # Obtain and process header information
                        header_info = self._read_hdu_header(hdul[ext_no])

                        # prepare empty row
                        new_row = []

                        # Add old data
                        for col in self.data_table.colnames:
                            new_row.append(row[col])

                        # Add extension number
                        new_row.append(ext_no)

                        # Add header information
                        for key, info in header_info.items():
                            if key != 'WCS':
                                new_row.append(info['value'])

                        # Add pixel coordinates; Obtain observation time and wcs list for moving targets
                        if self.is_motion:
                            new_row.extend([np.nan, np.nan])
                            midtimes.append(header_info['MIDTIME']['value'])
                            wcs_dict[f"{row['OBSID']}_{row['FILTER']}_{ext_no}"] = header_info['WCS']['value']
                        else:
                            wcs = header_info['WCS']['value']
                            x, y = wcs.all_world2pix(self.ra, self.dec, 1)
                            new_row.extend([x, y])

                        # Add the new row to the table
                        processed_table.add_row(new_row)

        except Exception as e:
            print(f"Error processing file {fits_path}: {str(e)}")

    def _process_motion_target(self, processed_table, midtimes, wcs_dict, orbital_keywords):
        """
        Add orbital data for moving targets.

        Parameters
        ----------
        processed_table (Table): Based data table
        midtimes (list):  
            Observation times
        wcs_dict (dict): WCS information
        orbital_keywords (list, optional): Keywords for orbital parameters to calculate

        Returns
        -------
        Table with added motion data
        """
        try:
            # Get orbital data
            orbit_ephem = self.calculate_orbit_info(Time(midtimes),orbital_keywords)

            # Merge table
            orbit_table = orbit_ephem.table
            merged_table = hstack([processed_table, orbit_table])

            # Calculate pixel positions
            x_pixels = []
            y_pixels = []

            for row in merged_table:
                wcs_key = f"{row['OBSID']}_{row['FILTER']}_{row['EXT_NO']}"
                wcs = wcs_dict[wcs_key]
                try:
                    x, y = wcs.all_world2pix(row['RA'], row['DEC'], 1)
                    x_pixels.append(x)
                    y_pixels.append(y)
                except Exception as e:
                    print(f"Error calculating pixel coordinates for {wcs_key}: {str(e)}")
                    x_pixels.append(np.nan)
                    y_pixels.append(np.nan)

            merged_table['x_pixel'] = x_pixels
            merged_table['y_pixel'] = y_pixels

            return merged_table

        except Exception as e:
            #print(f"Error processing motion target: {str(e)}")
            return processed_table
        
    def process_data(self, output_path=None, save_format='csv', selected_columns=None, return_table=False,
                     orbital_keywords=['ra', 'dec', 'delta', 'r', 'elongation']):
        """
        Process observation data and create output table.

        Parameters
        output_path (str, optional): Path to save processed data.
        save_format (str, optional): Format to save data ('csv', 'ascii.ecsv', 'fits', etc.).
        selected_columns (list, optional): Columns to include in output.
        return_table (bool): Whether to return processed table.
        orbital_keywords (list, optional): Keywords for orbital parameters to calculate

        Returns
        astropy.table.Table if return_table is True, None otherwise.
        """
        # Prepare empty table
        column_names, dtypes, units = self._prepare_table_structure()
        processed_table = QTable(names=column_names, dtype=dtypes, units=units)
        
        # Process each observation
        wcs_dict = {}
        midtimes = []
        
        for row in self.data_table:
            fits_path = self._get_fits_path(row['OBSID'], row['FILTER'])
            self._process_fits_file(fits_path, row, processed_table, wcs_dict, midtimes)
            
        # Add orbital data for moving targets
        if self.is_motion and midtimes:
            final_table = self._process_motion_target(processed_table, midtimes, wcs_dict, orbital_keywords)
        else:
            final_table = processed_table
            
        # Select columns and save
        if isinstance(selected_columns, list):
            final_table = final_table[selected_columns]
        self.data_table = final_table
        
        if return_table:
            return final_table
        else:
            process_astropy_table(final_table, output_path, save_format)
        
# Usage example

if __name__ == "__main__":
    #organizer = AstroDataOrganizer('C_2017K2',data_root_path='/Volumes/ZexiWork/data/Swift')
    #organizer = AstroDataOrganizer('1P',data_root_path='/Volumes/ZexiWork/data/Swift')
    #organizer.process_data()
    #organizer.process_data(output_path='1p_uvot_data.csv')
    #organizer.process_data()
    #print(logger.data_table)
    logger = ObservationLogger('46p',data_root_path='/Volumes/ZexiWork/data/Swift', target_alternate='90000549')
    output_path = '/Users/zexixing//Downloads/46p_uvot_data.csv'
    logger.process_data(output_path=output_path,
                        orbital_keywords=['ra', 'dec', 'delta', 'r', 'elongation'])
import os
import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import hstack, QTable
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from sbpy.data import Ephem
from uvotimgpy.base.file_io import process_astropy_table


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
    def __init__(self, target_name, data_root_path, target_alternate=None, location='500'):
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

        self.header_keys = {
            'DATE-OBS': {'dtype':str, 'unit': None},
            'DATE-END': {'dtype':str, 'unit': None},
            'MIDTIME': {'dtype':str, 'unit': None},
            'EXPTIME': {'dtype':float, 'unit': u.second},
            'FILTER': {'dtype':str, 'unit': None},
            'INSTRUME': {'dtype':str, 'unit': None},
            'REFFRAME': {'dtype':str, 'unit': None},
            'ORIENTAT': {'dtype':float, 'unit': u.degree},
            'WCS': {'dtype': 'wcs', 'unit': None},
        }

    def _read_fits_header(self, fits_file_path):
        """
        Read the header of a FITS file and extract necessary information.
        
        Parameters:
        fits_file (str): Path to the FITS file.
        
        Returns:
        dict: Extracted header information.
        """
        with fits.open(fits_file_path) as hdul:
            header_0ext = hdul[0].header
            header_1ext = hdul[1].header
            header_info = {}
            start_time = Time(header_0ext.get('EXPSTART', 0.0), format='mjd')
            end_time = Time(header_0ext.get('EXPEND', 0.0), format='mjd')
            mid_time = start_time + (end_time - start_time) / 2

            for key, info in self.header_keys.items():
                try:
                    if key == 'DATE-OBS':
                        value = start_time.isot
                    elif key == 'DATE-END':
                        value = end_time.isot
                    elif key == 'MIDTIME':
                        value = mid_time.isot
                    elif key == 'WCS':
                        value = WCS(header_1ext)
                    elif key in ['ORIENTAT']:
                        value = info['dtype'](header_1ext.get(key, ''))
                    else:
                        value = info['dtype'](header_0ext.get(key, ''))
                    if info['unit'] is not None:
                        value = value * info['unit']
                except (ValueError, TypeError):
                    value = None
                header_info[key] = {'value': value, 'dtype': info['dtype'], 'unit': info['unit']}

        return header_info

    def calculate_orbit_info(self, times,
                             orbital_keywords):
        """
        Calculate orbital information for moving targets with sbpy.

        Parameters
        times: Time points for orbit calculation.
        orbital_keywords (list, optional): Keywords for orbital parameters to calculate

        Returns
        Ephem: Orbital ephemeris data.

        Raises
        Exception: If target name is ambiguous or calculation fails.
        """
        try:
            target = self.target_alternate or self.target_name
            eph = Ephem.from_horizons(target, location=self.location, epochs=times)
            return eph[orbital_keywords]
        except Exception as e:
            if "Ambiguous target name" in str(e):
                print(f"Please provide exact target ID. Error: {str(e)}")
            raise

    def _process_fits_file(self):
        """
        Process observation data and create output table with header info.
        """
        # Prepare empty table
        column_names = list(self.data_table.colnames[:-1])
        column_names_old = list(self.data_table.colnames[:-1])
        dtypes = [self.data_table[col].dtype for col in column_names]
        units = [self.data_table[col].unit for col in column_names]

        extra_columns = ([(k, v['dtype'], v['unit']) for k, v in self.header_keys.items() if k != 'WCS'])
        extra_columns.extend([('x_pixel', float, None), ('y_pixel', float, None)])
                    
        for col, dtype, unit in extra_columns:
            column_names.append(col)
            dtypes.append(dtype)
            units.append(unit)

        processed_table = QTable(names=column_names, dtype=dtypes, units=units)
        # ------------
        # Lists to store times for ephemeris calculation
        targettimes = []
        wcs_dict = {}
        
        # Process each FITS file
        for row in self.data_table:
            fits_path = row['file_path']
            try:
                header_info = self._read_fits_header(fits_path)
                # Parse observation time
                date_obs = header_info['DATE-OBS']['value']
                #time_obs = Time(date_obs, format='isot', scale='utc')
                targettimes.append(date_obs)
                # Prepare row data
                new_row = []
                for col in column_names_old:
                    new_row.append(row[col])
                for key, info in header_info.items():
                    if key != 'WCS':
                        new_row.append(info['value'])
                new_row.extend([np.nan, np.nan])
                wcs_dict[f"{row['date']}_{row['file_name']}"] = header_info['WCS']['value']
                processed_table.add_row(new_row)
            except Exception as e:
                print(f"Error processing file {fits_path}: {str(e)}")
        return processed_table, targettimes, wcs_dict
    
    def _merge_ephem_table(self, processed_table, targettimes, wcs_dict, orbital_keywords):
        orbit_ephem = self.calculate_orbit_info(Time(targettimes),orbital_keywords)
        orbit_table = orbit_ephem.table
        merged_table = hstack([processed_table, orbit_table])

        x_pixels = []
        y_pixels = []

        for row in merged_table:
            wcs_key = f"{row['date']}_{row['file_name']}"
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
    
    def process_data(self, output_path=None, save_format='csv', selected_columns=None, return_table=False,
                     orbital_keywords=['ra', 'dec', 'delta', 'r', 'elongation']):
        processed_table, targettimes, wcs_dict = self._process_fits_file()
        final_table = self._merge_ephem_table(processed_table, targettimes, wcs_dict, orbital_keywords)

        if isinstance(selected_columns, list):
            final_table = final_table[selected_columns]
        self.data_table = final_table

        if return_table:
            return final_table
        else:
            process_astropy_table(final_table, output_path, save_format)

def main():
    data_root_path = '/Volumes/ZexiWork/data/HST'  # 替换为您的数据根目录路径
    target_name = '29P'  # 您的目标名称
    #organizer = HstAstroDataOrganizer(target_name, data_root_path)
    #data_table = organizer.organize_data()
    #print(data_table)
    logger = HstObservationLogger(target_name, data_root_path, target_alternate='90000395')
    output_path = None
    logger.process_data(output_path=output_path, save_format='csv')

if __name__ == '__main__':
    main()
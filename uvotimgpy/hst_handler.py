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
from uvotimgpy.base.math_tools import GaussianFitter2D
from uvotimgpy.base.region import RegionSelector
import matplotlib.pyplot as plt
from IPython import get_ipython




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

        self.header_keys = {
            'DATE_OBS': {'dtype':str, 'unit': None},
            'DATE_END': {'dtype':str, 'unit': None},
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
                    if key == 'DATE_OBS':
                        value = start_time.isot
                    elif key == 'DATE_END':
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
                date_obs = header_info['MIDTIME']['value']
                time_obs = Time(date_obs, format='isot', scale='utc')
                targettimes.append(time_obs)
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
    
    def _add_fitted_center_pixel(self, processed_table):
        center_pixel_path = '/Volumes/ZexiWork/projects/29p/HST/comet_position_hst.csv'
        data = np.genfromtxt(center_pixel_path, delimiter=',', skip_header=1)
        fitted_col_pixel = data[:,2]
        fitted_row_pixel = data[:,3]
        fitted_x_pixel = fitted_col_pixel + 1
        fitted_y_pixel = fitted_row_pixel + 1
        processed_table['fitted_x_pixel'] = fitted_x_pixel
        processed_table['fitted_y_pixel'] = fitted_y_pixel
        return processed_table
    
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
                     orbital_keywords=['ra', 'dec', 'delta', 'r', 'elongation', 'alpha']):
        processed_table, targettimes, wcs_dict = self._process_fits_file()
        processed_table = self._add_fitted_center_pixel(processed_table)
        final_table = self._merge_ephem_table(processed_table, targettimes, wcs_dict, orbital_keywords)

        if isinstance(selected_columns, list):
            final_table = final_table[selected_columns]
        self.data_table = final_table

        if return_table:
            return final_table
        else:
            process_astropy_table(final_table, output_path, save_format)

def fit_peak_in_region(image, region):
    """
    在给定region内拟合带旋转角度的高斯函数并返回峰值位置在原始图像中的坐标
    
    Parameters
    ----------
    image : np.ndarray
        输入的2D图像
    region : regions.PixelRegion
        要分析的区域
        
    Returns
    -------
    tuple
        峰值在原始图像中的坐标和旋转角度 (col, row, theta)
    """
    # 获取region的mask和cutout
    mask = region.to_mask()
    cutout = mask.cutout(image)
    mask_data = mask.data
    
    # 创建有效数据掩模
    valid_mask = mask_data > 0
    
    # 获取有效像素的信息
    rows, cols = np.where(valid_mask)
    values = cutout[valid_mask]
    

    # 获取bounding_box信息用于坐标转换
    bbox = region.bounding_box
    row_min, row_max, col_min, col_max = bbox.iymin, bbox.iymax, bbox.ixmin, bbox.ixmax

    # 计算更合理的初始参数
    height, width = cutout.shape
    max_value = np.max(values)
    background = np.percentile(values, 10)  # 使用较低百分位数作为背景估计
    
    # 使用最大值位置作为中心的初始猜测
    max_pos = np.unravel_index(np.argmax(cutout), cutout.shape)
    initial_row, initial_col = max_pos
    
    # 估计初始sigma（使用区域大小的1/4到1/6）
    initial_sigma = min(width, height) / 5
    
    # 确保sigma不会太小
    initial_sigma = max(initial_sigma, 1.0)
    
    # 初始旋转角度设为0
    initial_theta = 0.0
    
    # 创建旋转高斯拟合器
    gaussian_fitter = GaussianFitter2D()
    
    try:
        # 设置更合理的初始参数
        fitted_model, _ = gaussian_fitter.fit(
            cutout,
            n_gaussians=1,
            threshold=background,  # 使用估计的背景值作为阈值
            position_list=[(initial_col, initial_row)],
            amplitude_list=[max_value - background],  # 减去背景值
            sigma_list=[initial_sigma],
            theta_list=[initial_theta],  # 添加初始旋转角度
        )
        
    except Exception as e:
        print("\n拟合出错:", str(e))
        print("\n详细诊断信息:")
        print("初始参数:")
        print(f"- 中心位置 (col, row): ({initial_col}, {initial_row})")
        print(f"- 振幅: {max_value - background}")
        print(f"- Sigma: {initial_sigma}")
        print(f"- Theta: {initial_theta}")
        print(f"- 背景: {background}")
        print("\n数据统计:")
        print(f"- 最大值: {max_value}")
        print(f"- 最小值: {np.min(values)}")
        print(f"- 平均值: {np.mean(values)}")
        print(f"- 中位数: {np.median(values)}")
        print(f"- 标准差: {np.std(values)}")
        raise

    # 获取拟合后的高斯函数参数（在cutout坐标系中）
    g = fitted_model[0]  # 第一个高斯分量
    col_cutout = g.x_mean.value  # 在cutout中的col坐标
    row_cutout = g.y_mean.value  # 在cutout中的row坐标
    theta = g.theta.value  # 旋转角度（弧度）
    
    # 转换到原始图像坐标系
    col_orig = col_cutout + col_min
    row_orig = row_cutout + row_min

    fig = gaussian_fitter.plot_results(cutout, fitted_model)
    plt.show()
    
    return (col_orig, row_orig, theta)

def obtain_real_position(obs_table, i):
    data_root_path = '/Volumes/ZexiWork/data/HST'
    target_name = '29P'
    obs_info = obs_table[i]
    date = obs_info['date']
    file_name = obs_info['file_name']
    filepath = os.path.join(data_root_path, target_name, date, f'{file_name}.fits')
    x_pixel = obs_info['x_pixel']
    y_pixel = obs_info['y_pixel']
    col_pixel = x_pixel - 1 
    row_pixel = y_pixel - 1
    row_range = (row_pixel-100, row_pixel+100)
    col_range = (col_pixel-100, col_pixel+100)

    # 读取图像
    with fits.open(filepath) as hdul:
        image = hdul[1].data
        
    # 获取regions
    selector = RegionSelector(image, vmin=0, vmax=2, row_range=row_range, col_range=col_range, shape='square')
    plt.pause(0.1)
    aperture = selector.get_apertures()[0]

    col_py, row_py, theta = fit_peak_in_region(image, aperture)
    x_ds9 = col_py + 1
    y_ds9 = row_py + 1
    
    plt.imshow(image, vmin=0, vmax=2, origin='lower')
    plt.xlim(col_pixel-100, col_pixel+100)
    plt.ylim(row_pixel-100, row_pixel+100)
    plt.plot(col_py, row_py, color = 'r', marker='x', markersize=5)
    plt.plot(col_pixel, row_pixel, color = 'w', marker='x', markersize=5)
    plt.show()

    print(date, file_name, theta, x_pixel, y_pixel, x_ds9, y_ds9)

if __name__ == '__main__':
    data_root_path = '/Volumes/ZexiWork/data/HST'
    target_name = '29P'
    logger = HstObservationLogger(target_name, data_root_path, target_alternate='90000395')
    obs_table = logger.process_data(save_format='csv', output_path='/Volumes/ZexiWork/projects/29p/HST/obs_log_29p_hst.csv')
    print(obs_table)
    #obtain_real_position(obs_table, 1)
    #print(obs_table)
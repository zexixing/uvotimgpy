import os
from astropy.io import fits
from datetime import datetime
from astropy.time import Time, TimeDelta
import numpy as np
from typing import Union, Tuple, List, Dict, Any, Optional
from astropy.table import Table
import subprocess
from pathlib import Path
import inspect
import ipynbname
import pandas as pd
import astropy.units as u
from astropy.units import Quantity
from uvotimgpy.config import paths
from sbpy.data import Ephem
import csv
import re
from astropy.coordinates import SkyCoord
import matplotlib.dates as mdates



def save_astropy_table(data_table, output_path, verbose=True):
    """
    Save an Astropy table to a file.

    Parameters:
    data_table (astropy.table.Table): The Astropy table to save.
    output_path (str): Absolute path for the output file. ecsv recommended.
    """
    #os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if isinstance(data_table, Table):
        data_table.write(output_path, overwrite=True, delimiter=',')
        if verbose:
            file_format = str(output_path).split('.')[-1]
            print(f"Data saved to: {output_path}, format: {file_format}")

def show_or_save_astropy_table(data_table, output_path=None):
    """
    Process an Astropy table by either saving it to a file or printing it to console.

    Parameters:
    data_table (astropy.table.Table): The Astropy table to process.
    output_path (str, optional): Path for the output file. If None, the table will be printed to console. For saving, absolute is recommended.
    """
    if output_path is None:
        data_table.pprint(max_lines=-1, max_width=-1)
    else:
        save_astropy_table(data_table, output_path)

def load_table(table_path, type='astropy_table', verbose=True):
    if verbose:
        file_format = str(table_path).split('.')[-1]
        print(f"Loading data from: {table_path}, format: {file_format}")
    file_path = Path(table_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if type == 'df' and file_path.suffix.lower() == '.csv':
        # CSV文件
        dtype_dict = {'OBSID': str} if 'OBSID' in pd.read_csv(file_path, nrows=0).columns else {}
        df = pd.read_csv(file_path, dtype=dtype_dict)
        return df
    elif type == 'astropy_table' and file_path.suffix.lower() in ['.ecsv', '.csv']:
        # ECSV文件 (astropy格式)
        table = Table.read(file_path)
        return table 
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")

def is_dict_list(obj) -> bool:
    return (
        isinstance(obj, list)
        and all(isinstance(item, dict) for item in obj)
    )

def is_list_dict(obj) -> bool:
    return (
        isinstance(obj, dict)
        and all(isinstance(v, list) for v in obj.values())
        and len(set(len(v) for v in obj.values())) <= 1  # 可选：所有列表长度一致
    )

def remove_table_units(table):
    """
    通过转换为 pandas DataFrame 来移除astropy Table的单位
    """
    # 转换为 pandas DataFrame（自动移除单位）
    df = table.to_pandas()
    
    # 转回 astropy Table
    return Table.from_pandas(df)

class TableConverter:
    @staticmethod
    def df_to_astropy_table(df: pd.DataFrame) -> Table:
        """
        Convert a DataFrame to an Astropy Table.
        Assumes each value in the DataFrame is either:
        - a Quantity (e.g., [1*u.km, 2*u.km]) [RECOMMENDED]
        - or a normal scalar
        """
        table = Table()
        for col in df.columns:
            col_data = df[col]
            if isinstance(col_data.iloc[0], Quantity):
                table[col] = u.Quantity(col_data.tolist())
            else:
                table[col] = col_data.tolist()
        return table

    @staticmethod
    def astropy_table_to_df(table: Table) -> pd.DataFrame:
        """Convert Astropy Table to DataFrame. Each value will be Quantity if the column has units."""
        data = {}
        for col in table.colnames:
            col_data = table[col]
            if hasattr(col_data, 'unit') and col_data.unit is not None:
                data[col] = [val * col_data.unit for val in col_data]
            else:
                data[col] = col_data.tolist()
        return pd.DataFrame(data)

    @staticmethod
    def astropy_table_to_dict_list(table: Table) -> List[Dict[str, Any]]:
        """Convert Astropy Table to list of dicts. Values preserve units as Quantities."""
        rows = []
        for row in table:
            row_dict = {}
            for col in table.colnames:
                val = row[col]
                if hasattr(table[col], 'unit') and table[col].unit is not None:
                    row_dict[col] = val * table[col].unit
                else:
                    row_dict[col] = val
            rows.append(row_dict)
        return rows

    @staticmethod
    def dict_list_to_astropy_table(dict_list: List[Dict[str, Any]]) -> Table:
        """Convert list of dicts to Astropy Table. Quantities are recognized and units applied."""
        if not dict_list:
            return Table()
        columns = {}
        for key in dict_list[0].keys():
            col_values = [d[key] for d in dict_list]
            if isinstance(col_values[0], Quantity):
                columns[key] = u.Quantity(col_values)
            else:
                columns[key] = col_values
        return Table(columns)

    @staticmethod
    def astropy_table_to_list_dict(table: Table) -> Dict[str, List[Any]]:
        """Convert Astropy Table to dict of lists. Preserves units using Quantities."""
        result = {}
        for col in table.colnames:
            col_data = table[col]
            if hasattr(col_data, 'unit') and col_data.unit is not None:
                result[col] = [val * col_data.unit for val in col_data]
            else:
                result[col] = col_data.tolist()
        return result

    @staticmethod
    def list_dict_to_astropy_table(list_dict: Dict[str, List[Any]]) -> Table:
        """Convert dict of lists to Astropy Table. Quantities are recognized and units applied."""
        if not list_dict:
            return Table()
        lengths = {len(v) for v in list_dict.values()}
        if len(lengths) != 1:
            raise ValueError("All lists must have the same length.")
        columns = {}
        for key, values in list_dict.items():
            if isinstance(values[0], Quantity):
                columns[key] = u.Quantity(values)
            else:
                columns[key] = values
        return Table(columns)
    
    @staticmethod
    def df_to_dict_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert a DataFrame to a list of dictionaries."""
        if df.empty:
            return []
        return df.to_dict(orient='records')
    
    @staticmethod
    def dict_list_to_df(dict_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert a list of dictionaries to a DataFrame."""
        if not dict_list:
            return pd.DataFrame()
        return pd.DataFrame(dict_list)
    
    @staticmethod
    def dict_list_to_list_dict(dict_list: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        if not dict_list:
            return {}
        keys = dict_list[0].keys()
        return {key: [d[key] for d in dict_list] for key in keys}

    @staticmethod
    def list_dict_to_dict_list(list_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        if not list_dict:
            return []
        lengths = {len(v) for v in list_dict.values()}
        if len(lengths) != 1:
            raise ValueError("All lists in the dictionary must have the same length.")
        keys = list_dict.keys()
        length = next(iter(lengths))
        return [{key: list_dict[key][i] for key in keys} for i in range(length)]
    

def expand_quantity_in_dict(data: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """
    遍历 dict 中所有列（list），若某列是 Quantity，则将其展开为元素为 Quantity 的列表。
    例：[1, 2, 3]*u.km -> [1*u.km, 2*u.km, 3*u.km]
    Parameters:
        data: dict of lists，列名对应值列表
    Returns:
        dict: 新的 dict，所有 Quantity 类型已被展开
    """
    result = {}
    for key, values in data.items():
        if isinstance(values, Quantity):
            result[key] = [v * values.unit for v in values.value]
        else:
            result[key] = values
    return result
    

def compress_quantity_in_dict(data: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """
    将 dict 中所有列（list）中元素均为 Quantity 的，压缩为单个 Quantity 列。
    例：[1*u.km, 2*u.km, 3*u.km] → [1, 2, 3]*u.km
    Parameters:
        data: dict of lists，列名对应值列表
    Returns:
        dict: 新的 dict，Quantity 列被压缩为统一 Quantity
    """
    result = {}
    for key, values in data.items():
        if isinstance(values, list) and values and isinstance(values[0], Quantity):
            try:
                result[key] = u.Quantity(values)
            except Exception:
                # 有可能单位不一致，如 [1*u.km, 2*u.m]，则不压缩
                result[key] = values
        else:
            result[key] = values
    return result

class TableColumnManager:
    def __init__(self, table_like):
        self._original = table_like
        self.table = self._parse_table()

    def _parse_table(self):
        """解析 self._original 成 astropy Table"""
        if isinstance(self._original, str) and os.path.isfile(self._original):
            return load_table(self._original)
        elif isinstance(self._original, pd.DataFrame):
            return TableConverter.df_to_astropy_table(self._original)
        elif isinstance(self._original, list) and is_dict_list(self._original):
            return TableConverter.dict_list_to_astropy_table(self._original)
        elif isinstance(self._original, dict) and is_list_dict(self._original):
            return TableConverter.list_dict_to_astropy_table(self._original)
        elif isinstance(self._original, Table):
            return self._original
        else:
            raise ValueError("Unsupported table format.")

    def _save(self):
        """如果输入是路径，则保存到原路径"""
        ext = str(self._original).split('.')[-1].lower()
        if ext in ['csv', 'ecsv']:
            save_astropy_table(self.table, self._original, verbose=False)
            print(f"💾 Table saved to {self._original}")
        else:
            raise ValueError(f"Cannot determine format from extension: '{ext}'")

    def _get_table(self):
        if isinstance(self._original, pd.DataFrame):
            return TableConverter.astropy_table_to_df(self.table)
        elif isinstance(self._original, list) and is_dict_list(self._original):
            return TableConverter.astropy_table_to_dict_list(self.table)
        elif isinstance(self._original, dict) and is_list_dict(self._original):
            return TableConverter.astropy_table_to_list_dict(self.table)
        elif isinstance(self._original, Table):
            return self.table
        else:
            raise ValueError("Unsupported table format.")
        #return self.table

    @staticmethod
    def add_columns(table_like, new_columns: dict, after_column: str = None):
        manager = TableColumnManager(table_like)
        table = manager.table
        n_rows = len(table)

        wrong_length = [
            col for col, val in new_columns.items() if len(val) != n_rows
        ]
        if wrong_length:
            raise ValueError(f"❌ Length mismatch: {wrong_length} (table has {n_rows} rows)")

        insert_at = len(table.columns)
        if after_column:
            if after_column not in table.colnames:
                raise ValueError(f"❌ Column '{after_column}' not found.")
            insert_at = table.colnames.index(after_column) + 1

        for col_name, col_data in new_columns.items():
            table.add_column(col_data, name=col_name, index=insert_at)
            insert_at += 1

        print(f"✅ Added columns: {list(new_columns.keys())}")
        return manager._save() if isinstance(table_like, str) else manager._get_table()

    @staticmethod
    def remove_columns(table_like, columns_to_remove):
        manager = TableColumnManager(table_like)
        table = manager.table

        if isinstance(columns_to_remove, str):
            columns_to_remove = [columns_to_remove]

        missing = [col for col in columns_to_remove if col not in table.colnames]
        if missing:
            raise ValueError(f"❌ These columns do not exist: {missing}")

        table.remove_columns(columns_to_remove)
        print(f"✅ Removed columns: {columns_to_remove}")
        return manager._save() if isinstance(table_like, str) else manager._get_table()
    
def get_caller_filename():
    """
    获取调用者的文件名，支持普通Python脚本和Jupyter Notebook
    
    Returns
    -------
    str
        调用者的文件名
        - 对于普通Python脚本: 返回脚本文件名 (如 'script.py')
        - 对于Jupyter Notebook: 返回notebook文件名 (如 'analysis.ipynb')
        - 如果无法获取: 返回默认名称(Unknown)
    """
    
    frame = inspect.currentframe()
    try:
        # 获取调用者的frame (上一级调用栈)
        caller_frame = frame.f_back
        caller_filename = caller_frame.f_code.co_filename
        
        # 处理jupyter notebook的情况
        if caller_filename.startswith('<ipython-input-') or caller_filename == '<stdin>':
            try:
                # 方法1: 尝试从IPython获取notebook名称
                notebook_path = ipynbname.path()
                return os.path.basename(notebook_path)
            except ImportError:
                try:
                    # 方法2: 尝试从环境变量获取
                    if 'JPY_SESSION_NAME' in os.environ:
                        return os.environ['JPY_SESSION_NAME'] + '.ipynb'
                    else:
                        return "Unknown"
                except:
                    return "Unknown"
            except:
                return "Unknown"
        else:
            # 普通Python脚本
            return os.path.basename(caller_filename)
    finally:
        del frame

def compress_fits(fits_file, remove_original=True):
    fits_file = Path(fits_file)
    compressed_path = fits_file.with_name(fits_file.name + ".fz")
    if compressed_path.exists():
        os.remove(compressed_path)
    try:
        subprocess.run(["fpack", str(fits_file)], check=True)
        if remove_original:
            os.remove(fits_file)
    except subprocess.CalledProcessError as e:
        print("Compression failed:", e)

def ephemeris_keywords():
    # https://astroquery.readthedocs.io/en/latest/api/astroquery.jplhorizons.HorizonsClass.html#astroquery.jplhorizons.HorizonsClass.ephemerides
    return ['targetname', 'M1', 'solar_presence', 'k1', 'interfering_body', 'RA', 'DEC', 'RA_app', 'DEC_app',
            'RA*cos(Dec)_rate', 'DEC_rate', 'AZ', 'EL', 'AZ_rate', 'EL_rate', 'sat_X', 'sat_Y',
            'sat_PANG', 'siderealtime', 'airmass', 'magextinct', 'Tmag', 'Nmag', 'illumination', 'illum_defect',
            'sat_sep', 'sat_vis', 'ang_width', 'PDObsLon', 'PDObsLat', 'PDSunLon', 'PDSunLat',
            'SubSol_ang', 'SubSol_dist', 'NPole_ang', 'NPole_dist', 'EclLon', 'EclLat', 'r', 'r_rate',
            'delta', 'delta_rate', 'lighttime', 'vel_sun', 'vel_obs', 'elong', 'elongFlag',
            'alpha', 'IB_elong', 'IB_illum', 'sat_alpha', 'sunTargetPA', 'velocityPA', 'OrbPlaneAng',
            'constellation', 'TDB-UT', 'ObsEclLon', 'ObsEclLat', 'NPole_RA', 'NPole_DEC', 'GlxLon', 'GlxLat',
            'solartime', 'earth_lighttime', 'RA_3sigma', 'DEC_3sigma', 'SMAA_3sigma', 'SMIA_3sigma', 'Theta_3sigma',
            'Area_3sigma', 'RSS_3sigma', 'r_3sigma', 'r_rate_3sigma', 'SBand_3sigma', 'XBand_3sigma', 'DoppDelay_3sigma',
            'true_anom', 'hour_angle', 'alpha_true', 'PABLon', 'PABLat', 'epoch']

def create_time_array(start, end, step):
    """
    从字符串和 Quantity 创建时间数组
    
    参数:
        start: str, 开始时间字符串 (如 '2022-01-01' 或 '2022-01-01 00:00:00')
        end: str, 结束时间字符串
        step: astropy Quantity, 时间步长 (如 30*u.day, 2*u.hour)
    
    返回:
        astropy.time.Time 数组
    """
    # 将字符串转换为 Time 对象
    start_time = Time(start)
    end_time = Time(end)
    
    # 将 step 转换为 TimeDelta
    step_delta = TimeDelta(step)
    
    # 生成时间数组
    times = []
    current = start_time
    while current <= end_time:
        times.append(current.isot)
        current = current + step_delta
    
    return Time(times)

def get_ephemeris_batch(times, target_id, location=500, orbital_keywords=None, batch_size=50):
    """
    分批获取历表数据的辅助函数
    Parameters
    ----------
    times : array-like 'astropy.time.core.Time' object
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
            eph = Ephem.from_horizons(target_id, location=location, epochs=batch_times)
            results.append(eph)
        except Exception as e:
            if "Ambiguous target name" in str(e):
                print(f"请提供准确的目标ID。错误: {str(e)}")
            else:
                print(f"错误: {str(e)}")
            raise
    # 如果只有一批数据，直接返回
    if len(results) == 1:
        if orbital_keywords is None:
            return results[0]
        else:
            return results[0][orbital_keywords]
    # 使用sbpy的vstack合并结果
    final_eph = results[0]
    for eph in results[1:]:
        final_eph.vstack(eph)
    if orbital_keywords is None:
        return final_eph
    else:
        return final_eph[orbital_keywords]



def target_name_converter(input_value: str, output_type: str = 'all', csv_path: Optional[Union[str, Path]] = None) -> Optional[Union[str, Dict[str, str]]]:
    """
    将彗星的data_folder_name、target_simplified_name或target_full_name相互转换
    
    参数:
        input_value: 输入的彗星名称（可以是三种格式中的任意一种）
        output_type: 输出类型
            - 'all': 返回包含所有三个值的字典（默认）
            - 'data_folder_name': 只返回data_folder_name
            - 'target_simplified_name': 只返回target_simplified_name
            - 'target_full_name': 只返回target_full_name
        csv_file: CSV文件路径，默认为'comets_data.csv'
    
    返回:
        根据output_type返回相应的值或字典，如果未找到则返回None
    
    示例:
        >>> comet_name_converter('1P')
        {'data_folder_name': '1P', 'target_simplified_name': '1P/Halley', 'target_full_name': "1P/Halley (Halley's Comet)"}
        
        >>> comet_name_converter('C/2006 P1', 'data_folder_name')
        'C_2006P1'
        
        >>> comet_name_converter('2I/Borisov (Interstellar comet)', 'target_simplified_name')
        'C/2019 Q4'
    """
    if csv_path is None:
        csv_path = paths.get_subpath(paths.data, 'Swift', 'target_name.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件 '{csv_path}' 不存在")
    # 读取CSV文件
    df = pd.read_csv(csv_path, sep=',')
    
    # 在三列中查找
    mask = (
        (df['data_folder_name'].str.lower() == input_value.lower()) |
        (df['target_simplified_name'].str.lower() == input_value.lower()) |
        (df['target_full_name'].str.lower() == input_value.lower())
    )
    
    result = df[mask]
    
    if result.empty:
        return None
    
    row = result.iloc[0]
    
    if output_type == 'all':
        return row.to_dict()
    elif output_type in ['data_folder_name', 'target_simplified_name', 'target_full_name']:
        return row[output_type]
    else:
        raise ValueError(f"无效的output_type: {output_type}")

def _parse_orbital_line(line):
    """
    将轨道元素字符串解析成列表，自动识别并拆分RA/DEC组合
    
    日期格式: 
    - 2026-Apr-01 00:00 (日期+时间作为一个元素)
    - 2460035.500000000 (儒略日)
    
    RA/DEC模式: 3个数字 + 3个数字（可能带正负号）
    例如: 16 45 53.94 -40 23 03.9
    """
    # 移除 /L 这样的标记
    line = re.sub(r'/[A-Za-z]+', '', line)
    
    # 首先处理日期时间（如果是 YYYY-Mon-DD HH:MM 或 HH:MM:SS 格式，合并为一个元素）
    date_time_pattern = r'(\d{4}-\w{3}-\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)'
    match = re.search(date_time_pattern, line)
    if match:
        datetime_str = match.group(1)
        line = line.replace(datetime_str, '__DATETIME__')
        parts = line.strip().split()
        parts = [datetime_str if p == '__DATETIME__' else p for p in parts]
    else:
        parts = line.strip().split()
    
    result = []
    i = 0
    
    while i < len(parts):
        # 检查是否可能是RA/DEC组合的开始（需要至少6个元素）
        if i + 5 < len(parts):
            try:
                # 检查是否符合RA/DEC模式
                h = float(parts[i])
                m = float(parts[i+1])
                s = float(parts[i+2])
                
                # DEC的度数部分（可能带符号）
                dec_deg_str = parts[i+3]
                dec_m = float(parts[i+4])
                dec_s = float(parts[i+5])
                
                # 更严格的RA检查：
                # 1. 小时应该是整数或接近整数（天文坐标通常如此）
                # 2. 对于RA，第一个数字通常在0-23范围且经常是整数
                # 3. DEC的第一个数字应该带符号或在-90到90范围内
                
                # 检查 RA 的小时是否像真实的小时值（通常是整数或最多1位小数）
                h_is_hour_like = (h == int(h)) or (h * 10 == int(h * 10))
                
                # 检查 DEC 度数是否像真实的度数（带符号或合理范围）
                dec_deg_float = float(dec_deg_str)
                dec_is_deg_like = (
                    dec_deg_str.startswith('+') or 
                    dec_deg_str.startswith('-') or 
                    (-90 <= dec_deg_float <= 90 and '.' not in parts[i+3])  # 度数通常是整数
                )
                
                # 额外检查：分和秒的小数位数（天文坐标通常不会有太多小数位）
                m_decimal_places = len(str(m).split('.')[-1]) if '.' in str(m) else 0
                s_decimal_places = len(str(s).split('.')[-1]) if '.' in str(s) else 0
                
                # RA/DEC 的分秒通常不会有超过2-3位小数
                reasonable_precision = (m_decimal_places <= 3 and s_decimal_places <= 3)
                
                if (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60 and
                    0 <= dec_m < 60 and 0 <= dec_s < 60 and
                    h_is_hour_like and dec_is_deg_like and reasonable_precision):
                    # 这是一个RA/DEC组合
                    ra = f"{parts[i]} {parts[i+1]} {parts[i+2]}"
                    dec = f"{parts[i+3]} {parts[i+4]} {parts[i+5]}"
                    result.append(ra)
                    result.append(dec)
                    i += 6
                    continue
            except (ValueError, IndexError):
                pass
        
        # 不是RA/DEC组合，直接添加当前元素
        result.append(parts[i])
        i += 1
    
    return result

def _format_orbital_dict(data_dict):
    """更灵活的版本，使用配置定义处理方式"""
    
    # 定义 RA/DEC 对
    ra_dec_pairs = [
        ('RA', 'DEC'),
        ('RA_app', 'DEC_app'),
        # 可以添加更多对
    ]
    
    formatted = {}
    processed_keys = set()
    
    # 处理 RA/DEC 对
    for ra_key, dec_key in ra_dec_pairs:
        if ra_key in data_dict and dec_key in data_dict:
            formatted[ra_key] = []
            formatted[dec_key] = []
            
            for ra_str, dec_str in zip(data_dict[ra_key], data_dict[dec_key]):
                try:
                    coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
                    formatted[ra_key].append(coord.ra.degree)
                    formatted[dec_key].append(coord.dec.degree)
                except:
                    formatted[ra_key].append(ra_str)
                    formatted[dec_key].append(dec_str)
            
            processed_keys.update([ra_key, dec_key])
    
    # 处理其余的键
    for key, values in data_dict.items():
        if key in processed_keys:
            continue
            
        if 'Date' in key or 'date' in key:
            # 处理日期
            formatted[key] = []
            for date_str in values:
                try:
                    if ':' in date_str:
                        fmt = "%Y-%b-%d %H:%M:%S" if date_str.count(':') == 2 else "%Y-%b-%d %H:%M"
                        dt = datetime.strptime(date_str, fmt)
                        formatted[key].append(Time(dt, scale='utc').isot)
                    else:
                        formatted[key].append(Time(float(date_str), format='jd', scale='utc').isot)
                except:
                    formatted[key].append(date_str)
        else:
            # 处理数值
            formatted[key] = []
            for value in values:
                if value in ['n.a.', 'n.a']:
                    formatted[key].append(np.nan)
                else:
                    try:
                        formatted[key].append(float(value))
                    except:
                        formatted[key].append(value)
    
    return formatted

def read_horizons_table(jpltabpath):
    """
    Expected JPL title list:
     Date__(UT)__HR:MN     R.A._____(ICRF)_____DEC   dRA*cosD d(DEC)/dt    T-mag   N-mag                
     r        rdot             delta      deldot     S-O-T /r     S-T-O    PsAng   PsAMV  
     Sky_motion  Sky_mot_PA  RelVel-ANG  Lun_Sky_Brt  sky_SNR
    """
    f = open(jpltabpath, 'r')
    jplfile = f.readlines()
    f.close()
    startline = jplfile.index('$$SOE\n') + 1
    endline = jplfile.index('$$EOE\n')
    titleline = startline - 3
    
    titles = jplfile[titleline]
    titles = titles.strip()
    title_list = titles.split(' ')
    title_list = [title for title in title_list if title != '']
    title_list_new = []
    for title in title_list:
        title = title.strip()
        if ('Date__(UT)__' in title) or (title == 'Date_________JDUT'): 
            title_list_new.append('date')
        elif title == 'R.A._____(ICRF)_____DEC': 
            title_list_new.append('RA')
            title_list_new.append('DEC')
        elif title == 'R.A.__(a-apparent)__DEC':
            title_list_new.append('RA_app')
            title_list_new.append('DEC_app')
        elif title == '/r': pass
        else: title_list_new.append(title)

    jpl_dict = {}
    for title in title_list_new:
        jpl_dict[title] = []

    jpltable = []
    for line in jplfile[startline:endline]:
        line_list = []
        line = line.strip()
        line_list = _parse_orbital_line(line)
        jpltable.append(line_list)

    for i, title in enumerate(title_list_new):
        for line_list in jpltable:
            jpl_dict[title].append(line_list[i])
    
    jpl_dict = _format_orbital_dict(jpl_dict)
    return jpl_dict

def get_eph_dict(eph):
    eph_table = Table(eph.table)
    eph_table = remove_table_units(eph_table)
    eph_dict = TableConverter.astropy_table_to_list_dict(eph_table)
    return eph_dict

def parse_date_string(date_str):
    """
    解析各种格式的日期字符串
    
    支持格式:
    - '2025-10-10' (ISO格式)
    - '2025 Oct. 10' 或 '2025 Oct 10'
    - '2025 October 10'
    - '2025-Oct-10'
    """
    if not date_str:
        return None
    
    # 清理字符串
    date_str = date_str.strip()
    
    # 尝试直接用 Time 解析（ISO格式）
    try:
        return Time(date_str)
    except:
        pass

    # 处理 YYYY-MM-DD.fraction （小数天）
    m = re.match(r'^(\d{4}-\d{2}-\d{2})\.(\d+)$', date_str)
    if m:
        base_date = m.group(1)
        frac_day = float("0." + m.group(2))
        return Time(base_date) + frac_day * u.day

    # 处理 "2025 Oct. 10" 这种格式
    # 移除点号
    #date_str = date_str.replace('.', '')
    date_str = re.sub(r'(?<=[A-Za-z])\.', '', date_str)
    
    # 尝试各种格式
    formats_to_try = [
        "%Y %b %d",      # 2025 Oct 10
        "%Y %B %d",      # 2025 October 10
        "%Y-%b-%d",      # 2025-Oct-10
        "%Y/%b/%d",      # 2025/Oct/10
    ]
    
    for fmt in formats_to_try:
        try:
            dt = datetime.strptime(date_str, fmt)
            return Time(dt)
        except:
            continue
    
    # 如果都失败了，抛出错误
    raise ValueError(f"无法解析日期: {date_str}")

def smart_float_format(x):
    abs_x = abs(x)
    # 特别大或特别小：用科学计数法
    if abs_x != 0 and (abs_x < 1e-3 or abs_x > 1e5):
        return f"{x:.3e}"
    # 大于 1000 的普通数：保留三位小数
    elif abs_x >= 10:
        return f"{x:.3f}"
    # 其他情况：保留 4 位有效数字
    else:
        return f"{x:.4g}"  # g 格式自动切换科学计数法和浮点数

def dates_to_plot_dates(date_input):
    """
    将各种日期输入转换为 matplotlib plot_date 数值
    
    参数:
        date_input: 可以是:
            - str: 单个日期字符串
            - Time: 单个 astropy Time 对象
            - list/array: 包含 str 或 Time 对象的列表
            - 混合列表: 同时包含 str 和 Time 对象
    
    返回:
        - 单个输入: float (plot_date 数值)
        - 列表输入: numpy array (plot_date 数值数组)
    """
    
    def convert_single_date(date):
        """转换单个日期为 plot_date"""
        if isinstance(date, Time):
            # 已经是 Time 对象
            dt = date.to_datetime()
        elif isinstance(date, str):
            # 字符串，需要解析
            try:
                # 尝试直接用 Time 解析
                t = Time(date)
            except:
                # 使用 parse_date_string 处理其他格式
                t = parse_date_string(date)
            dt = t.to_datetime()
        elif isinstance(date, datetime):
            # 已经是 datetime 对象
            dt = date
        else:
            raise ValueError(f"不支持的日期类型: {type(date)}")
        
        return float(mdates.date2num(dt))
    
    # 检查输入类型
    if isinstance(date_input, (str, Time, datetime)):
        # 单个输入
        return convert_single_date(date_input)
    
    elif hasattr(date_input, '__iter__'):
        # 可迭代对象（列表、数组等）
        plot_dates = []
        for date in date_input:
            plot_dates.append(convert_single_date(date))
        return plot_dates
    
    else:
        raise ValueError(f"不支持的输入类型: {type(date_input)}")


def write_profile_csv(path, radii, values, errors, comment=None):

    radii = np.asarray(radii)
    values = np.asarray(values)
    errors = np.asarray(errors)

    if not (len(radii) == len(values) == len(errors)):
        raise ValueError("radii, values, errors must have the same length")

    with open(path, "w", newline="") as f:

        if comment is not None:
            for line in comment.split("\n"):
                f.write(f"# {line}\n")

        writer = csv.writer(f)
        writer.writerow(["radii", "values", "errors"])

        for r, v, e in zip(radii, values, errors):
            writer.writerow([r, v, e])


def read_profile_csv(path):

    radii = []
    values = []
    errors = []
    comment_lines = []

    with open(path, "r") as f:

        # 读取comment
        pos = f.tell()
        line = f.readline()

        while line.startswith("#"):
            comment_lines.append(line[1:].strip())
            pos = f.tell()
            line = f.readline()

        f.seek(pos)

        reader = csv.DictReader(f)

        for row in reader:
            radii.append(float(row["radii"]))
            values.append(float(row["values"]))
            errors.append(float(row["errors"]))

    comment = "\n".join(comment_lines)

    return np.array(radii), np.array(values), np.array(errors)
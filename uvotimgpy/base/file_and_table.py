import os
from astropy.io import fits
from datetime import datetime
from astropy.time import Time
import numpy as np
from typing import Union, Tuple, List, Dict, Any, Optional
from astropy.table import Table
import subprocess
from pathlib import Path
import inspect
import os
import ipynbname
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.units import Quantity
from uvotimgpy.config import paths


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
    
    if file_path.suffix.lower() == '.csv' and type == 'df':
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

def target_name_converter(input_value, output_type='all', csv_path=None):
    """
    将彗星的data_folder_name、target_simplified_name或target_full_name相互转换
    
    参数:
        input_value: 输入的彗星名称（可以是三种格式中的任意一种）
        output_type: 输出类型，可选值：
            - 'all': 返回包含所有三个值的字典（默认）
            - 'data_folder_name': 只返回data_folder_name
            - 'target_simplified_name': 只返回target_simplified_name
            - 'target_full_name': 只返回target_full_name
        csv_path: CSV文件路径，默认为'comets_data.csv'
    
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
    # 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件 '{csv_path}' 不存在")
    
    # 读取CSV文件并查找匹配的记录
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            # 检查输入值是否匹配任何一个字段
            if (input_value == row['data_folder_name'] or 
                input_value == row['target_simplified_name'] or 
                input_value == row['target_full_name']):
                
                # 根据output_type返回相应的值
                if output_type == 'all':
                    return {
                        'data_folder_name': row['data_folder_name'],
                        'target_simplified_name': row['target_simplified_name'],
                        'target_full_name': row['target_full_name']
                    }
                elif output_type in ['data_folder_name', 'target_simplified_name', 'target_full_name']:
                    return row[output_type]
                else:
                    raise ValueError(f"无效的output_type: {output_type}")
    
    # 如果没找到匹配的记录
    return None

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
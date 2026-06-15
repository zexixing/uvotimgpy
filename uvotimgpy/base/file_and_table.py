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
        # CSV file
        dtype_dict = {'OBSID': str} if 'OBSID' in pd.read_csv(file_path, nrows=0).columns else {}
        df = pd.read_csv(file_path, dtype=dtype_dict)
        return df
    elif type == 'astropy_table' and file_path.suffix.lower() in ['.ecsv', '.csv']:
        # ECSV file (astropy format)
        table = Table.read(file_path)
        return table 
    else:
        # raise ValueError(f"不支持的文件格式: {file_path.suffix}")
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def is_dict_list(obj) -> bool:
    return (
        isinstance(obj, list)
        and all(isinstance(item, dict) for item in obj)
    )

def is_list_dict(obj) -> bool:
    return (
        isinstance(obj, dict)
        and all(isinstance(v, list) for v in obj.values())
        and len(set(len(v) for v in obj.values())) <= 1  # Optional: all lists have the same length
    )

def remove_table_units(table):
    """
    Remove units from an astropy Table by converting it to a pandas DataFrame.
    """
    # Convert to a pandas DataFrame, which automatically removes units
    df = table.to_pandas()
    
    # Convert back to an astropy Table
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
    Iterate through all columns (lists) in a dict; if a column is a Quantity,
    expand it into a list whose elements are Quantity objects.
    Example: [1, 2, 3]*u.km -> [1*u.km, 2*u.km, 3*u.km]
    Parameters:
        data: dict of lists; column names map to value lists
    Returns:
        dict: new dict where all Quantity columns have been expanded
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
    Compress every dict column (list) whose elements are all Quantity objects
    into a single Quantity column.
    Example: [1*u.km, 2*u.km, 3*u.km] -> [1, 2, 3]*u.km
    Parameters:
        data: dict of lists; column names map to value lists
    Returns:
        dict: new dict where Quantity columns have been compressed into unified Quantity objects
    """
    result = {}
    for key, values in data.items():
        if isinstance(values, list) and values and isinstance(values[0], Quantity):
            try:
                result[key] = u.Quantity(values)
            except Exception:
                # Units may be inconsistent, e.g. [1*u.km, 2*u.m], so do not compress
                result[key] = values
        else:
            result[key] = values
    return result

class TableColumnManager:
    def __init__(self, table_like):
        self._original = table_like
        self.table = self._parse_table()

    def _parse_table(self):
        """Parse self._original into an astropy Table."""
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
        """If the input is a path, save back to the original path."""
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
    Get the caller filename, supporting ordinary Python scripts and Jupyter notebooks.
    
    Returns
    -------
    str
        Caller filename.
        - For ordinary Python scripts: return the script filename, e.g. 'script.py'
        - For Jupyter notebooks: return the notebook filename, e.g. 'analysis.ipynb'
        - If unavailable: return the default name, Unknown
    """
    
    frame = inspect.currentframe()
    try:
        # Get the caller frame, i.e. the previous call stack frame
        caller_frame = frame.f_back
        caller_filename = caller_frame.f_code.co_filename
        
        # Handle the Jupyter notebook case
        if caller_filename.startswith('<ipython-input-') or caller_filename == '<stdin>':
            try:
                # Method 1: try to get the notebook name from IPython
                notebook_path = ipynbname.path()
                return os.path.basename(notebook_path)
            except ImportError:
                try:
                    # Method 2: try to get it from environment variables
                    if 'JPY_SESSION_NAME' in os.environ:
                        return os.environ['JPY_SESSION_NAME'] + '.ipynb'
                    else:
                        return "Unknown"
                except:
                    return "Unknown"
            except:
                return "Unknown"
        else:
            # Ordinary Python script
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
    Create a time array from strings and a Quantity.
    
    Parameters:
        start: str, start time string, e.g. '2022-01-01' or '2022-01-01 00:00:00'
        end: str, end time string
        step: astropy Quantity, time step, e.g. 30*u.day or 2*u.hour
    
    Returns:
        astropy.time.Time array.
    """
    # Convert strings to Time objects
    start_time = Time(start)
    end_time = Time(end)
    
    # Convert step to TimeDelta
    step_delta = TimeDelta(step)
    
    # Generate the time array
    times = []
    current = start_time
    while current <= end_time:
        times.append(current.isot)
        current = current + step_delta
    
    return Time(times)

def get_ephemeris_batch(times, target_id, location=500, orbital_keywords=None, batch_size=50):
    """
    Helper function for fetching ephemeris data in batches.
    Parameters
    ----------
    times : array-like 'astropy.time.core.Time' object
        List of time points.
    orbital_keywords : list
        List of orbital parameter keywords to retrieve.
    batch_size : int
        Number of time points to process per batch.
    Returns
    -------
    Ephem 
        Merged ephemeris data, with the same format as a direct eph[orbital_keywords] call.
    """
    results = []  # Store results from all batches
    # Process in batches
    for i in range(0, len(times), batch_size):
        try:
            batch_times = times[i:min(i + batch_size, len(times))]
            eph = Ephem.from_horizons(target_id, location=location, epochs=batch_times)
            results.append(eph)
        except Exception as e:
            if "Ambiguous target name" in str(e):
                # print(f"请提供准确的目标ID。错误: {str(e)}")
                print(f"Please provide an exact target ID. Error: {str(e)}")
            else:
                # print(f"错误: {str(e)}")
                print(f"Error: {str(e)}")
            raise
    # If there is only one batch, return it directly
    if len(results) == 1:
        if orbital_keywords is None:
            return results[0]
        else:
            return results[0][orbital_keywords]
    # Use sbpy vstack to merge results
    final_eph = results[0]
    for eph in results[1:]:
        final_eph.vstack(eph)
    if orbital_keywords is None:
        return final_eph
    else:
        return final_eph[orbital_keywords]



def target_name_converter(input_value: str, output_type: str = 'all', csv_path: Optional[Union[str, Path]] = None) -> Optional[Union[str, Dict[str, str]]]:
    """
    Convert between a comet's data_folder_name, target_simplified_name, and target_full_name.
    
    Parameters:
        input_value: input comet name, in any of the three supported formats
        output_type: output type
            - 'all': return a dictionary containing all three values (default)
            - 'data_folder_name': return only data_folder_name
            - 'target_simplified_name': return only target_simplified_name
            - 'target_full_name': return only target_full_name
        csv_file: CSV file path; default is 'comets_data.csv'
    
    Returns:
        Value or dictionary corresponding to output_type; return None if no match is found.
    
    Examples:
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
        # raise FileNotFoundError(f"CSV文件 '{csv_path}' 不存在")
        raise FileNotFoundError(f"CSV file '{csv_path}' does not exist")
    # Read the CSV file
    df = pd.read_csv(csv_path, sep=',')
    
    # Search the three columns
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
        # raise ValueError(f"无效的output_type: {output_type}")
        raise ValueError(f"Invalid output_type: {output_type}")

def _parse_orbital_line(line):
    """
    Parse an orbital-element string into a list, automatically recognizing and splitting RA/DEC pairs.
    
    Date formats:
    - 2026-Apr-01 00:00 (date + time as one element)
    - 2460035.500000000 (Julian date)
    
    RA/DEC pattern: 3 numbers + 3 numbers, possibly with a sign.
    Example: 16 45 53.94 -40 23 03.9
    """
    # Remove markers such as /L
    line = re.sub(r'/[A-Za-z]+', '', line)
    
    # First handle date/time; merge YYYY-Mon-DD HH:MM or HH:MM:SS into one element
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
        # Check whether this may be the start of an RA/DEC pair; at least 6 elements are needed
        if i + 5 < len(parts):
            try:
                # Check whether it matches the RA/DEC pattern
                h = float(parts[i])
                m = float(parts[i+1])
                s = float(parts[i+2])
                
                # Degree part of DEC, possibly signed
                dec_deg_str = parts[i+3]
                dec_m = float(parts[i+4])
                dec_s = float(parts[i+5])
                
                # Stricter RA checks:
                # 1. Hours should be integer or near-integer, as astronomical coordinates usually are
                # 2. For RA, the first number is usually in 0-23 and often an integer
                # 3. The first DEC number should be signed or within -90 to 90
                
                # Check whether the RA hour looks like a real hour value, usually integer or at most 1 decimal place
                h_is_hour_like = (h == int(h)) or (h * 10 == int(h * 10))
                
                # Check whether the DEC degrees look like real degrees, with sign or reasonable range
                dec_deg_float = float(dec_deg_str)
                dec_is_deg_like = (
                    dec_deg_str.startswith('+') or 
                    dec_deg_str.startswith('-') or 
                    (-90 <= dec_deg_float <= 90 and '.' not in parts[i+3])  # Degrees are usually integers
                )
                
                # Extra check: decimal places in minutes and seconds; astronomical coordinates usually do not have many
                m_decimal_places = len(str(m).split('.')[-1]) if '.' in str(m) else 0
                s_decimal_places = len(str(s).split('.')[-1]) if '.' in str(s) else 0
                
                # RA/DEC minutes and seconds usually do not have more than 2-3 decimal places
                reasonable_precision = (m_decimal_places <= 3 and s_decimal_places <= 3)
                
                if (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60 and
                    0 <= dec_m < 60 and 0 <= dec_s < 60 and
                    h_is_hour_like and dec_is_deg_like and reasonable_precision):
                    # This is an RA/DEC pair
                    ra = f"{parts[i]} {parts[i+1]} {parts[i+2]}"
                    dec = f"{parts[i+3]} {parts[i+4]} {parts[i+5]}"
                    result.append(ra)
                    result.append(dec)
                    i += 6
                    continue
            except (ValueError, IndexError):
                pass
        
        # Not an RA/DEC pair, so add the current element directly
        result.append(parts[i])
        i += 1
    
    return result

def _format_orbital_dict(data_dict):
    """More flexible version that uses configuration to define processing."""
    
    # Define RA/DEC pairs
    ra_dec_pairs = [
        ('RA', 'DEC'),
        ('RA_app', 'DEC_app'),
        # Add more pairs if needed
    ]
    
    formatted = {}
    processed_keys = set()
    
    # Process RA/DEC pairs
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
    
    # Process the remaining keys
    for key, values in data_dict.items():
        if key in processed_keys:
            continue
            
        if 'Date' in key or 'date' in key:
            # Process dates
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
            # Process numeric values
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
    Parse date strings in various formats.
    
    Supported formats:
    - '2025-10-10' (ISO format)
    - '2025 Oct. 10' or '2025 Oct 10'
    - '2025 October 10'
    - '2025-Oct-10'
    """
    if not date_str:
        return None
    
    # Clean the string
    date_str = date_str.strip()
    
    # Try direct parsing with Time first, for ISO format
    try:
        return Time(date_str)
    except:
        pass

    # Handle YYYY-MM-DD.fraction, i.e. fractional days
    m = re.match(r'^(\d{4}-\d{2}-\d{2})\.(\d+)$', date_str)
    if m:
        base_date = m.group(1)
        frac_day = float("0." + m.group(2))
        return Time(base_date) + frac_day * u.day

    # Handle formats such as "2025 Oct. 10"
    # Remove dots
    #date_str = date_str.replace('.', '')
    date_str = re.sub(r'(?<=[A-Za-z])\.', '', date_str)
    
    # Try various formats
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
    
    # If all attempts fail, raise an error
    # raise ValueError(f"无法解析日期: {date_str}")
    raise ValueError(f"Unable to parse date: {date_str}")

def smart_float_format(x):
    abs_x = abs(x)
    # Very large or very small values: use scientific notation
    if abs_x != 0 and (abs_x < 1e-3 or abs_x > 1e5):
        return f"{x:.3e}"
    # Ordinary numbers greater than 1000: keep three decimal places
    elif abs_x >= 10:
        return f"{x:.3f}"
    # Otherwise, keep 4 significant digits
    else:
        return f"{x:.4g}"  # g format automatically switches between scientific and fixed-point notation

def dates_to_plot_dates(date_input):
    """
    Convert various date inputs to matplotlib plot_date values.
    
    Parameters:
        date_input: can be:
            - str: a single date string
            - Time: a single astropy Time object
            - list/array: a list containing str or Time objects
            - mixed list: contains both str and Time objects
    
    Returns:
        - single input: float (plot_date value)
        - list input: numpy array (plot_date values)
    """
    
    def convert_single_date(date):
        """Convert a single date to plot_date."""
        if isinstance(date, Time):
            # Already a Time object
            dt = date.to_datetime()
        elif isinstance(date, str):
            # String, needs parsing
            try:
                # Try direct parsing with Time
                t = Time(date)
            except:
                # Use parse_date_string for other formats
                t = parse_date_string(date)
            dt = t.to_datetime()
        elif isinstance(date, datetime):
            # Already a datetime object
            dt = date
        else:
            # raise ValueError(f"不支持的日期类型: {type(date)}")
            raise ValueError(f"Unsupported date type: {type(date)}")
        
        return float(mdates.date2num(dt))
    
    # Check input type
    if isinstance(date_input, (str, Time, datetime)):
        # Single input
        return convert_single_date(date_input)
    
    elif hasattr(date_input, '__iter__'):
        # Iterable object, such as a list or array
        plot_dates = []
        for date in date_input:
            plot_dates.append(convert_single_date(date))
        return plot_dates
    
    else:
        # raise ValueError(f"不支持的输入类型: {type(date_input)}")
        raise ValueError(f"Unsupported input type: {type(date_input)}")


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

        # Read comments
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

def classify_time_groups(midtime_list, epoch_gap_days=None, orbit_gap_minutes=None):
    """
    Group a MIDTIME list by epoch and orbit, with automatic detection by default or manual thresholds.

    Automatic mode: sort all adjacent-observation time gaps and find ratio jumps
    (ratio >= 3) as boundaries.
    If two jump levels are found, the larger one is the epoch boundary and the
    smaller one is the orbit boundary.
    If only one level is found, inspect subgroup structure to decide whether it
    is an epoch or orbit boundary.

    Args:
        midtime_list: list of MIDTIME strings
        epoch_gap_days: epoch gap threshold in days; None means automatic detection
        orbit_gap_minutes: orbit gap threshold in minutes; None means automatic detection

    Returns:
        tuple: (epoch_list, orbit_list, orbit_index_list)
            - epoch_list: date strings for epoch midpoint times, e.g. '2025-08-18'
            - orbit_list: strings for orbit midpoint times, e.g. '2025-08-18T10:48:31'
            - orbit_index_list: orbit index within the epoch, starting from 1
    """
    from datetime import datetime

    n = len(midtime_list)
    if n == 0:
        return [], [], []

    def parse_time(t):
        t_parsed = parse_date_string(t)
        return t_parsed.to_datetime()
        #for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S'):
        #    try:
        #        return datetime.strptime(t, fmt)
        #    except ValueError:
        #        continue
        #raise ValueError(f"Cannot parse time: {t}")

    def get_gaps(indices):
        s = sorted(indices, key=lambda i: times[i])
        return [(times[s[k]] - times[s[k-1]]).total_seconds() for k in range(1, len(s))]

    def find_break(gaps, min_ratio=3.0):
        """Find the largest ratio jump in sorted gaps and return the geometric-mean threshold."""
        pos = sorted(g for g in gaps if g > 0)
        if len(pos) <= 1:
            return None
        best_r, best_i = 0, -1
        for i in range(len(pos) - 1):
            r = pos[i+1] / pos[i]
            if r > best_r:
                best_r, best_i = r, i
        if best_r < min_ratio:
            return None
        return (pos[best_i] * pos[best_i+1]) ** 0.5

    def split_groups(indices, threshold):
        if threshold is None or len(indices) <= 1:
            return [list(indices)]
        s = sorted(indices, key=lambda i: times[i])
        groups = [[s[0]]]
        for k in range(1, len(s)):
            if (times[s[k]] - times[s[k-1]]).total_seconds() > threshold:
                groups.append([s[k]])
            else:
                groups[-1].append(s[k])
        return groups

    def midpoint(indices):
        ts = [times[i] for i in indices]
        lo, hi = min(ts), max(ts)
        return lo + (hi - lo) / 2

    times = [parse_time(t) for t in midtime_list]
    sorted_indices = sorted(range(n), key=lambda i: times[i])

    # --- Determine thresholds ---
    epoch_th = epoch_gap_days * 86400 if epoch_gap_days is not None else None
    orbit_th = orbit_gap_minutes * 60 if orbit_gap_minutes is not None else None

    if epoch_th is None or orbit_th is None:
        all_gaps = get_gaps(sorted_indices)
        pos = sorted(g for g in all_gaps if g > 0)

        if len(pos) > 1:
            ratios = [(pos[i+1] / pos[i], i) for i in range(len(pos) - 1)]
            ratios.sort(reverse=True)
            sig = [(r, i) for r, i in ratios if r >= 3.0]

            if epoch_th is None and orbit_th is None:
                # Fully automatic: find two levels of jumps
                if len(sig) >= 2:
                    i1, i2 = sig[0][1], sig[1][1]
                    t1 = (pos[i1] * pos[i1+1]) ** 0.5
                    t2 = (pos[i2] * pos[i2+1]) ** 0.5
                    epoch_th, orbit_th = max(t1, t2), min(t1, t2)
                elif len(sig) == 1:
                    # One jump level: check whether subgroups contain additional boundaries to determine the level
                    th = (pos[sig[0][1]] * pos[sig[0][1]+1]) ** 0.5
                    temp_groups = split_groups(sorted_indices, th)
                    has_sub = any(
                        find_break(get_gaps(g)) is not None
                        for g in temp_groups if len(g) > 1
                    )
                    if has_sub:
                        epoch_th = th  # Subgroups have internal structure -> this is an epoch boundary
                    else:
                        orbit_th = th  # Subgroups have no internal structure -> this is an orbit boundary, single epoch
            elif epoch_th is None:
                # Orbit threshold is specified; find epoch threshold automatically
                above = sorted(g for g in all_gaps if g > orbit_th)
                if len(above) > 1:
                    epoch_th = find_break(above)

    # --- Group and label ---
    epoch_list = [None] * n
    orbit_list = [None] * n
    orbit_index_list = [None] * n

    for group in split_groups(sorted_indices, epoch_th):
        epoch_label = midpoint(group).strftime('%Y-%m-%d')
        for idx in group:
            epoch_list[idx] = epoch_label

        local_orbit_th = orbit_th if orbit_th is not None else find_break(get_gaps(group))
        for oi, orbit in enumerate(split_groups(group, local_orbit_th), 1):
            orbit_label = midpoint(orbit).strftime('%Y-%m-%dT%H:%M:%S')
            for idx in orbit:
                orbit_list[idx] = orbit_label
                orbit_index_list[idx] = oi

    return epoch_list, orbit_list, orbit_index_list

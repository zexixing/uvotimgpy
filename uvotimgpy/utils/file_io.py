import os

def process_astropy_table(data_table, output_path=None, save_format='csv'):
    """
    Process an Astropy table by either saving it to a file or printing it to console.

    Parameters:
    data_table (astropy.table.Table): The Astropy table to process.
    output_path (str, optional): Path for the output file. If None, the table will be printed to console. For saving, absolute is recommended.
    save_format (str, optional): Output file format if saving. Default is 'csv'.
    """
    if output_path is None:
        data_table.pprint(max_lines=-1, max_width=-1)
    else:
        save_astropy_table(data_table, output_path, save_format)

def save_astropy_table(data_table, output_path, save_format='csv'):
    """
    Save an Astropy table to a file.

    Parameters:
    data_table (astropy.table.Table): The Astropy table to save.
    output_path (str): Path for the output file. Absolute path is recommended.
    save_format (str, optional): Output file format. Default is 'csv'.
    """
    if not os.path.isabs(output_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_output_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'output', output_path))
    else:
        full_output_path = output_path
    os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
    data_table.write(full_output_path, format=save_format, overwrite=True)
    print(f"Data saved to: {full_output_path}")

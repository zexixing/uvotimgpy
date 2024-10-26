def normalize_filter_name(filter_input, output_format='filename'):
    """
    Normalize Swift/UVOT filter names to standard format.
    
    Parameters:
    -----------
    filter_input : str or iterable
        Input filter name(s). Can be a single string or an iterable of strings
    output_format : str, optional
        Output format, either 'filename' or 'display' (default: 'filename')
        
    Returns:
    --------
    str or list
        Normalized filter name(s). Returns a string if input is a string,
        or a list if input is an iterable
    """
    filter_map = {
        'uvw1': {'filename': 'uw1', 'display': 'UVW1'},
        'uw1': {'filename': 'uw1', 'display': 'UVW1'},
        'uvw2': {'filename': 'uw2', 'display': 'UVW2'},
        'uw2': {'filename': 'uw2', 'display': 'UVW2'},
        'uvm2': {'filename': 'um2', 'display': 'UVM2'},
        'um2': {'filename': 'um2', 'display': 'UVM2'},
        'uuu': {'filename': 'uuu', 'display': 'U'},
        'u': {'filename': 'uuu', 'display': 'U'},
        'uvv': {'filename': 'uvv', 'display': 'V'},
        'v': {'filename': 'uvv', 'display': 'V'},
        'ubb': {'filename': 'ubb', 'display': 'B'},
        'b': {'filename': 'ubb', 'display': 'B'},
        'ugu': {'filename': 'ugu', 'display': 'UV grism'},
        'ugrism': {'filename': 'ugu', 'display': 'UV grism'},
        'uv grism': {'filename': 'ugu', 'display': 'UV grism'},
        'ugv': {'filename': 'ugv', 'display': 'V grism'},
        'vgrism': {'filename': 'ugv', 'display': 'V grism'},
        'v grism': {'filename': 'ugv', 'display': 'V grism'}
    }
    
    # If input is a string, process directly
    if isinstance(filter_input, str):
        return filter_map[filter_input.lower().strip()][output_format]
    
    # If input is iterable, convert each element
    try:
        return [filter_map[f.lower().strip()][output_format] for f in filter_input]
    except AttributeError:
        raise TypeError("Filter input must be a string or an iterable of strings")
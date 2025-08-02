from astropy.modeling import models, fitting
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Tuple
from numbers import Number
from uncertainties import ufloat
from astropy import constants as const
from astropy import units as u
import warnings

class GaussianFitter2D:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    def __init__(self):
        """Initialize 2D Gaussian fitter"""
        pass
        
    @staticmethod
    def validate_param_list(param_name: str, 
                          param_value: Union[None, Number, List, Tuple], 
                          n_gaussians: int,
                          is_position: bool = False) -> List:
        """
        Validate and convert parameter lists
        
        Parameters
        ----------
        param_name : str
            Parameter name for error messages
        param_value : None, float, list, or tuple
            Parameter value
        n_gaussians : int
            Number of Gaussian functions
        is_position : bool
            Whether this is a position parameter (requires special tuple handling)
            
        Returns
        -------
        list
            Converted parameter list
        """
        if param_value is None:
            return [None] * n_gaussians
        
        if is_position:
            # Handle special case for position parameters
            if isinstance(param_value, tuple) and len(param_value) == 2:
                if n_gaussians == 1:
                    return [param_value]
                else:
                    raise ValueError(f"{param_name} must be a list containing {n_gaussians} (col, row) tuples")
            
            # Validate that each element in the list is a valid position tuple
            if not isinstance(param_value, list):
                raise ValueError(f"{param_name} must be a tuple or list of tuples")
                
            if len(param_value) != n_gaussians:
                raise ValueError(f"Length of {param_name} ({len(param_value)}) does not match n_gaussians ({n_gaussians})")
                
            for pos in param_value:
                if not (isinstance(pos, tuple) and len(pos) == 2):
                    raise ValueError(f"Each element in {param_name} must be a (col, row) tuple")
            
            return list(param_value)
        else:
            # Handle single numeric value case
            if isinstance(param_value, Number):
                if n_gaussians == 1:
                    return [param_value]
                else:
                    raise ValueError(f"{param_name} must be a list of length {n_gaussians}")
            
            # Handle list case
            if len(param_value) != n_gaussians:
                raise ValueError(f"Length of {param_name} ({len(param_value)}) does not match n_gaussians ({n_gaussians})")
            
            return list(param_value)

    def fit(self, 
            image: np.ndarray,
            n_gaussians: int = 1,
            threshold: Optional[float] = None,
            sigma_list: Optional[Union[float, List[float]]] = None,
            amplitude_list: Optional[Union[float, List[float]]] = None,
            position_list: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
            theta_list: Optional[Union[float, List[float]]] = None,  # 新增theta参数
            fixed_sigma: bool = False,
            fixed_position: bool = False,
            fixed_amplitude: bool = False,
            fixed_theta: bool = False) -> tuple:  # 新增fixed_theta参数
        """
        Perform 2D Gaussian fitting on image
        
        Additional parameters:
        ----------
        theta_list : float or list, optional
            Initial rotation angle (radians) for each Gaussian function
        fixed_theta : bool
            Whether to fix rotation angle and exclude from fitting
        """
        # Validate parameter lists
        sigma_list = self.validate_param_list("sigma_list", sigma_list, n_gaussians)
        amplitude_list = self.validate_param_list("amplitude_list", amplitude_list, n_gaussians)
        position_list = self.validate_param_list("position_list", position_list, n_gaussians, is_position=True)
        theta_list = self.validate_param_list("theta_list", theta_list, n_gaussians)  # Validate theta list
        
        # Create coordinate grid
        row, col = np.mgrid[:image.shape[0], :image.shape[1]]
        
        # If position parameters not provided, find local maxima as initial guess
        if any(pos is None for pos in position_list):
            from scipy.ndimage import maximum_filter
            local_max = maximum_filter(image, size=3) == image
            if threshold is not None:
                local_max &= image > threshold
            
            coordinates = np.argwhere(local_max)  # [row, col]
            peaks = image[local_max]
            
            # Select n_gaussians strongest peaks
            if len(peaks) < n_gaussians:
                raise ValueError(f"Number of peaks found ({len(peaks)}) is less than requested number of Gaussians ({n_gaussians})")
            
            sorted_indices = np.argsort(peaks)[-n_gaussians:]
            coordinates = coordinates[sorted_indices]
            peaks = peaks[sorted_indices]
        
        # Create model
        model = None
        for i in range(n_gaussians):
            # Set initial parameters
            if position_list[i] is None:
                row_mean, col_mean = coordinates[i]
            else:
                col_mean, row_mean = position_list[i]
                
            amplitude = amplitude_list[i] if amplitude_list[i] is not None else peaks[i]
            sigma = sigma_list[i] if sigma_list[i] is not None else 2.0
            theta = theta_list[i] if theta_list[i] is not None else 0.0  # Default angle is 0
                        
            gaussian = models.Gaussian2D(
                amplitude=amplitude,
                x_mean=col_mean,
                y_mean=row_mean,
                x_stddev=sigma,
                y_stddev=sigma,
                theta=theta  # Add theta parameter
            )
            
            # Set parameter constraints
            gaussian.amplitude.min = 0
            gaussian.x_stddev.min = 0
            gaussian.y_stddev.min = 0
            gaussian.theta.min = -np.pi/2  # theta range constraint
            gaussian.theta.max = np.pi/2
            
            # Fix parameters (if needed)
            if fixed_sigma:
                gaussian.x_stddev.fixed = True
                gaussian.y_stddev.fixed = True
            if fixed_position:
                gaussian.x_mean.fixed = True
                gaussian.y_mean.fixed = True
            if fixed_amplitude:
                gaussian.amplitude.fixed = True
            if fixed_theta:  # New theta fixing option
                gaussian.theta.fixed = True
            
            if model is None:
                model = gaussian
            else:
                model += gaussian
        
        # Add constant background
        model += models.Const2D(amplitude=np.min(image))
        
        # Create fitter
        fitter = fitting.LevMarLSQFitter()
        
        # Execute fitting
        fitted_model = fitter(model, col, row, image)
        
        return fitted_model, fitter

    @staticmethod
    def print_results(fitted_model, fitter=None):
        """
        Print fitting results with improved formatting
        """
        n_gaussians = len(fitted_model.submodel_names) - 1  # Subtract background term

        # Print fitting status information
        if fitter is not None and hasattr(fitter, 'fit_info'):
            print("\nFitting Status:")
            if 'message' in fitter.fit_info:
                print(f"Message: {fitter.fit_info['message']}")
            if 'ierr' in fitter.fit_info:
                print(f"Return Code: {fitter.fit_info['ierr']}")
            if 'nfev' in fitter.fit_info:
                print(f"Function Evaluations: {fitter.fit_info['nfev']}")

        # Print parameters for each Gaussian component
        print("\nFitting Parameters:")
        for i in range(n_gaussians):
            g = fitted_model[i]
            print(f"\nGaussian Component {i+1}:")
            print("─" * 40)  # Separator line
            print(f"Amplitude:     {g.amplitude:10.3f}")
            print(f"Center Position: ({g.x_mean:8.3f}, {g.y_mean:8.3f})")
            print(f"Standard Deviation:   ({g.x_stddev:8.3f}, {g.y_stddev:8.3f})")
            print(f"Rotation Angle: {g.theta:8.3f} rad ({np.degrees(g.theta):8.3f}°)")  # New angle output

        # Print background value
        print("\nBackground:")
        print("─" * 40)
        print(f"Constant Value:   {fitted_model[n_gaussians].amplitude:10.3f}")


    @staticmethod
    def plot_results(image, fitted_model):
        """Visualize fitting results"""
        # Generate model image
        row, col = np.mgrid[:image.shape[0], :image.shape[1]]
        model_image = fitted_model(col, row)
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        im1 = ax1.imshow(image, origin='lower')
        ax1.set_title('Original Data')
        plt.colorbar(im1, ax=ax1)
        
        # Fitting results
        im2 = ax2.imshow(model_image, origin='lower')
        ax2.set_title('Fitted Model')
        plt.colorbar(im2, ax=ax2)
        
        # Residual
        residual = image - model_image
        im3 = ax3.imshow(residual, origin='lower')
        ax3.set_title('Residual')
        plt.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        return fig

# utils/median_error.py

def obtain_n_samples(data: List[np.ndarray], 
                     axis: int = 0) -> Union[np.ndarray, float]:
    """Get number of valid samples"""
    valid_mask_list = [~np.isnan(d) for d in data]
    return np.sum(valid_mask_list, axis=axis)

def calculate_median_error(data: Union[List[float], np.ndarray, List[np.ndarray]], 
                          errors: Optional[Union[List[float], np.ndarray, List[np.ndarray]]] = None,
                          axis: Optional[int] = 0,
                          method: str = 'mean',) -> tuple:
    """Calculate median and its error along specified axis
    
    Parameters
    ----------
    data : array-like
        Input data as ndarray, list of ndarrays, or list of floats
    errors : array-like, optional
        Input errors matching data type and shape
    axis : int, default=0
        Axis along which to compute median
    method : str, default='mean'
        Method to compute median error, 'mean' or 'std'
        
    Returns
    -------
    median_val : ndarray or float
        Computed median values
    median_err : ndarray or float
        Computed median errors
    """
    # Validate method parameter
    if method not in ['mean', 'std']:
        raise ValueError("Method must be either 'mean' or 'std'")
    
    # Convert list of floats to numpy array
    if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
        data = np.array(data)
        if errors is not None:
            errors = np.array(errors)
    
    # Convert list of arrays to 2D array
    if isinstance(data, list) and isinstance(data[0], np.ndarray):
        data = np.array(data)
        if errors is not None:
            errors = np.array(errors)
    
    def mean_error(sum_squared_errors, N):
        return 1.253 * np.sqrt(sum_squared_errors) / N
    
    def std_error(std, N):
        return 1.253 * std / np.sqrt(N)
    
    # Handle axis=None case (flattens the array)
    if axis is None:
        # Flatten both data and errors arrays
        flat_data = data.flatten()
        flat_errors = None if errors is None else errors.flatten()
        
        # Remove NaN values
        valid_mask = ~np.isnan(flat_data)
        valid_data = flat_data[valid_mask]
        valid_errors = None if flat_errors is None else flat_errors[valid_mask]
        
        # If all values are nan, return nan for both median and error
        if len(valid_data) == 0:
            return np.nan, np.nan
        
        # Calculate median
        median_val = np.median(valid_data)
        
        # Calculate error
        if method == 'mean' and valid_errors is not None:
            # Error propagation for median
            N = len(valid_data)
            sum_squared_errors = np.sum(valid_errors**2)
            median_err = mean_error(sum_squared_errors, N)
        elif method == 'std':
            # Standard error of median
            std = np.std(valid_data)
            N = len(valid_data)
            
            if N == 1:
                # Single value case
                if valid_errors is not None:
                    median_err = np.median(valid_errors)
                else:
                    median_err = np.abs(median_val)
            else:
                median_err = std_error(std, N)
        
        return median_val, median_err
    
    # Regular case with specific axis
    # Calculate median ignoring nan values with a check for all-NaN slices
    
    # First check which slices contain only NaNs
    # Create a mask for all-NaN slices
    all_nan_mask = np.all(np.isnan(data), axis=axis)
    
    # Calculate median ignoring nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_val = np.nanmedian(data, axis=axis)
    
    # Set median values to NaN where all-NaN slices were found
    if np.any(all_nan_mask):
        if np.isscalar(median_val):
            if all_nan_mask:
                median_val = np.nan
        else:
            median_val = np.where(all_nan_mask, np.nan, median_val)
    
    # If all values are nan, return nan for both median and error
    if np.all(np.isnan(data)):
        return np.nan, np.nan
    
    if method == 'mean' and errors is not None:
        # Calculate error using error propagation
        # First calculate mean error (sqrt(sum of error square)/N)
        # Then multiply by 1.253 for median error
        valid_mask = ~np.isnan(data)
        N = np.sum(valid_mask, axis=axis)
        squared_errors = np.where(valid_mask, errors**2, 0)
        sum_squared_errors = np.sum(squared_errors, axis=axis)
        
        # Handle divisions by zero (where N = 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_err = mean_error(sum_squared_errors, N)
        
        # Set error to NaN where all-NaN slices were found
        if np.any(all_nan_mask):
            if np.isscalar(median_err):
                if all_nan_mask:
                    median_err = np.nan
            else:
                median_err = np.where(all_nan_mask, np.nan, median_err)
        
    elif method == 'std':
        # Calculate standard deviation ignoring nan values
        # Then calculate median error as 1.253 * std / sqrt(N)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            std = np.nanstd(data, axis=axis)
        
        N = np.sum(~np.isnan(data), axis=axis)
        
        # Initialize median_err with the standard calculation
        # Avoid division by zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_err = std_error(std, N)
        
        # Set error to NaN where all-NaN slices were found
        if np.any(all_nan_mask):
            if np.isscalar(median_err):
                if all_nan_mask:
                    median_err = np.nan
            else:
                median_err = np.where(all_nan_mask, np.nan, median_err)
        
        # Find positions where std is zero (could be because N=1 or all values are identical)
        zero_std_mask = (std == 0)
        
        # Skip positions that have all NaNs
        if not np.isscalar(zero_std_mask) and not np.isscalar(all_nan_mask):
            zero_std_mask = zero_std_mask & ~all_nan_mask
        
        # For positions where std is zero
        if np.any(zero_std_mask):
            # Handle these positions specially
            if errors is not None:
                # Function to get the median error from errors array for a specific position
                def get_error_for_position(pos, pos_values):
                    # Create index for the full data array
                    full_idx = list(pos)
                    # Insert a slice at the axis position to select all values along that axis
                    full_idx.insert(axis, slice(None))
                    
                    # Get errors for this position
                    pos_errors = errors[tuple(full_idx)]
                    # Get mask for non-nan values in data
                    valid_mask = ~np.isnan(data[tuple(full_idx)])
                    # Get valid errors
                    valid_errors = pos_errors[valid_mask]
                    
                    if len(valid_errors) > 0:
                        return np.median(valid_errors)
                    else:
                        # If no valid errors, use the absolute value of the data
                        return np.abs(pos_values)
                
                # Apply to each position where std is zero
                if np.isscalar(zero_std_mask):
                    # Handle scalar case
                    if zero_std_mask:
                        # For a 1D array, directly use the median of errors if available
                        valid_errors = errors[~np.isnan(data)]
                        if len(valid_errors) > 0:
                            median_err = np.median(valid_errors)
                        else:
                            median_err = np.abs(median_val)
                else:
                    # Handle array case
                    for pos in np.ndindex(zero_std_mask.shape):
                        if zero_std_mask[pos]:
                            median_err[pos] = get_error_for_position(list(pos), median_val[pos])
            else:
                # If no errors provided, use the absolute value of median_val
                if np.isscalar(zero_std_mask):
                    # Handle scalar case
                    if zero_std_mask:
                        median_err = np.abs(median_val)
                else:
                    # Handle array case
                    median_err[zero_std_mask] = np.abs(median_val[zero_std_mask])
        
    return median_val, median_err

class ErrorPropagation:
    """Error Propagation Calculation Class"""
    @staticmethod
    def _check_shape_and_convert(x):
        """
        Determine data type and convert to numpy array
        
        Parameters:
        x: float/list/numpy.ndarray
        
        Returns:
        data: numpy.ndarray or float
        shape: int, data dimension
        """
        
        if isinstance(x, (int, float)):
            return x, 0
        elif isinstance(x, list):
            x = np.array(x)
        if isinstance(x, np.ndarray):
            return x, x.shape #len(np.shape(x))
        else:
            raise TypeError("Input must be float, list or numpy.ndarray")

    @staticmethod
    def check_consistency(a, b):
        """
        Check consistency of values and errors
            
        Parameters:
        a: value data
        b: error data
        
        Returns:
        bool: True if consistent, False otherwise
        """
        # check type first
        a, a_shape = ErrorPropagation._check_shape_and_convert(a)
        b, b_shape = ErrorPropagation._check_shape_and_convert(b)
        
        # if the types are not consistent (e.g., one is scalar and the other is array), return False
        if (a_shape == 0) != (b_shape == 0):
            return False
            
        # if both are scalars, return True
        if a_shape == 0 and b_shape == 0:
            return True
            
        # if both are arrays, check if the shapes are the same
        if a_shape != b_shape:
            return False
            
        # check if the nan positions are the same
        nan_positions_a = np.isnan(a)
        nan_positions_b = np.isnan(b)
        return np.array_equal(nan_positions_a, nan_positions_b)
    
    @staticmethod
    def _initialize_inputs(a, a_err, b=None, b_err=None):
        """Initialize and check the consistency of the input data"""
        # check data type and convert
        a, a_shape = ErrorPropagation._check_shape_and_convert(a)
        a_err, _ = ErrorPropagation._check_shape_and_convert(a_err)

        if not (isinstance(b, type(None))) and not (isinstance(b_err, type(None))):
            b, b_shape = ErrorPropagation._check_shape_and_convert(b)
            b_err, _ = ErrorPropagation._check_shape_and_convert(b_err)

            # check consistency
            if not (ErrorPropagation.check_consistency(a, a_err) and ErrorPropagation.check_consistency(b, b_err)):
                raise ValueError("Values and their errors must have consistent shapes and NaN positions")

            # check if the shapes of the two datasets match
            if a_shape != b_shape:
                raise ValueError("Inputs of two values must have the same shape")

            return a, a_err, b, b_err
        else:
            if not ErrorPropagation.check_consistency(a, a_err):
                raise ValueError("Values and their errors must have consistent shapes and NaN positions")
            return a, a_err
            
    @staticmethod
    def add(a, a_err, b, b_err):
        """Addition and error propagation"""
        a, a_err, b, b_err = ErrorPropagation._initialize_inputs(a, a_err, b, b_err)
        return a + b, np.sqrt(a_err**2 + b_err**2)
    
    @staticmethod    
    def subtract(a, a_err, b, b_err):
        """Subtraction and error propagation"""
        a, a_err, b, b_err = ErrorPropagation._initialize_inputs(a, a_err, b, b_err)
        return a - b, np.sqrt(a_err**2 + b_err**2)
    
    @staticmethod
    def multiply(a, a_err, b, b_err):
        """Multiplication and error propagation"""
        a, a_err, b, b_err = ErrorPropagation._initialize_inputs(a, a_err, b, b_err)
        result = a * b
        with np.errstate(divide='ignore', invalid='ignore'):
            error = np.sqrt((a_err*b)**2 + (a*b_err)**2)
        return result, error
    
    @staticmethod
    def divide(a, a_err, b, b_err):
        """Division and error propagation"""
        a, a_err, b, b_err = ErrorPropagation._initialize_inputs(a, a_err, b, b_err)
        result = a / b
        with np.errstate(divide='ignore', invalid='ignore'):
            error = np.sqrt((a_err/b)**2 + (a*b_err/(b**2))**2)
        return result, error
    
    @staticmethod
    def max(err, axis=None):
        """Maximum error"""
        err, _ = ErrorPropagation._check_shape_and_convert(err)
        return np.nanmax(err, axis=axis)
    
    @staticmethod
    def mean(a, a_err, axis=None, ignore_nan=True):
        """
        Calculate the mean and its error

        Parameters:
        a: array_like, input data
        a_err: array_like, error of the input data
        axis: int or None, axis to calculate the mean
        ignore_nan: bool, whether to ignore nan values

        Returns:
        mean_value: float or ndarray, mean value
        mean_error: float or ndarray, error of the mean value
        """
        # check data consistency
        a, a_err = ErrorPropagation._initialize_inputs(a, a_err)

        if ignore_nan:
            # use nanmean and nansum to handle nan
            n = np.sum(~np.isnan(a), axis=axis)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Mean of empty slice')
                mean_value = np.nanmean(a, axis=axis)
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_error = np.sqrt(np.nansum(a_err**2, axis=axis)) / n
        else:
            # do not ignore nan - if there is nan in the inputs, the result will be nan
            n = np.sum(np.ones_like(a), axis=axis)
            mean_value = np.mean(a, axis=axis)
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_error = np.sqrt(np.sum(a_err**2, axis=axis)) / n

        return mean_value, mean_error
    
    @staticmethod
    def sum(a, a_err, axis=None, ignore_nan=True):
        """
        Calculate the sum and its error

        Parameters:
        a: array_like, input data
        a_err: array_like, error of the input data
        axis: int or None, axis to calculate the sum
        ignore_nan: bool, whether to ignore nan values

        Returns:
        sum_value: float or ndarray, sum value
        sum_error: float or ndarray, error of the sum value
        """
        a, a_err = ErrorPropagation._initialize_inputs(a, a_err)
        if ignore_nan:
            sum_value = np.nansum(a, axis=axis)
            with np.errstate(divide='ignore', invalid='ignore'):
                sum_error = np.sqrt(np.nansum(a_err**2, axis=axis))
        else:
            sum_value = np.sum(a, axis=axis)
            with np.errstate(divide='ignore', invalid='ignore'):
                sum_error = np.sqrt(np.sum(a_err**2, axis=axis))
        return sum_value, sum_error
    
    @staticmethod
    def median(a, a_err, axis=None, ignore_nan=True, method='mean', mask=True):
        """
        Calculate the median and its error

        Parameters:
        """
        a, a_err = ErrorPropagation._initialize_inputs(a, a_err)

        # check 'method' parameter
        if method not in ['mean', 'std']:
            raise ValueError("method must be 'mean' or 'std'")
        
        if method == 'mean':
            _, mean_error = ErrorPropagation.mean(a, a_err, axis=axis, ignore_nan=ignore_nan)
            median_error = mean_error*1.253

            if ignore_nan:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='All-NaN slice encountered')
                    median_value = np.nanmedian(a, axis=axis)
            else:
                median_value = np.median(a, axis=axis)

        elif method == 'std':
            if ignore_nan:
                median_value = np.nanmedian(a, axis=axis)
                n = np.sum(~np.isnan(a), axis=axis)
                std = np.nanstd(a, axis=axis)
                with np.errstate(divide='ignore', invalid='ignore'):
                    median_error = std * 1.253 / np.sqrt(n)
            else:
                median_value = np.median(a, axis=axis)
                n = np.sum(np.ones_like(a), axis=axis)
                std = np.std(a, axis=axis)
                with np.errstate(divide='ignore', invalid='ignore'):
                    median_error = std * 1.253 / np.sqrt(n)

            if mask:
                max_error = ErrorPropagation.max(a_err, axis=axis)
                mask = np.nanmax(a,axis=axis) == np.nanmin(a,axis=axis)
                if isinstance(median_error, np.ndarray) and isinstance(max_error, np.ndarray):
                    median_error[mask] = max_error[mask]
                    if not ignore_nan:
                        median_error[np.isnan(median_value)] = np.nan
                elif isinstance(median_error, float) and isinstance(max_error, float):
                    if mask:
                        median_error = max_error
                    if not ignore_nan and np.isnan(median_value):
                        median_error = np.nan

        return median_value, median_error

def get_std(a, axis=None, ignore_nan=True, mask=True):
            if ignore_nan:
                std = np.nanstd(a, axis=axis)
            else:
                std = np.std(a, axis=axis)

            if mask:
                mask = np.nanmax(a,axis=axis) == np.nanmin(a,axis=axis)
                if isinstance(std, np.ndarray):
                    std[mask] = np.nan
                elif isinstance(std, float):
                    if mask:
                        std = np.nan
            return std

class UnitConverter:
    """Unit Conversion Tool Class"""
        
    @staticmethod
    def arcsec_to_au(arcsec: float, delta: float) -> float:
        """Convert arcseconds to astronomical units"""
        return arcsec * delta * np.pi / (180 * 3600)
    
    @staticmethod
    def au_to_arcsec(au: float, delta: float) -> float:
        """Convert astronomical units to arcseconds"""
        return au * 3600 * 180 / (np.pi * delta)
    
    @staticmethod
    def au_to_km(au: float) -> float:
        """Convert astronomical units to kilometers"""
        return au * const.au.to(u.km).value  # 1 AU in kilometers
    
    @staticmethod
    def km_to_au(km: float) -> float:
        """Convert kilometers to astronomical units"""
        return km / const.au.to(u.km).value
    
    @staticmethod
    def arcsec_to_km(arcsec, delta):
        """Convert arcseconds to kilometers"""
        return UnitConverter.au_to_km(UnitConverter.arcsec_to_au(arcsec, delta))
    
    @staticmethod
    def km_to_arcsec(km: float, delta: float) -> float:
        """Convert kilometers to arcseconds"""
        return UnitConverter.au_to_arcsec(UnitConverter.km_to_au(km), delta)
    
    @staticmethod
    def arcsec_to_pixel(arcsec: float, scale: float) -> float:
        """
        Convert arcseconds to pixels
        scale: arcsec/pixel
        """
        return arcsec / scale
    
    @staticmethod
    def pixel_to_arcsec(pixel, scale):
        """
        Convert pixels to arcseconds
        scale: arcsec/pixel
        """
        return pixel * scale 
    
def calculate_motion_pa(ra_rate, dec_rate):
    """
    计算速度的position angle
    
    Parameters:
    -----------
    dec_rate : float
        向北（上）的速度分量，可以为负. arcsec/hr
    ra_rate : float
        向东（左）的速度分量，可以为负. arcsec/hr
    
    Returns:
    --------
    pa : float
        Position angle in degrees [0, 360)
    """
    # 使用atan2计算角度
    # atan2(x, y) 给出从正y轴（北）逆时针到向量的角度
    pa_rad = np.arctan2(ra_rate, dec_rate)
    
    # 转换为角度
    pa_deg = np.degrees(pa_rad)
    pa_deg = pa_deg % 360
    
    return pa_deg
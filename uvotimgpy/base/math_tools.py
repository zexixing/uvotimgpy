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
        """初始化2D高斯拟合器"""
        pass
        
    @staticmethod
    def validate_param_list(param_name: str, 
                          param_value: Union[None, Number, List, Tuple], 
                          n_gaussians: int,
                          is_position: bool = False) -> List:
        """
        验证和转换参数列表
        
        Parameters
        ----------
        param_name : str
            参数名称，用于错误信息
        param_value : None, float, list, or tuple
            参数值
        n_gaussians : int
            高斯函数的数量
        is_position : bool
            是否是位置参数（需要特殊处理元组）
            
        Returns
        -------
        list
            转换后的参数列表
        """
        if param_value is None:
            return [None] * n_gaussians
        
        if is_position:
            # 处理位置参数的特殊情况
            if isinstance(param_value, tuple) and len(param_value) == 2:
                if n_gaussians == 1:
                    return [param_value]
                else:
                    raise ValueError(f"{param_name}必须是包含{n_gaussians}个(col, row)元组的列表")
            
            # 验证列表中的每个元素是否为有效的位置元组
            if not isinstance(param_value, list):
                raise ValueError(f"{param_name}必须是元组或元组列表")
                
            if len(param_value) != n_gaussians:
                raise ValueError(f"{param_name}的长度({len(param_value)})与n_gaussians({n_gaussians})不匹配")
                
            for pos in param_value:
                if not (isinstance(pos, tuple) and len(pos) == 2):
                    raise ValueError(f"{param_name}中的每个元素必须是(col, row)元组")
            
            return list(param_value)
        else:
            # 处理单个数值的情况
            if isinstance(param_value, Number):
                if n_gaussians == 1:
                    return [param_value]
                else:
                    raise ValueError(f"{param_name}必须是长度为{n_gaussians}的列表")
            
            # 处理列表的情况
            if len(param_value) != n_gaussians:
                raise ValueError(f"{param_name}的长度({len(param_value)})与n_gaussians({n_gaussians})不匹配")
            
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
        对图像进行2D高斯拟合
        
        新增参数:
        ----------
        theta_list : float or list, optional
            每个高斯函数的初始旋转角度（弧度）
        fixed_theta : bool
            是否固定旋转角度不参与拟合
        """
        # 验证参数列表
        sigma_list = self.validate_param_list("sigma_list", sigma_list, n_gaussians)
        amplitude_list = self.validate_param_list("amplitude_list", amplitude_list, n_gaussians)
        position_list = self.validate_param_list("position_list", position_list, n_gaussians, is_position=True)
        theta_list = self.validate_param_list("theta_list", theta_list, n_gaussians)  # 验证theta列表
        
        # 创建坐标网格
        row, col = np.mgrid[:image.shape[0], :image.shape[1]]
        
        # 如果没有提供位置参数，查找局部最大值作为初始猜测
        if any(pos is None for pos in position_list):
            from scipy.ndimage import maximum_filter
            local_max = maximum_filter(image, size=3) == image
            if threshold is not None:
                local_max &= image > threshold
            
            coordinates = np.argwhere(local_max)  # [row, col]
            peaks = image[local_max]
            
            # 选择n_gaussians个最强的峰
            if len(peaks) < n_gaussians:
                raise ValueError(f"找到的峰值数量({len(peaks)})少于请求的高斯函数数量({n_gaussians})")
            
            sorted_indices = np.argsort(peaks)[-n_gaussians:]
            coordinates = coordinates[sorted_indices]
            peaks = peaks[sorted_indices]
        
        # 创建模型
        model = None
        for i in range(n_gaussians):
            # 设置初始参数
            if position_list[i] is None:
                row_mean, col_mean = coordinates[i]
            else:
                col_mean, row_mean = position_list[i]
                
            amplitude = amplitude_list[i] if amplitude_list[i] is not None else peaks[i]
            sigma = sigma_list[i] if sigma_list[i] is not None else 2.0
            theta = theta_list[i] if theta_list[i] is not None else 0.0  # 默认角度为0
                        
            gaussian = models.Gaussian2D(
                amplitude=amplitude,
                x_mean=col_mean,
                y_mean=row_mean,
                x_stddev=sigma,
                y_stddev=sigma,
                theta=theta  # 添加theta参数
            )
            
            # 设置参数约束
            gaussian.amplitude.min = 0
            gaussian.x_stddev.min = 0
            gaussian.y_stddev.min = 0
            gaussian.theta.min = -np.pi/2  # theta的范围约束
            gaussian.theta.max = np.pi/2
            
            # 固定参数（如果需要）
            if fixed_sigma:
                gaussian.x_stddev.fixed = True
                gaussian.y_stddev.fixed = True
            if fixed_position:
                gaussian.x_mean.fixed = True
                gaussian.y_mean.fixed = True
            if fixed_amplitude:
                gaussian.amplitude.fixed = True
            if fixed_theta:  # 新增theta的固定选项
                gaussian.theta.fixed = True
            
            if model is None:
                model = gaussian
            else:
                model += gaussian
        
        # 添加常数背景
        model += models.Const2D(amplitude=np.min(image))
        
        # 创建拟合器
        fitter = fitting.LevMarLSQFitter()
        
        # 执行拟合
        fitted_model = fitter(model, col, row, image)
        
        return fitted_model, fitter

    @staticmethod
    def print_results(fitted_model, fitter=None):
        """
        打印拟合结果，格式化和改进输出效果
        """
        n_gaussians = len(fitted_model.submodel_names) - 1  # 减去背景项

        # 打印拟合状态信息
        if fitter is not None and hasattr(fitter, 'fit_info'):
            print("\n拟合状态:")
            if 'message' in fitter.fit_info:
                print(f"信息: {fitter.fit_info['message']}")
            if 'ierr' in fitter.fit_info:
                print(f"返回码: {fitter.fit_info['ierr']}")
            if 'nfev' in fitter.fit_info:
                print(f"函数评估次数: {fitter.fit_info['nfev']}")

        # 打印每个高斯分量的参数
        print("\n拟合参数:")
        for i in range(n_gaussians):
            g = fitted_model[i]
            print(f"\n高斯分量 {i+1}:")
            print("─" * 40)  # 分隔线
            print(f"振幅:     {g.amplitude:10.3f}")
            print(f"中心位置: ({g.x_mean:8.3f}, {g.y_mean:8.3f})")
            print(f"标准差:   ({g.x_stddev:8.3f}, {g.y_stddev:8.3f})")
            print(f"旋转角度: {g.theta:8.3f} rad ({np.degrees(g.theta):8.3f}°)")  # 新增角度输出

        # 打印背景值
        print("\n背景:")
        print("─" * 40)
        print(f"常数值:   {fitted_model[n_gaussians].amplitude:10.3f}")


    @staticmethod
    def plot_results(image, fitted_model):
        """可视化拟合结果"""
        # 生成模型图像
        row, col = np.mgrid[:image.shape[0], :image.shape[1]]
        model_image = fitted_model(col, row)
        
        # 创建图形
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        im1 = ax1.imshow(image, origin='lower')
        ax1.set_title('Original Data')
        plt.colorbar(im1, ax=ax1)
        
        # 拟合结果
        im2 = ax2.imshow(model_image, origin='lower')
        ax2.set_title('Fitted Model')
        plt.colorbar(im2, ax=ax2)
        
        # 残差
        residual = image - model_image
        im3 = ax3.imshow(residual, origin='lower')
        ax3.set_title('Residual')
        plt.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        return fig

# utils/median_error.py

def obtain_n_samples(data: List[np.ndarray], 
                     axis: int = 0) -> Union[np.ndarray, float]:
    """获取有效样本数"""
    valid_mask_list = [~np.isnan(d) for d in data]
    return np.sum(valid_mask_list, axis=axis)

def calculate_median_error(data: Union[List[float], np.ndarray, List[np.ndarray]], 
                          errors: Optional[Union[List[float], np.ndarray, List[np.ndarray]]] = None,
                          axis: Optional[int] = 0,
                          method: str = 'mean',
                          n_bootstrap: int = 1000) -> tuple:
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
  

#class ErrorPropagation:
#    """误差传播计算类"""
#    
#    @staticmethod
#    def propagate(func, *args, derivatives=None):
#        values = [arg[0] for arg in args]
#        errors = [arg[1] for arg in args]
#        """统一处理数组和标量的误差传播"""
#        if derivatives is None:
#            # 使用 uncertainties 包计算
#            
#            # 如果是数组，需要向量化处理
#            if any(isinstance(v, np.ndarray) for v in values):
#                # 创建向量化函数
#                make_ufloat = np.frompyfunc(lambda v, e: ufloat(v, e), 2, 1)
#                get_nominal = np.frompyfunc(lambda x: x.nominal_value, 1, 1)
#                get_std = np.frompyfunc(lambda x: x.std_dev, 1, 1)
#
#                uarrays = [make_ufloat(v, e) for v, e in zip(values, errors)]
#                
#                # 向量化原始函数
#                vec_func = np.frompyfunc(lambda *args: func(*args), len(uarrays), 1)
#                
#                # 计算结果
#                result = vec_func(*uarrays)
#                
#                # 提取标称值和标准差
#                result_values = get_nominal(result).astype(float)
#                result_errors = get_std(result).astype(float)
#            else:
#                # 标量计算
#                uvals = [ufloat(v, e) for v, e in zip(values, errors)]
#                result = func(*uvals)
#                result_values = result.nominal_value
#                result_errors = result.std_dev
#                
#        else:
#            # 使用自定义导数计算
#            result_values = func(*values)
#            partial_derivatives = derivatives(*values)
#            
#            # 计算每个项的平方项
#            squared_terms = []
#            for deriv, err in zip(partial_derivatives, errors):
#                term = np.multiply(deriv, err)
#                squared_term = np.multiply(term, term)
#                squared_terms.append(squared_term)
#            
#            # 计算平方和和平方根
#            sum_squares = np.nansum(squared_terms, axis=0)
#            result_errors = np.sqrt(sum_squares)
#
#            all_nan_mask = np.all(np.isnan(values), axis=0)
#            if np.any(all_nan_mask):
#                if np.isscalar(result_values):
#                    if all_nan_mask:
#                        result_values = np.nan
#                        result_errors = np.nan
#                else:
#                    result_values = np.where(all_nan_mask, np.nan, result_values)
#                    result_errors = np.where(all_nan_mask, np.nan, result_errors)
#            
#            all_nan_mask = np.all(np.isnan(squared_terms), axis=0)
#            if np.any(all_nan_mask):
#                if np.isscalar(result_errors):
#                    if all_nan_mask:
#                        result_errors = np.nan
#                else:
#                    result_errors = np.where(all_nan_mask, np.nan, result_errors)
#            
#        return result_values, result_errors
#    
#
#    @staticmethod
#    def add(*args):
#        """加法误差传播
#        
#        Parameters
#        ----------
#        *args : tuple
#            每个参数是一个 (value, error) 元组，支持数组和带单位的量
#            
#        Returns
#        -------
#        tuple
#            (result_value, result_error)
#        """
#        def add_func(*values):
#            return np.nansum(values, axis=0)
#            
#        def add_derivatives(*values):
#            return [1] * len(values)  # 加法的偏导数都是1
#        
#        return ErrorPropagation.propagate(add_func, *args, derivatives=add_derivatives)
#    
#    @staticmethod
#    def multiply(*args):
#        """乘法误差传播"""
#        def multiply_func(*values):  # 改为接收可变参数
#            return np.nanprod(values, axis=0)
#            
#        def multiply_derivatives(*values):
#            return [
#                np.nanprod(values[:i] + values[i+1:], axis=0)
#                for i in range(len(values))
#            ]
#        
#        return ErrorPropagation.propagate(multiply_func, *args, derivatives=multiply_derivatives)
#    
#    @staticmethod
#    def subtract(*args):
#        """减法误差传播 (a1 - a2 - a3 - ...)"""
#        def subtract_func(*values):
#            return values[0] - np.nansum(values[1:], axis=0)
#            
#        def subtract_derivatives(*values):
#            return [1] + [-1] * (len(values)-1)  # 第一个是1，其他都是-1
#        
#        return ErrorPropagation.propagate(subtract_func, *args, derivatives=subtract_derivatives)
#    
#    @staticmethod
#    def divide(*args):
#        """除法误差传播 (a1 / a2 / a3 / ...)"""
#        def divide_func(*values):
#            return values[0] / np.nanprod(values[1:], axis=0)
#            
#        def divide_derivatives(*values):
#            prod_others = np.nanprod(values[1:], axis=0)  # 所有除数的乘积
#            return [
#                1/prod_others,  # 对被除数的偏导数
#                *[-values[0]/(val * prod_others)  # 对每个除数的偏导数
#                  for val in values[1:]]
#            ]
#        
#        return ErrorPropagation.propagate(divide_func, *args, derivatives=divide_derivatives)
#
#    @staticmethod
#    def median(*args, axis=0, method='mean'):
#        """中位数误差计算
#        
#        Parameters
#        ----------
#        *args : tuple
#            每个参数是一个 (value, error) 元组，支持数组
#            
#        Returns
#        -------
#        tuple
#            (median_value, median_error)
#        """
#        if len(args) == 1 and isinstance(args[0], tuple) and len(args[0]) == 2:
#            values = args[0][0]
#            errors = args[0][1]
#        else:
#            values = [arg[0] for arg in args]
#            errors = [arg[1] for arg in args]
#        return calculate_median_error(data=values, errors=errors, axis=axis, method=method)
#    
#    def mean(*args, axis=0):
#        if len(args) == 1 and isinstance(args[0], tuple) and len(args[0]) == 2:
#            values = args[0][0]
#            errors = args[0][1]
#        else:
#            values = [arg[0] for arg in args]
#            errors = [arg[1] for arg in args]
#        _, median_error = calculate_median_error(data=values, errors=errors, axis=axis, method='mean')
#        mean_image = np.nanmean(values, axis=axis)
#        mean_error = median_error/1.253
#        return mean_image, mean_error

class ErrorPropagation:
    """误差传播计算类"""
    @staticmethod
    def _check_shape_and_convert(x):
        """
        判断数据类型并统一转换为numpy array
        
        Parameters:
        x: float/list/numpy.ndarray
        
        Returns:
        data: numpy.ndarray 或 float
        shape: int, 数据维度
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
        检查数值和误差的一致性
        
        Parameters:
        a: 数值数据
        a_err: 误差数据
        
        Returns:
        bool: True if consistent, False otherwise
        """
        # 先检查类型
        a, a_shape = ErrorPropagation._check_shape_and_convert(a)
        b, b_shape = ErrorPropagation._check_shape_and_convert(b)
        
        # 如果两者类型不一致（一个是标量一个是数组），返回False
        if (a_shape == 0) != (b_shape == 0):
            return False
            
        # 如果都是标量，返回True
        if a_shape == 0 and b_shape == 0:
            return True
            
        # 如果是数组，检查形状是否相同
        if a_shape != b_shape:
            return False
            
        # 检查nan的位置是否一致
        nan_positions_a = np.isnan(a)
        nan_positions_b = np.isnan(b)
        return np.array_equal(nan_positions_a, nan_positions_b)
    
    @staticmethod
    def _initialize_inputs(a, a_err, b=None, b_err=None):
        """初始化并检查输入数据的一致性"""
        # 检查数据类型并转换
        a, a_shape = ErrorPropagation._check_shape_and_convert(a)
        a_err, _ = ErrorPropagation._check_shape_and_convert(a_err)

        if not (isinstance(b, type(None))) and not (isinstance(b_err, type(None))):
            b, b_shape = ErrorPropagation._check_shape_and_convert(b)
            b_err, _ = ErrorPropagation._check_shape_and_convert(b_err)

            # 检查一致性
            if not (ErrorPropagation.check_consistency(a, a_err) and ErrorPropagation.check_consistency(b, b_err)):
                raise ValueError("Values and their errors must have consistent shapes and NaN positions")

            # 检查两组数据的形状是否匹配
            if a_shape != b_shape:
                raise ValueError("Inputs of two values must have the same shape")

            return a, a_err, b, b_err
        else:
            if not ErrorPropagation.check_consistency(a, a_err):
                raise ValueError("Values and their errors must have consistent shapes and NaN positions")
            return a, a_err
    
    # fun3: 定义4个函数：add, multiply, subtract, divide；每个函数输入a, a_err, b, b_err，对这些数据初始化‘
    # 判断是否是相同数据类型，判断a和a_err，b,和b_err的nan是否在相同位置；返回结果和误差 
            
    @staticmethod
    def add(a, a_err, b, b_err):
        """加法运算及其误差传播"""
        a, a_err, b, b_err = ErrorPropagation._initialize_inputs(a, a_err, b, b_err)
        return a + b, np.sqrt(a_err**2 + b_err**2)
    
    @staticmethod    
    def subtract(a, a_err, b, b_err):
        """减法运算及其误差传播"""
        a, a_err, b, b_err = ErrorPropagation._initialize_inputs(a, a_err, b, b_err)
        return a - b, np.sqrt(a_err**2 + b_err**2)
    
    @staticmethod
    def multiply(a, a_err, b, b_err):
        """乘法运算及其误差传播"""
        a, a_err, b, b_err = ErrorPropagation._initialize_inputs(a, a_err, b, b_err)
        result = a * b
        with np.errstate(divide='ignore', invalid='ignore'):
            error = np.sqrt((a_err*b)**2 + (a*b_err)**2)
        return result, error
    
    @staticmethod
    def divide(a, a_err, b, b_err):
        """除法运算及其误差传播"""
        a, a_err, b, b_err = ErrorPropagation._initialize_inputs(a, a_err, b, b_err)
        result = a / b
        with np.errstate(divide='ignore', invalid='ignore'):
            error = np.sqrt((a_err/b)**2 + (a*b_err/(b**2))**2)
        return result, error
    
    @staticmethod
    def max(err, axis=None):
        """最大值及其误差"""
        err, _ = ErrorPropagation._check_shape_and_convert(err)
        return np.nanmax(err, axis=axis)
    
    # fun4: 定义mean
    @staticmethod
    def mean(a, a_err, axis=None, ignore_nan=True):
        """
        计算平均值及其误差

        Parameters:
        a: array_like, 输入数据
        a_err: array_like, 输入数据的误差
        axis: int or None, 计算平均值的轴
        ignore_nan: bool, 是否忽略nan值

        Returns:
        mean_value: float or ndarray, 平均值
        mean_error: float or ndarray, 平均值的误差
        """
        # 检查数据一致性
        a, a_err = ErrorPropagation._initialize_inputs(a, a_err)

        if ignore_nan:
            # 使用nanmean和nansum直接处理nan
            n = np.sum(~np.isnan(a), axis=axis)
            mean_value = np.nanmean(a, axis=axis)
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_error = np.sqrt(np.nansum(a_err**2, axis=axis)) / n
        else:
            # 不忽略nan，如果有nan则结果为nan
            n = np.sum(np.ones_like(a), axis=axis)
            mean_value = np.mean(a, axis=axis)
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_error = np.sqrt(np.sum(a_err**2, axis=axis)) / n

        return mean_value, mean_error
    
    @staticmethod
    def sum(a, a_err, axis=None, ignore_nan=True):
        """
        计算总和及其误差

        Parameters:
        a: array_like, 输入数据
        a_err: array_like, 输入数据的误差
        axis: int or None, 计算总和的轴
        ignore_nan: bool, 是否忽略nan值

        Returns:
        sum_value: float or ndarray, 总和
        sum_error: float or ndarray, 总和的误差
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
        计算中位数及其误差

        Parameters:
        """
        a, a_err = ErrorPropagation._initialize_inputs(a, a_err)

        # 检查method参数
        if method not in ['mean', 'std']:
            raise ValueError("method must be 'mean' or 'std'")
        
        if method == 'mean':
            _, mean_error = ErrorPropagation.mean(a, a_err, axis=axis, ignore_nan=ignore_nan)
            median_error = mean_error*1.253

            if ignore_nan:
                median_value = np.nanmedian(a, axis=axis)
            else:
                median_value = np.median(a, axis=axis)

        elif method == 'std':
            if ignore_nan:
                median_value = np.nanmedian(a, axis=axis)
                #n = np.sum(~np.isnan(a), axis=axis)
                std = np.nanstd(a, axis=axis)
                with np.errstate(divide='ignore', invalid='ignore'):
                    median_error = std #* 1.253 / np.sqrt(n)
            else:
                median_value = np.median(a, axis=axis)
                #n = np.sum(np.ones_like(a), axis=axis)
                std = np.std(a, axis=axis)
                with np.errstate(divide='ignore', invalid='ignore'):
                    median_error = std #* 1.253 / np.sqrt(n)

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


class UnitConverter:
    """天文单位转换工具类"""
        
    @staticmethod
    def arcsec_to_au(arcsec: float, delta: float) -> float:
        """将角秒转换为天文单位"""
        return arcsec * delta * np.pi / (180 * 3600)
    
    @staticmethod
    def au_to_arcsec(au: float, delta: float) -> float:
        """将天文单位转换为角秒"""
        return au * 3600 * 180 / (np.pi * delta)
    
    @staticmethod
    def au_to_km(au: float) -> float:
        """将天文单位转换为千米"""
        return au * const.au.to(u.km).value  # 1 AU in kilometers
    
    @staticmethod
    def km_to_au(km: float) -> float:
        """将千米转换为天文单位"""
        return km / const.au.to(u.km).value
    
    @staticmethod
    def arcsec_to_km(arcsec, delta):
        """将角秒转换为千米"""
        return UnitConverter.au_to_km(UnitConverter.arcsec_to_au(arcsec, delta))
    
    @staticmethod
    def km_to_arcsec(km: float, delta: float) -> float:
        """将千米转换为角秒"""
        return UnitConverter.au_to_arcsec(UnitConverter.km_to_au(km), delta)
    
    @staticmethod
    def arcsec_to_pixel(arcsec: float, scale: float) -> float:
        """将角秒转换为像素
        """
        return arcsec / scale
    
    @staticmethod
    def pixel_to_arcsec(pixel, scale):
        """将像素转换为角秒
        """
        return pixel * scale 
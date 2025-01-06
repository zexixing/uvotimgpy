from astropy.modeling import models, fitting
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from typing import Union, List, Optional, Tuple
from numbers import Number
from scipy import stats
from uncertainties import ufloat
from uvotimgpy.base.unit_tools import convert_sequence_to_array, QuantitySeparator, UnitPropagator, quantity_wrap
from functools import reduce
from operator import mul

class GaussianFitter2D:
    def __init__(self):
        """初始化2D高斯拟合器"""
        pass
        
    @staticmethod
    def validate_param_list(param_name: str, 
                          param_value: Union[None, Number, List, Tuple, u.Quantity], 
                          n_gaussians: int,
                          is_position: bool = False) -> List:
        """
        验证和转换参数列表
        
        Parameters
        ----------
        param_name : str
            参数名称，用于错误信息
        param_value : None, float, list, tuple, or astropy.units.Quantity
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
            if isinstance(param_value, (Number, u.Quantity)):
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
            sigma_list: Optional[Union[float, List[float], u.Quantity]] = None,
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
            
            # 处理带单位的sigma
            if isinstance(sigma, u.Quantity):
                sigma = sigma.value
            
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
            print(f"振幅:     {g.amplitude.value:10.3f}")
            print(f"中心位置: ({g.x_mean.value:8.3f}, {g.y_mean.value:8.3f})")
            print(f"标准差:   ({g.x_stddev.value:8.3f}, {g.y_stddev.value:8.3f})")
            print(f"旋转角度: {g.theta.value:8.3f} rad ({np.degrees(g.theta.value):8.3f}°)")  # 新增角度输出

        # 打印背景值
        print("\n背景:")
        print("─" * 40)
        print(f"常数值:   {fitted_model[n_gaussians].amplitude.value:10.3f}")


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

class ErrorPropagation:
    """误差传播计算类"""
    @staticmethod
    def _prepare_array_calculation(values, errors):
        """准备数组计算，处理形状并扩展标量
        
        Parameters
        ----------
        values : list
            值列表
        errors : list
            误差列表
            
        Returns
        -------
        tuple
            (shape, array_values, array_errors) 如果有数组
            (None, values, errors) 如果全是标量
        """
        # 获取数组形状
        shape = None
        for val in values:
            if isinstance(val, np.ndarray):
                shape = val.shape
                break
                
        if shape is None:
            return None, values, errors
            
        # 将所有标量扩展为数组
        array_values = []
        array_errors = []
        for val, err in zip(values, errors):
            if not isinstance(val, np.ndarray):
                hasunit = hasattr(val, 'unit')
                val = np.full(shape, val)
                if hasunit:
                    val = val*val.unit
            if not isinstance(err, np.ndarray):
                hasunit = hasattr(err, 'unit')
                err = np.full(shape, err)
                if hasunit:
                    err = err*err.unit
            array_values.append(val)
            array_errors.append(err)
            
        return shape, array_values, array_errors

    @staticmethod
    def _prepare_inputs(args, errors):
        """预处理所有输入，去掉单位
        
        Parameters
        ----------
        args : list
            值列表
        errors : list
            误差列表
            
        Returns
        -------
        tuple
            (处理后的值列表, 处理后的误差列表, 单位)
        """
        # 处理所有序列类型的输入
        processed_args = []
        processed_errors = []
        args_units = []
        errors_units = []
        
        # 处理值
        for arg in args:
            processed_arg = convert_sequence_to_array(arg)
            
            if isinstance(processed_arg, u.Quantity):
                processed_args.append(processed_arg.value)
                args_units.append(processed_arg.unit)
            else:
                processed_args.append(processed_arg)
                args_units.append(u.dimensionless_unscaled)
        # 处理误差
        for err in errors:
            processed_err = convert_sequence_to_array(err)
                
            if isinstance(processed_err, u.Quantity):
                processed_errors.append(processed_err.value)
                errors_units.append(processed_err.unit)

            else:
                processed_errors.append(processed_err)
                errors_units.append(u.dimensionless_unscaled)
            
        return processed_args, processed_errors, args_units, errors_units
    
    @staticmethod
    def _propagate_array(func, values, errors, derivatives=None):
        """处理数组的误差传播"""
        processed_values, processed_errors, values_units, errors_units = ErrorPropagation._prepare_inputs(values, errors)
        
        if derivatives is None:
            # 使用 uncertainties 包计算
            shape = processed_values[0].shape
            
            # 创建 ufloat 向量化函数
            make_ufloat = np.frompyfunc(lambda v, e: ufloat(v, e), 2, 1)
            get_nominal = np.frompyfunc(lambda x: x.nominal_value, 1, 1)
            get_std = np.frompyfunc(lambda x: x.std_dev, 1, 1)
            
            # 创建 ufloat 数组
            uarrays = [make_ufloat(v, e) for v, e in zip(processed_values, processed_errors)]
            
            # 向量化原始函数
            vec_func = np.frompyfunc(lambda *args: func(*args), len(uarrays), 1)
            
            # 计算结果
            result = vec_func(*uarrays)
            
            # 提取标称值和标准差
            result_values = get_nominal(result).astype(float)
            result_errors = get_std(result).astype(float)

            # 获取单位字典
            final_unit = UnitPropagator.propagate(func, *values)
            if final_unit:
                return result_values*final_unit, result_errors*final_unit
            return result_values, result_errors
        else:
            # 使用自定义导数计算
            result_values = quantity_wrap(func, *values)
            partial_derivatives = quantity_wrap(derivatives, *values)
            
            # 计算每个项的平方项
            squared_terms = []
            for deriv, err in zip(partial_derivatives, errors):
                # 计算 deriv * err
                term = quantity_wrap(np.multiply, deriv, err)
                # 计算 (deriv * err)^2
                squared_term = quantity_wrap(np.multiply, term, term)
                squared_terms.append(squared_term)
            
            # 计算平方和，保持数组维度
            sum_squares = quantity_wrap(np.sum, squared_terms, axis=0)
            # 计算平方根
            result_errors = quantity_wrap(np.sqrt, sum_squares)
            
            # 确保返回值带有正确的单位
            return result_values, result_errors

    @staticmethod
    def _propagate_scalar(func, values, errors, derivatives=None):
        """处理标量的误差传播"""
        if derivatives is None:
            # 使用 uncertainties 包计算
            processed_values, processed_errors, values_units, errors_units = ErrorPropagation._prepare_inputs(values, errors)
            uvals = [ufloat(v, e) for v, e in zip(processed_values, processed_errors)]
            result = func(*uvals)
            final_unit = UnitPropagator.propagate(func, *values)
            result_values = result.nominal_value
            result_errors = result.std_dev
            if final_unit:
                return result_values*final_unit, result_errors*final_unit
            else:
                return result_values, result_errors
        else:
            # 使用自定义导数计算
            result_value = quantity_wrap(func, *values)
            partial_derivatives = quantity_wrap(derivatives, *values)
            
            # 计算每个项的平方项
            squared_terms = []
            for deriv, err in zip(partial_derivatives, errors):
                # 计算 deriv * err
                term = quantity_wrap(np.multiply, deriv, err)
                # 计算 (deriv * err)^2
                squared_term = quantity_wrap(np.multiply, term, term)
                squared_terms.append(squared_term)
            
            # 计算平方和
            sum_squares = quantity_wrap(np.sum, squared_terms)
            # 计算平方根
            result_error = quantity_wrap(np.sqrt, sum_squares)
            
            return result_value, result_error
        
    @staticmethod
    def propagate(func, args, errors, derivatives=None, output_unit=None):
        """使用 uncertainties 包或自定义导数计算任意函数的误差传播"""
        # 准备数组计算
        shape, calc_values, calc_errors = ErrorPropagation._prepare_array_calculation(
            args, errors
        )

        # 计算结果
        if shape is not None:
            result_values, result_errors = ErrorPropagation._propagate_array(
                func, calc_values, calc_errors, derivatives
            )
        else:
            result_values, result_errors = ErrorPropagation._propagate_scalar(
                func, calc_values, calc_errors, derivatives
            )
        
        # 添加单位
        return result_values, result_errors


    @staticmethod
    def add(*args, output_unit=None):
        """加法误差传播
        
        Parameters
        ----------
        *args : tuple
            每个参数是一个 (value, error) 元组，支持数组和带单位的量
            
        Returns
        -------
        tuple
            (result_value, result_error)
        """
        def add_func(*values):
            return np.sum(values, axis=0)
            
        def add_derivatives(*values):
            return [1] * len(values)  # 加法的偏导数都是1
        
        values = [arg[0] for arg in args]
        errors = [arg[1] for arg in args]
        return ErrorPropagation.propagate(add_func, values, errors, add_derivatives, output_unit)
    
    @staticmethod
    def multiply(*args, output_unit=None):
        """乘法误差传播"""
        def multiply_func(*values):
            return reduce(mul, values)
            
        def multiply_derivatives(*values):
            return [
                reduce(mul, values[:i] + values[i+1:])
                for i in range(len(values))
            ]
        
        values = [arg[0] for arg in args]
        errors = [arg[1] for arg in args]
        
        result_values, result_errors = ErrorPropagation.propagate(
            multiply_func, values, errors, 
            multiply_derivatives, output_unit
        )
        return result_values, result_errors
    
    @staticmethod
    def subtract(*args, output_unit=None):
        """减法误差传播 (a1 - a2 - a3 - ...)"""
        def subtract_func(*values):
            return values[0] - np.sum(values[1:], axis=0)
            
        def subtract_derivatives(*values):
            return [1] + [-1] * (len(values)-1)  # 第一个是1，其他都是-1
        
        values = [arg[0] for arg in args]
        errors = [arg[1] for arg in args]
        return ErrorPropagation.propagate(subtract_func, values, errors, subtract_derivatives, output_unit)
    
    @staticmethod
    def divide(*args, output_unit=None):
        """除法误差传播 (a1 / a2 / a3 / ...)"""
        def divide_func(*values):
            return values[0] / reduce(mul, values[1:])
            
        def divide_derivatives(*values):
            prod_others = reduce(mul, values[1:])  # 所有除数的乘积
            return [
                1/prod_others,  # 对被除数的偏导数
                *[-values[0]/(val * prod_others)  # 对每个除数的偏导数
                  for val in values[1:]]
            ]
        
        values = [arg[0] for arg in args]
        errors = [arg[1] for arg in args]
        return ErrorPropagation.propagate(divide_func, values, errors, divide_derivatives, output_unit)


# 1. 测试乘法
def test_multiply():
    # 创建2x2测试数组
    values1 = np.array([[1.0, 2.0],
                       [3.0, 4.0]])*u.m
    errors1 = np.array([[0.1, 0.1],
                       [0.1, 0.1]])*u.m
    values2 = np.array([[2.0, 2.0],
                       [2.0, 2.0]])*u.m
    errors2 = np.array([[0.2, 0.2],
                       [0.2, 0.2]])*u.m
    
    # 方法1：使用 multiply 函数
    result_val1, result_err1 = ErrorPropagation.multiply(
        (values1, errors1),
        (values2, errors2)
    )
    print("\n方法1 - multiply函数:")
    print("值:")
    print(result_val1)
    print("误差:")
    print(result_err1)
    
    # 方法2：使用 propagate 和 uncertainties
    def multiply_func(*values):
        return reduce(mul, values)
    
    result_val2, result_err2 = ErrorPropagation.propagate(
        multiply_func,
        [values1, values2],
        [errors1, errors2],
        derivatives=None  # 使用 uncertainties 包
    )
    print("\n方法2 - propagate函数:")
    print("值:")
    print(result_val2)
    print("误差:")
    print(result_err2)
    
    # 打印每个位置的详细计算
    print("\n详细结果:")
    for i in range(2):
        for j in range(2):
            print(f"位置[{i},{j}]:")
            print(f"{values1[i,j]}±{errors1[i,j]} × {values2[i,j]}±{errors2[i,j]} = ", end="")
            print(f"{result_val1[i,j]:.2f}±{result_err1[i,j]:.2f} (方法1)")
            print(f"{' '*len(str(values1[i,j]))}  ×  {' '*len(str(values2[i,j]))}    = ", end="")
            print(f"{result_val2[i,j]:.2f}±{result_err2[i,j]:.2f} (方法2)")

test_multiply()

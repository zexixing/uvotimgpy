from astropy.modeling import models, fitting
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Tuple
from numbers import Number
from uncertainties import ufloat

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

class ErrorPropagation:
    """误差传播计算类"""
    
    @staticmethod
    def propagate(func, *args, derivatives=None):
        values = [arg[0] for arg in args]
        errors = [arg[1] for arg in args]
        """统一处理数组和标量的误差传播"""
        if derivatives is None:
            # 使用 uncertainties 包计算
            
            # 如果是数组，需要向量化处理
            if any(isinstance(v, np.ndarray) for v in values):
                # 创建向量化函数
                make_ufloat = np.frompyfunc(lambda v, e: ufloat(v, e), 2, 1)
                get_nominal = np.frompyfunc(lambda x: x.nominal_value, 1, 1)
                get_std = np.frompyfunc(lambda x: x.std_dev, 1, 1)

                uarrays = [make_ufloat(v, e) for v, e in zip(values, errors)]
                
                # 向量化原始函数
                vec_func = np.frompyfunc(lambda *args: func(*args), len(uarrays), 1)
                
                # 计算结果
                result = vec_func(*uarrays)
                
                # 提取标称值和标准差
                result_values = get_nominal(result).astype(float)
                result_errors = get_std(result).astype(float)
            else:
                # 标量计算
                uvals = [ufloat(v, e) for v, e in zip(values, errors)]
                result = func(*uvals)
                result_values = result.nominal_value
                result_errors = result.std_dev
                
        else:
            # 使用自定义导数计算
            result_values = func(*values)
            partial_derivatives = derivatives(*values)
            
            # 计算每个项的平方项
            squared_terms = []
            for deriv, err in zip(partial_derivatives, errors):
                term = np.multiply(deriv, err)
                squared_term = np.multiply(term, term)
                squared_terms.append(squared_term)
            
            # 计算平方和和平方根
            sum_squares = np.sum(squared_terms, axis=0)
            result_errors = np.sqrt(sum_squares)
            
        return result_values, result_errors
    

    @staticmethod
    def add(*args):
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
        
        return ErrorPropagation.propagate(add_func, *args, derivatives=add_derivatives)
    
    @staticmethod
    def multiply(*args):
        """乘法误差传播"""
        def multiply_func(*values):  # 改为接收可变参数
            return np.prod(values, axis=0)
            
        def multiply_derivatives(*values):
            return [
                np.prod(values[:i] + values[i+1:], axis=0)
                for i in range(len(values))
            ]
        
        return ErrorPropagation.propagate(multiply_func, *args, derivatives=multiply_derivatives)
    
    @staticmethod
    def subtract(*args):
        """减法误差传播 (a1 - a2 - a3 - ...)"""
        def subtract_func(*values):
            return values[0] - np.sum(values[1:], axis=0)
            
        def subtract_derivatives(*values):
            return [1] + [-1] * (len(values)-1)  # 第一个是1，其他都是-1
        
        return ErrorPropagation.propagate(subtract_func, *args, derivatives=subtract_derivatives)
    
    @staticmethod
    def divide(*args):
        """除法误差传播 (a1 / a2 / a3 / ...)"""
        def divide_func(*values):
            return values[0] / np.prod(values[1:], axis=0)
            
        def divide_derivatives(*values):
            prod_others = np.prod(values[1:], axis=0)  # 所有除数的乘积
            return [
                1/prod_others,  # 对被除数的偏导数
                *[-values[0]/(val * prod_others)  # 对每个除数的偏导数
                  for val in values[1:]]
            ]
        
        return ErrorPropagation.propagate(divide_func, *args, derivatives=divide_derivatives)


# 1. 测试乘法
def test_multiply():
    # 创建2x2测试数组
    values1 = np.array([[1.0, 2.0],
                       [3.0, 4.0]])
    errors1 = np.array([[0.1, 0.1],
                       [0.1, 0.1]])
    values2 = np.array([[2.0, 2.0],
                       [2.0, 2.0]])
    errors2 = np.array([[0.2, 0.2],
                       [0.2, 0.2]])
    
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
        return np.prod(values, axis=0)
    
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

def test_divide():
    """测试除法误差传播"""
    # 创建2x2测试数组
    values1 = np.array([[10.0, 20.0],
                       [30.0, 40.0]])
    errors1 = np.array([[1.0, 1.0],
                       [1.0, 1.0]])
    values2 = np.array([[2.0, 2.0],
                       [2.0, 2.0]])
    errors2 = np.array([[0.2, 0.2],
                       [0.2, 0.2]])
    
    # 方法1：使用 divide 函数
    result_val1, result_err1 = ErrorPropagation.divide(
        (values1, errors1),
        (values2, errors2)
    )
    print("\n方法1 - divide函数:")
    print("值:")
    print(result_val1)
    print("误差:")
    print(result_err1)
    
    # 方法2：使用 propagate 和 uncertainties
    def divide_func(*values):
        return values[0] / values[1]
    
    result_val2, result_err2 = ErrorPropagation.propagate(
        divide_func,
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
            print(f"{values1[i,j]}±{errors1[i,j]} ÷ {values2[i,j]}±{errors2[i,j]} = ", end="")
            print(f"{result_val1[i,j]:.2f}±{result_err1[i,j]:.2f} (方法1)")
            print(f"{' '*len(str(values1[i,j]))}  ÷  {' '*len(str(values2[i,j]))}    = ", end="")
            print(f"{result_val2[i,j]:.2f}±{result_err2[i,j]:.2f} (方法2)")

def test_add_subtract():
    """测试加法和减法误差传播"""
    # 创建2x2测试数组
    values1 = np.array([[10.0, 20.0],
                       [30.0, 40.0]])
    errors1 = np.array([[10.0, 20.0],
                       [30.0, 40.0]])
    values2 = np.array([[5.0, 10.0],
                       [15.0, 20.0]])
    errors2 = np.array([[5.0, 10.0],
                       [15.0, 20.0]])
    
    print("\n=== 测试加法 ===")
    # 方法1：使用 add 函数
    add_val1, add_err1 = ErrorPropagation.add(
        (values1, errors1),
        (values2, errors2)
    )
    print("\n方法1 - add函数:")
    print("值:")
    print(add_val1)
    print("误差:")
    print(add_err1)
    
    # 方法2：使用 propagate
    def add_func(*values):
        return values[0] + values[1]
    
    add_val2, add_err2 = ErrorPropagation.propagate(
        add_func,
        [values1, values2],
        [errors1, errors2],
        derivatives=None
    )
    print("\n方法2 - propagate函数:")
    print("值:")
    print(add_val2)
    print("误差:")
    print(add_err2)
    
    print("\n=== 测试减法 ===")
    # 方法1：使用 subtract 函数
    sub_val1, sub_err1 = ErrorPropagation.subtract(
        (values1, errors1),
        (values2, errors2)
    )
    print("\n方法1 - subtract函数:")
    print("值:")
    print(sub_val1)
    print("误差:")
    print(sub_err1)
    
    # 方法2：使用 propagate
    def subtract_func(*values):
        return values[0] - values[1]
    
    sub_val2, sub_err2 = ErrorPropagation.propagate(
        subtract_func,
        [values1, values2],
        [errors1, errors2],
        derivatives=None
    )
    print("\n方法2 - propagate函数:")
    print("值:")
    print(sub_val2)
    print("误差:")
    print(sub_err2)
    

def test_custom_func():
    """测试自定义函数 f(x,y) = x*y + x 的误差传播"""
    print("\n=== 测试自定义函数: f(x,y) = x*y + x ===")
    
    # 定义函数和其导数
    #def custom_func(*values):
    #    x, y = values
    #    return x * y + x
    #    
    #def custom_derivatives(*values):
    #    x, y = values
    #    return [y + 1,  # ∂f/∂x = y + 1
    #           x]      # ∂f/∂y = x
    
    def custom_func(x, y):
        return x * y + x
    
    def custom_derivatives(x, y):
        return [y + 1,  # ∂f/∂x = y + 1
               x]      # ∂f/∂y = x
    
    # 测试1：数组输入
    print("\n--- 测试数组输入 ---")
    x1 = np.array([[1.0, 2.0],
                   [3.0, 4.0]])
    y1 = np.array([[2.0, 2.0],
                   [2.0, 2.0]])
    dx1 = np.array([[0.1, 0.1],
                    [0.1, 0.1]])
    dy1 = np.array([[0.2, 0.2],
                    [0.2, 0.2]])
    
    # 使用 uncertainties 包
    print("\n使用 uncertainties 包:")
    result_val1, result_err1 = ErrorPropagation.propagate(
        custom_func,
        (x1, dx1),
        (y1, dy1)
    )
    print("值:")
    print(result_val1)
    print("误差:")
    print(result_err1)
    
    # 使用自定义导数
    print("\n使用自定义导数:")
    result_val2, result_err2 = ErrorPropagation.propagate(
        custom_func,
        (x1, dx1),
        (y1, dy1),
        derivatives=custom_derivatives
    )
    print("值:")
    print(result_val2)
    print("误差:")
    print(result_err2)
    
    # 测试2：标量输入
    print("\n--- 测试标量输入 ---")
    x2 = 2.0
    y2 = 3.0
    dx2 = 0.2
    dy2 = 0.3
    
    # 使用 uncertainties 包
    print("\n使用 uncertainties 包:")
    result_val3, result_err3 = ErrorPropagation.propagate(
        custom_func,
        (x2, dx2),
        (y2, dy2)
    )
    print(f"值: {result_val3}")
    print(f"误差: {result_err3}")
    
    # 使用自定义导数
    print("\n使用自定义导数:")
    result_val4, result_err4 = ErrorPropagation.propagate(
        custom_func,
        (x2, dx2),
        (y2, dy2),
        derivatives=custom_derivatives
    )
    print(f"值: {result_val4}")
    print(f"误差: {result_err4}")
    
    # 打印详细比较
    print("\n详细比较:")
    print("数组输入:")
    for i in range(2):
        for j in range(2):
            print(f"\n位置[{i},{j}]:")
            print(f"uncertainties包: {result_val1[i,j]:.2f}±{result_err1[i,j]:.2f}")
            print(f"自定义导数:     {result_val2[i,j]:.2f}±{result_err2[i,j]:.2f}")
    
    print("\n标量输入:")
    print(f"uncertainties包: {result_val3:.2f}±{result_err3:.2f}")
    print(f"自定义导数:     {result_val4:.2f}±{result_err4:.2f}")

if __name__ == '__main__':
    test_multiply()
    test_divide()
    test_add_subtract()
    test_custom_func()

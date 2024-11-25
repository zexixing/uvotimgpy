from astropy.modeling import models, fitting
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from typing import Union, List, Optional, Tuple
from numbers import Number

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
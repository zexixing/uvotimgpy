from typing import Tuple, Optional
import numpy as np
from scipy.interpolate import interp1d
from uvotimgpy.utils.image_operation import calc_radial_profile, DistanceMap, profile_to_image
import matplotlib.pyplot as plt
class ImageEnhancer:
    @staticmethod
    def by_azimuthal_median(image: np.ndarray,
                            center: Tuple[float, float],
                            step: float = 1.0,
                            image_err: Optional[np.ndarray] = None,
                            bad_pixel_mask: Optional[np.ndarray] = None,
                            start: Optional[float] = None,
                            end: Optional[float] = None,
                            method: str = 'division',
                            median_err_params: Optional[dict] = None,
                            return_radial_profile: bool = False) -> np.ndarray:
        """
        使用方位角中值剖面增强彗星图像
        method: 'division' or 'subtraction'
        TODO: add angle limit to get radial_profile
        """
        # 获取径向剖面
        radial_profile = calc_radial_profile(
            image=image,
            center=center,
            step=step,
            image_err=image_err,
            bad_pixel_mask=bad_pixel_mask,
            start=start,
            end=end,
            method='median',
            median_err_params=median_err_params
        )
        
        if image_err is not None:
            radii, values, _ = radial_profile
        else:
            radii, values = radial_profile
        
        value_center = image[center[1], center[0]] # center[0]:col, center[1]:row
        # radii = np.insert(radii, 0, 0) # 在radii的第一个位置插入0
        # values = np.insert(values, 0, value_center) # 在values的第一个位置插入value_center
        
        # 创建距离图像
        r = DistanceMap(image, center).get_distance_map()

        # 使用插值函数创建2D模型图像
        model_image = profile_to_image(radii, values, r, fill_value=np.nan, start_r=0, start_value=value_center)

        ## 使用DistanceMap获取索引图
        #distance_map = DistanceMap(image, center)
        #index_map = distance_map.get_index_map(step)

        ## 创建查找表，长度等于index_map的最大值
        #lut = np.full(index_map.max() + 1, np.nan)
        ## 填充values到对应的索引位置
        #lut[0:len(values)] = values

        ## 使用索引图获取model_image
        #model_image = lut[index_map]
        
        # 计算增强后的图像，对于模型为0的位置设为nan
        if method == 'division':
            #enhanced_image = np.where(model_image <= 0, np.nan, image / model_image)
            enhanced_image = image / model_image
        elif method == 'subtraction':
            enhanced_image = image - model_image
        else:
            raise ValueError(f"Unsupported method: {method}")
        if not return_radial_profile:
            return enhanced_image
        else:
            return enhanced_image, radii, values
    
    def by_one_over_rho(image: np.ndarray,
                      center: Tuple[float, float],
                      ) -> np.ndarray:
        r = DistanceMap(image, center).get_distance_map()
        one_over_rho = 1 / r
        enhanced_image = image / one_over_rho
        return enhanced_image

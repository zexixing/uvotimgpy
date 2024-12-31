from typing import Union, Tuple, List, Optional
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from regions import CirclePixelRegion, PixCoord, PixelRegion
from photutils.aperture import ApertureMask
from uvotimgpy.utils.image_operation import RadialProfile, DistanceMap, ImageDistanceCalculator
from uvotimgpy.query import StarCatalogQuery
from uvotimgpy.base.region import RegionConverter, RegionCombiner, RegionSelector
from scipy import ndimage
from skimage import restoration

class StarIdentifier:
    """识别图像中的stars, cosmic rays等需要移除的像素"""
    
    def __init__(self):
        self.last_mask = None  # 存储最近一次识别的mask
    
    def by_comparison(self, image1: np.ndarray, image2: np.ndarray,
                     threshold: float = 0.) -> Tuple[np.ndarray, np.ndarray]:
        """通过比较两张图像识别"""
        diff = image1 - image2
        mask_pos = diff > threshold
        mask_neg = diff < -threshold
        self.last_mask = mask_pos | mask_neg
        return mask_pos, mask_neg # mask是star
    
    def by_sigma_clip(self, image: np.ndarray, sigma: float = 3.,
                     maxiters: Optional[int] = 3,
                     exclude_region: Optional[Union[np.ndarray, ApertureMask, PixelRegion]] = None) -> np.ndarray:
        """用sigma-clip方法识别"""
        mask = np.zeros_like(image, dtype=bool)
        if exclude_region is not None:
            region_mask = RegionConverter.to_bool_array(exclude_region, image.shape)
            valid_pixels = ~region_mask
            clipped = sigma_clip(image[valid_pixels], sigma=sigma, maxiters=maxiters, masked=True)
            mask[valid_pixels] = clipped.mask
        else:
            clipped = sigma_clip(image, sigma=sigma, maxiters=maxiters, masked=True)
            mask = clipped.mask
        self.last_mask = mask
        return mask # mask是star
    
    def by_manual(self, image: np.ndarray, 
                  row_range: Optional[Tuple[int, int]] = None,
                  col_range: Optional[Tuple[int, int]] = None,) -> np.ndarray:
        """手动输入位置识别"""
        print("Creating selector...")
        
        # 确保之前的窗口都已关闭
        plt.close('all')
        
        # 创建选择器
        selector = RegionSelector(image, vmin=0, vmax=2, 
                                row_range=row_range, 
                                col_range=col_range)
        
        print("Getting regions...")
        # 显式调用 show 并等待窗口关闭
        plt.show(block=True)
        
        # 获取区域
        regions = selector.get_regions()
        print("Regions obtained.")#, regions)
        
        if not regions:  # 如果没有选择任何区域
            return np.zeros_like(image, dtype=bool)
        
        # 合并所有选择的区域
        combined_regions = RegionCombiner.union(regions)
        
        # 转换为布尔数组
        mask = RegionConverter.region_to_bool_array(combined_regions, image_shape=image.shape)
        
        self.last_mask = mask
        return mask
    
    def by_catalog(self, image: np.ndarray, wcs: WCS, mag_limit: float = 15,
                  catalog: str = 'GSC', aperture_radius: float = 5) -> np.ndarray:
        """创建恒星掩膜"""

        # 计算图像四个角的天球坐标
        n_rows, n_cols = image.shape
        center = np.array([n_cols/2, n_rows/2])  # (col, row)
        center_sky = wcs.pixel_to_world(center[0], center[1])

        # 计算中心到边的最大距离（像素坐标，假设row和col方向的）
        max_dist = ImageDistanceCalculator.from_edges(image, center, distance_method='max', wcs=wcs)

        radius = 1.1 * max_dist

        # 查询星表
        catalog_query = StarCatalogQuery(center_sky, radius, mag_limit)
        stars, ra_key, dec_key = catalog_query.query(catalog)

        # 创建天球坐标对象，转换坐标并创建掩膜
        coords = SkyCoord(ra=stars[ra_key], dec=stars[dec_key])
        pixel_coords = wcs.world_to_pixel(coords)
        positions = np.array(pixel_coords).T

        # 筛选在图像范围内的星
        valid_stars = (
            (positions[:, 0] >= 0) & 
            (positions[:, 0] < image.shape[1]) & 
            (positions[:, 1] >= 0) & 
            (positions[:, 1] < image.shape[0])
        )
        positions = positions[valid_stars]

        # 创建圆形孔径
        #apertures = CircularAperture(positions, r=aperture_radius)
        # 创建掩膜
        #mask = np.zeros(image.shape, dtype=bool)
        #masks = apertures.to_mask(method='center')

        #for mask_obj in masks:
        #    # 获取掩膜的位置信息
        #    slices = mask_obj.get_overlap_slices(image.shape)
        #    if slices is not None:
        #        data_slc, mask_slc = slices
        #        mask[data_slc] |= mask_obj.data[mask_slc] > 0
        mask = np.zeros(image.shape, dtype=bool)

        centers = PixCoord(positions[:, 0], positions[:, 1])
        circles = [CirclePixelRegion(center=center, radius=aperture_radius) for center in centers]
        combined_regions = RegionCombiner.union(circles)
        mask = RegionConverter.region_to_bool_array(combined_regions, image_shape=image.shape)

        self.last_mask = mask
        return mask

class PixelFiller:
    """填充被标记的像素"""
    
    def by_comparison(self, image1: np.ndarray, image2: np.ndarray,
                     mask_pos: np.ndarray, mask_neg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """用两张图像互相填充"""
        filled1 = image1.copy()
        filled2 = image2.copy()
        
        filled1[mask_pos] = image2[mask_pos]
        filled2[mask_neg] = image1[mask_neg]
        
        return filled1, filled2
    
    def by_rings(self, image: Union[np.ndarray, u.Quantity], mask: np.ndarray, 
                 center: tuple, step: float, 
                 method: str = 'median',
                 start: Optional[float] = None,
                 end: Optional[float] = None) -> np.ndarray:
        """
        按环形区域填充被mask的像素

        Parameters
        ----------
        image : np.ndarray
            输入图像
        mask : np.ndarray
            坏像素掩膜，True表示被mask的像素
        center : tuple
            圆环中心坐标 (col, row)
        step : float
            圆环步长
        method : str
            计算方法，'median'或'mean'
        start, end : float, optional
            圆环的起始和结束半径

        Returns
        -------
        np.ndarray
            填充后的图像
        """
        # 参数检查
        if mask.shape != np.asarray(image).shape:
            raise ValueError("image and mask must have the same shape")
        if method not in ['median', 'mean']:
            raise ValueError("method must be 'median' or 'mean'")
        if step <= 0:
            raise ValueError("step must be positive")

        # 如果没有被mask的像素，直接返回原图
        if not np.any(mask):
            return image.copy()
        
        # 初始化
        filled_image = image.copy()

        # 使用RadialProfile计算每个环的值
        profile = RadialProfile(image, center=center, step=step, bad_pixel_mask=mask,
                                start=start, end=end, method=method)
        radii, values = profile.get_radial_profile()
        dist_map = DistanceMap(image, center)

        # 对每个有效的环进行处理
        for r, v in zip(radii, values):
            # 创建当前环的掩膜
            ring_mask = dist_map.get_range_mask(r-step/2, r+step/2)

            # 找到环内被mask的像素
            masked_pixels = mask & ring_mask

            # 如果环内有被mask的像素，用计算得到的值填充
            if np.any(masked_pixels):
                filled_image[masked_pixels] = v

        return filled_image, radii, values
    
    def _iterative_fill(self, data: np.ndarray, mask: np.ndarray, 
                        filter_func, footprint: np.ndarray) -> np.ndarray:
        """通用的迭代填充函数
        
        Parameters
        ----------
        data : np.ndarray
            需要填充的数据（可以是图像或误差数组）
        mask : np.ndarray
            需要填充的像素掩膜
        filter_func : callable
            用于计算填充值的函数（如np.nanmedian或error_propagation）
        footprint : np.ndarray
            用于定义邻域的结构元素
            
        Returns
        -------
        np.ndarray
            填充后的数组
        """
        # 初始化带NaN的数组
        working_data = data.copy()
        working_data[mask] = np.nan
        
        # 迭代填充直到没有更多变化或达到最大迭代次数
        max_iters = 10
        for _ in range(max_iters):
            filled_values = ndimage.generic_filter(
                working_data,
                function=filter_func,
                footprint=footprint,
                mode='constant',
                cval=np.nan
            )
            
            # 只更新mask的区域
            new_data = working_data.copy()
            new_data[mask] = filled_values[mask]
            
            # 检查是否收敛（是否所有值都被填充）
            if not np.any(np.isnan(new_data[mask])):
                working_data = new_data
                break
                
            # 检查是否有变化
            if np.allclose(new_data, working_data, equal_nan=True):
                break
                
            working_data = new_data
        
        return working_data

    def by_neighbors(self, image: np.ndarray, mask: np.ndarray, 
                    radius: int = 4, method: str = 'nearest',
                    error: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """使用邻近像素填充被mask的像素
        
        Parameters
        ----------
        image : np.ndarray
            输入图像
        mask : np.ndarray
            坏像素掩膜，True表示被mask的像素
        radius : int
            邻域半径（仅在method='nearest'时使用），默认为4
        method : str
            填充方法，可选：
            - 'nearest': 最近邻插值（默认），使用邻域中值填充
            - 'biharmonic': 双调和插值，适合平滑填充
        error : np.ndarray, optional
            输入图像的误差数组。如果提供，将计算填充像素的误差传播
            
        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            如果没有提供error参数，返回填充后的图像；
            如果提供了error参数，返回(filled_image, filled_error)元组
        """
        # 参数检查
        if mask.shape != image.shape:
            raise ValueError("image and mask must have the same shape")
        if method not in ['nearest', 'biharmonic']:
            raise ValueError("method must be one of: 'nearest', 'biharmonic'")
        if error is not None and error.shape != image.shape:
            raise ValueError("error array must have the same shape as image")
        
        # 如果没有被mask的像素，直接返回原图
        if not np.any(mask):
            if error is not None:
                return image.copy(), error.copy()
            return image.copy()
        
        # 初始化输出
        filled_image = image.copy()
        filled_error = error.copy() if error is not None else None
        
        if method == 'nearest':
            footprint = ndimage.generate_binary_structure(2, 1)
            footprint = ndimage.iterate_structure(footprint, radius)
            
            # 使用通用填充函数填充图像
            filled_image = self._iterative_fill(
                image, mask, np.nanmedian, footprint
            )
            
            # 计算误差（如果需要）
            if error is not None:
                filled_error = self._by_neighbors_calculate_error(
                    error, mask, method='nearest', footprint=footprint
                )
            
        else:  # 'biharmonic'
            from skimage.restoration import inpaint
            # 双调和插值，适合平滑填充
            filled_image = inpaint.inpaint_biharmonic(
                image, mask
            )
            
            # 计算误差（如果需要）
            if error is not None:
                filled_error = self._by_neighbors_calculate_error(
                    error, mask, method='biharmonic'
                )
        
        if error is not None:
            return filled_image, filled_error
        return filled_image
    
    def _by_neighbors_calculate_error(self, error: np.ndarray, mask: np.ndarray, 
                                        method: str = 'nearest', **kwargs) -> np.ndarray:
        """计算填充像素的误差传播"""
        if method not in ['nearest', 'biharmonic']:
            raise ValueError("method must be one of: 'nearest', 'biharmonic'")
        
        if method == 'nearest':
            if 'footprint' not in kwargs:
                raise ValueError("footprint is required for nearest method")
            
            def error_propagation(values):
                valid_mask = ~np.isnan(values)
                if not np.any(valid_mask):
                    return np.nan
                # 不再除以n_valid，因为我们使用中值填充
                # 保守估计：使用邻域内误差的均方根
                return np.sqrt(np.nanmean(values[valid_mask]**2))
            
            filled_errors = self._iterative_fill(
                error, mask, error_propagation, kwargs['footprint']
            )
            
        else:  # 'biharmonic'
            # 使用mask边界上的误差的最大值作为填充区域的误差
            boundary_mask = ndimage.binary_dilation(mask) & ~mask
            max_boundary_error = np.max(error[boundary_mask])
            
            filled_errors = error.copy()
            filled_errors[mask] = max_boundary_error
        
        return filled_errors
    
    def by_median_map(self, image: np.ndarray, mask: np.ndarray,
                     median_map: np.ndarray) -> np.ndarray:
        """用median map填充"""
        filled = image.copy()
        filled[mask] = median_map[mask]
        return filled

class BackgroundCleaner:
    """组合StarIdentifier和PixelFiller的高层接口"""
    
    def __init__(self):
        self.identifier = StarIdentifier()
        self.filler = PixelFiller()
    
    def process_single_image(self, image: np.ndarray,
                           identify_method: str = 'sigma_clip',
                           fill_method: str = 'neighbors',
                           **kwargs) -> np.ndarray:
        """处理单张图像的完整流程"""
        # 选择识别方法
        if identify_method == 'sigma_clip':
            mask = self.identifier.by_sigma_clip(image, **kwargs)
        elif identify_method == 'manual':
            mask = self.identifier.by_manual(image, **kwargs)
        else:
            raise ValueError(f"Unsupported identify method: {identify_method}")
        
        # 选择填充方法
        if fill_method == 'neighbors':
            cleaned = self.filler.by_neighbors(image, mask, **kwargs)
        elif fill_method == 'ring':
            cleaned = self.filler.by_ring(image, mask, **kwargs)
        elif fill_method == 'median_map':
            cleaned = self.filler.by_median_map(image, mask, **kwargs)
        else:
            raise ValueError(f"Unsupported fill method: {fill_method}")
        
        return cleaned
    
    def process_image_pair(self, image1: np.ndarray, image2: np.ndarray,
                          threshold: float = 0.) -> Tuple[np.ndarray, np.ndarray]:
        """处理图像对的完整流程"""
        mask_pos, mask_neg = self.identifier.by_comparison(image1, image2, threshold)
        cleaned1, cleaned2 = self.filler.by_comparison(image1, image2, mask_pos, mask_neg)
        return cleaned1, cleaned2

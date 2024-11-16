from typing import Union, Tuple, List, Optional
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from photutils.aperture import CircularAperture
from uvotimgpy.utils.image_operation import RadialProfile, DistanceMap, ImageDistanceCalculator
from uvotimgpy.query import StarCatalogQuery

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
    
    def by_rings(self, image: np.ndarray, center: Tuple[int, int], 
                radii: List[int], threshold: float = 3.) -> np.ndarray:
        """用同心圆环统计识别"""
        # TODO: 实现同心圆环方法
        mask = np.zeros_like(image, dtype=bool)
        self.last_mask = mask
        return mask
    
    def by_sigma_clip(self, image: np.ndarray, sigma: float = 3.,
                     maxiters: Optional[int] = 3) -> np.ndarray:
        """用sigma-clip方法识别"""
        clipped = sigma_clip(image, sigma=sigma, maxiters=maxiters, masked=True)
        self.last_mask = clipped.mask
        return clipped.mask # mask是star
    
    def by_manual(self, image: np.ndarray, positions: List[Tuple[int, int]], 
                 radius: int) -> np.ndarray:
        """手动输入位置识别"""
        # TODO: 实现手动标记方法
        mask = np.zeros_like(image, dtype=bool)
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
        apertures = CircularAperture(positions, r=aperture_radius)

        # 创建掩膜
        mask = np.zeros(image.shape, dtype=bool)
        masks = apertures.to_mask(method='center')

        for mask_obj in masks:
            # 获取掩膜的位置信息
            slices = mask_obj.get_overlap_slices(image.shape)
            if slices is not None:
                data_slc, mask_slc = slices
                mask[data_slc] |= mask_obj.data[mask_slc] > 0


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
        radii, values = profile.compute()
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
    
    def by_neighbors(self, image: np.ndarray, mask: np.ndarray,
                    kernel_size: int = 3) -> np.ndarray:
        """用邻近像素填充"""
        # TODO: 实现邻近像素填充方法
        filled = image.copy()
        return filled
    
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
        elif identify_method == 'rings':
            mask = self.identifier.by_rings(image, **kwargs)
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

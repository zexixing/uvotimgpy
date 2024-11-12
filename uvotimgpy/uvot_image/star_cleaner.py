from typing import Union, Tuple, List, Optional
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from scipy import ndimage
import matplotlib.pyplot as plt

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
                     maxiters: Optional[int] = None) -> np.ndarray:
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
    
    def by_catalog(self, image: np.ndarray, catalog: str, 
                  radius: int) -> np.ndarray:
        """用星表自动识别"""
        # TODO: 实现星表识别方法
        mask = np.zeros_like(image, dtype=bool)
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
    
    def by_ring(self, image: np.ndarray, mask: np.ndarray,
                center: Tuple[int, int], width: int = 2,
                method: str = 'median') -> np.ndarray:
        """用环形区域统计量填充"""
        # TODO: 实现环形填充方法
        filled = image.copy()
        return filled
    
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
from typing import Tuple, Union, Optional
import numpy as np
from photutils.aperture import ApertureMask, BoundingBox

class UnifiedMask:
    def __init__(self, mask_data: Union[np.ndarray, ApertureMask], image_shape: Tuple[int, int] = None):
        """
        统一的掩膜类
        
        Parameters
        ----------
        mask_data : numpy.ndarray 或 ApertureMask
            掩膜数据
        image_shape : tuple, optional
            原始图像的形状，当使用ApertureMask时必须提供
        """
        if isinstance(mask_data, ApertureMask):
            if image_shape is None:
                raise ValueError("image_shape must be provided when using ApertureMask")
            self._mask = mask_data
            self._image_shape = image_shape
            self._is_aperture = True
        else:
            self._mask = np.asarray(mask_data, dtype=bool)
            self._image_shape = mask_data.shape
            self._is_aperture = False
    
    def to_bool_array(self) -> np.ndarray:
        """
        转换为布尔数组
        
        Returns
        -------
        numpy.ndarray
            布尔数组形式的掩膜
        """
        if not self._is_aperture:
            return self._mask
        
        full_mask = np.zeros(self._image_shape, dtype=bool)
        bbox = self._mask.bbox
        yslice = slice(bbox.iymin, bbox.iymax)
        xslice = slice(bbox.ixmin, bbox.ixmax)
        full_mask[yslice, xslice] = self._mask.data > 0
        return full_mask
    
    def to_aperture_mask(self) -> ApertureMask:
        """
        转换为ApertureMask
        
        Returns
        -------
        ApertureMask
            photutils的ApertureMask对象
        """
        if self._is_aperture:
            return self._mask
        
        # 找到掩膜的边界框
        rows, cols = np.where(self._mask)
        if len(rows) == 0:  # 空掩膜
            return ApertureMask(np.array([[False]]), bbox=BoundingBox(0, 1, 0, 1))
        
        # 计算边界框
        ymin, ymax = rows.min(), rows.max() + 1
        xmin, xmax = cols.min(), cols.max() + 1
        
        # 提取边界框内的数据
        mask_data = self._mask[ymin:ymax, xmin:xmax]
        bbox = BoundingBox(ixmin=xmin, ixmax=xmax, iymin=ymin, iymax=ymax)
        
        return ApertureMask(mask_data, bbox=bbox)
    
    def __array__(self) -> np.ndarray:
        """使对象可以直接用作numpy数组"""
        return self.to_bool_array()
    
    @property
    def shape(self) -> Tuple[int, int]:
        """返回掩膜形状"""
        return self._image_shape
    
def mask_image(image: np.ndarray,
               bad_pixel_mask: Optional[Union[np.ndarray, ApertureMask]]) -> np.ndarray:
    """
    处理输入图像和掩模
    
    Parameters
    ----------
    image : np.ndarray
        输入图像
    bad_pixel_mask : np.ndarray or ApertureMask, optional
        坏像素掩模，True表示被mask的像素
        
    Returns
    -------
    np.ndarray
        处理后的图像，被mask的像素设为nan
    """
    if bad_pixel_mask is not None:
        mask = UnifiedMask(bad_pixel_mask, image.shape)
        masked_image = image.copy()
        masked_image[mask.to_bool_array()] = np.nan
        return masked_image
    return image

if __name__ == '__main__':
    image = np.random.normal(0, 1, (100, 100))
    bool_mask = image > 0.5
    mask1 = UnifiedMask(bool_mask)
    aperture_mask1 = mask1.to_aperture_mask()  # 转换为ApertureMask
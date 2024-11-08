from astropy.io import fits
import numpy as np
#from scipy.ndimage import shift, rotate
from skimage.transform import rotate
from typing import List, Tuple, Union, Optional

class DS9Converter:
    def __init__(self):
        """
        初始化对象
        """
        pass
    
    @staticmethod
    def ds9_to_coords(ds9_x: Union[float, int], 
                      ds9_y: Union[float, int], 
                      to_int: bool = True) -> Union[Tuple[int, int, int, int], 
                                                   Tuple[float, float, float, float]]:
        """
        将DS9中的坐标转换为DS9和Python中的坐标
        
        Parameters
        ----------
        ds9_x, ds9_y : float or int
            DS9中的源坐标（从1开始，像素中心是整数）
            ds9_x: 横向坐标（第一个数字）
            ds9_y: 纵向坐标（第二个数字）
        to_int : bool, optional
            是否将结果转换为整数，默认True
            
        Returns
        -------
        ds9_x, ds9_y : int or float
            DS9中的坐标（从1开始）
        python_column, python_row : int or float
            Python数组中的索引（从0开始）
            python_column对应ds9_x（array的第二个索引）
            python_row对应ds9_y（array的第一个索引）
        """
        if to_int:
            # DS9坐标范围[m-0.5, m+0.5)对应整数m
            ds9_out_x = int(np.floor(ds9_x + 0.5))
            ds9_out_y = int(np.floor(ds9_y + 0.5))
        else:
            ds9_out_x = ds9_x
            ds9_out_y = ds9_y
        
        # Python索引从0开始
        python_column = ds9_out_x - 1
        python_row = ds9_out_y - 1
        
        return ds9_out_x, ds9_out_y, python_column, python_row
    
    @staticmethod
    def coords_to_ds9(python_column: Union[float, int], 
                      python_row: Union[float, int], 
                      to_int: bool = True) -> Union[Tuple[int, int, int, int], 
                                                   Tuple[float, float, float, float]]:
        """
        将Python数组中的坐标转换为DS9和Python中的坐标
        
        Parameters
        ----------
        python_column, python_row : float or int
            Python数组中的索引（从0开始）
            python_column对应ds9_x（array的第二个索引）
            python_row对应ds9_y（array的第一个索引）
        to_int : bool, optional
            是否将结果转换为整数，默认True
            
        Returns
        -------
        ds9_x, ds9_y : int or float
            DS9中的坐标（从1开始）
        python_column, python_row : int or float
            Python数组中的索引（从0开始）
        """
        # DS9坐标从1开始
        ds9_x = python_column + 1
        ds9_y = python_row + 1
        
        if to_int:
            ds9_x = int(np.floor(ds9_x + 0.5))
            ds9_y = int(np.floor(ds9_y + 0.5))
            python_column = int(np.floor(python_column + 0.5))
            python_row = int(np.floor(python_row + 0.5))
            
        return ds9_x, ds9_y, python_column, python_row

def rescale_images(images: Union[np.ndarray, List[np.ndarray]], 
                  current_scales: Union[float, List[float]], 
                  new_scale: Optional[float] = None,
                  target_coords: Optional[List[Tuple[float, float]]] = None,
                  headers: Optional[List[fits.Header]] = None) -> Tuple:
    """
    将图像重新缩放到新的像素尺度
    
    Parameters
    ----------
    images : 单个图像数组或图像数组列表
    current_scales : 当前像素尺度或尺度列表
    new_scale : 新的像素尺度，如果为None则使用最大的尺度
    target_coords : 源在各个图像中的坐标列表 [(x1,y1), (x2,y2),...]
    headers : FITS头文件列表
    
    Returns
    -------
    rescaled_images, new_coords, updated_headers (如果提供了headers)
    """
    # 确保输入格式一致
    if not isinstance(images, list):
        images = [images]
        current_scales = [current_scales]
        target_coords = [target_coords] if target_coords is not None else None
    
    # 如果未指定新尺度，使用最大的尺度
    if new_scale is None:
        new_scale = max(current_scales)
    
    rescaled_images = []
    new_coords = []
    updated_headers = []
    
    for i, (img, scale) in enumerate(zip(images, current_scales)):
        # 计算缩放因子
        factor = scale / new_scale
        
        if abs(factor - round(factor)) < 1e-10:  # 整数倍关系
            factor = int(round(factor))
            if factor > 1:  # 需要缩小
                new_img = img[::factor, ::factor]
            elif factor < 1:  # 需要放大
                factor = abs(factor)
                shape = np.array(img.shape) * factor
                new_img = np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)
            else:
                new_img = img.copy()
        else:
            # 非整数倍关系，可以选择其他插值方法
            new_img = img  # 这里可以实现其他插值方法
            
        rescaled_images.append(new_img)
        
        # 更新坐标
        if target_coords:
            x, y = target_coords[i]
            new_coords.append((x/factor, y/factor))
            
        # 更新header
        if headers:
            header = headers[i].copy()
            header['PIXSCALE'] = new_scale
            updated_headers.append(header)
    
    if headers:
        return rescaled_images, new_coords, updated_headers
    elif target_coords:
        return rescaled_images, new_coords
    else:
        return rescaled_images

def rotate_image(img: np.ndarray, 
                target_coord: Tuple[Union[float, int], Union[float, int]], 
                angle: float,
                fill_value: Union[float, None] = np.nan) -> np.ndarray:
    """
    以源位置为中心旋转图像
    
    Parameters
    ----------
    img : 输入图像
    target_coord : (column, row)源的坐标
    angle : 旋转角度（度）
    fill_value : 填充值
    """
    if img.dtype.byteorder == '>':
        img = img.byteswap().newbyteorder()
    return rotate(img, 
                  -angle,
                  center=target_coord,
                  preserve_range=True,
                  mode='constant',
                  cval=fill_value,    # 指定填充值
                  clip=True)

def crop_image(img: np.ndarray, 
              target_coord: Tuple[Union[float, int], Union[float, int]], 
              new_target_coord: Tuple[Union[float, int], Union[float, int]],
              fill_value: Union[float, None] = np.nan) -> np.ndarray:
    """
    以源位置为中心裁剪图像
    
    Parameters
    ----------
    img : 输入图像
    target_coord : (column, row)源在原图中的坐标
    new_target_coord : (column, row)源在新图中的期望坐标
    fill_value : 填充值
    """
    col, row = target_coord
    new_col, new_row = new_target_coord
    
    # 计算新图像大小
    new_size = (2 * new_col + 1, 2 * new_row + 1)
    new_img = np.full(new_size, fill_value)
    
    # 计算裁剪范围
    start_col = col - new_col
    start_row = row - new_row
    
    # 复制有效区域
    valid_region = np.s_[
        max(0, start_row):min(img.shape[0], start_row + new_size[0]),
        max(0, start_col):min(img.shape[1], start_col + new_size[1])
    ]
    new_valid_region = np.s_[
        max(0, -start_row):min(new_size[0], img.shape[0]-start_row),
        max(0, -start_col):min(new_size[1], img.shape[1]-start_col)
    ]
    
    new_img[new_valid_region] = img[valid_region]
    return new_img

def align_images(images: List[np.ndarray], 
                target_coords: List[Tuple[Union[float, int], Union[float, int]]],
                new_target_coord: Tuple[Union[float, int], Union[float, int]],
                fill_value: Union[float, None] = np.nan) -> List[np.ndarray]:
    """
    对齐一系列图像
    """
    return [crop_image(img, coord, new_target_coord, fill_value) 
            for img, coord in zip(images, target_coords)]

def stack_images(images: List[np.ndarray], 
                method: str = 'median') -> np.ndarray:
    """
    叠加图像
    
    Parameters
    ----------
    images : 图像列表
    method : 'median' 或 'sum'
    """
    if method == 'median':
        return np.nanmedian(images, axis=0)
    elif method == 'sum':
        return np.nansum(images, axis=0)
    else:
        raise ValueError("method must be 'median' or 'sum'")
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    img_dict = {'18':(760,872),
                #'20':(773,884),
                '24':(766,877),
                #'26':(778,889)
                }
    img_list = []
    target_list = []
    for imgid in img_dict.keys():
        x, y = img_dict[imgid]
        hdul = fits.open('/Volumes/ZexiWork/data/HST/29P/2019/'+imgid+'.fits')
        img = hdul[1].data
        angle = float(hdul[1].header['ORIENTAT'])
        col, row = DS9Converter.ds9_to_coords(x, y)[2:]
        target_list.append((col, row))

        img = rotate_image(img, target_coord=(col,row), angle=angle, fill_value=np.nan)
        img_list.append(img)

    new_target_coord_ds9 = (100,100)
    col, row = DS9Converter.ds9_to_coords(new_target_coord_ds9[0], new_target_coord_ds9[1])[2:]
    img_list = align_images(img_list, target_list, (col,row))

    img_a = img_list[0]
    img_b = img_list[1]
    diff = img_a - img_b
    mask_pos = diff > 0.05
    mask_neg = diff < -0.05

    filled_a = img_a.copy()
    filled_b = img_b.copy()

    filled_a[mask_pos] = img_b[mask_pos]
    filled_b[mask_neg] = img_a[mask_neg]

    from visualizer import MaskInspector
    inspector = MaskInspector(img_a, mask_pos)
    inspector.show_comparison(vmin=0,vmax=2)
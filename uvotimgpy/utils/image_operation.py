from astropy.io import fits
import numpy as np
#from scipy.ndimage import shift, rotate
from skimage.transform import rotate
from typing import List, Tuple, Union, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from regions import CircleAnnulusPixelRegion, CirclePixelRegion, PixCoord
from scipy.interpolate import interp1d
from uvotimgpy.base.math_tools import ErrorPropagation
from uvotimgpy.base.region import mask_image, RegionStatistics, expand_shrink_region
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.nddata import block_reduce
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.ndimage import binary_dilation
warnings.filterwarnings('ignore', category=FITSFixedWarning)

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

def exposure_mask_with_nan(image: np.ndarray, 
                           exposure_map: np.ndarray, 
                           exposure_threshold: Optional[float] = None) -> np.ndarray:
    """
    将曝光图中小于0/小于等于threshold的像素对应的图像值替换为nan
    """
    if image.shape != exposure_map.shape:
        raise ValueError("image and exposure_map must have the same shape")
    if exposure_threshold is None:
        image[exposure_map <= 0] = np.nan
    else:
        image[exposure_map < exposure_threshold] = np.nan
    return image

def rescale_images(images: Union[np.ndarray, List[np.ndarray]], 
                  current_scales: Union[float, List[float]], 
                  new_scale: Optional[float] = None,
                  target_coords: Optional[List[Tuple[float, float]]] = None,
                  headers: Optional[List[fits.Header]] = None) -> Tuple:
    # TODO: check if this is correct
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

def rotate_image(image: np.ndarray, 
                target_coord: Tuple[Union[float, int], Union[float, int]], 
                angle: float,
                fill_value: Union[float, None] = np.nan,
                image_err: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    以源位置为中心旋转图像
    
    Parameters
    ----------
    image : 输入图像
    target_coord : (row, column)源的坐标
    angle : 旋转角度（度）
    fill_value : 填充值
    """
    if image.dtype.byteorder == '>':
        image = image.byteswap().newbyteorder()
    rotated_img =  rotate(image, 
                          -angle,
                          center=target_coord,
                          preserve_range=True,
                          mode='constant',
                          cval=fill_value,    # 指定填充值
                          clip=True)
    if image_err is None:
        return rotated_img
    else:
        rotated_err = rotate_image(image = image_err, target_coord = target_coord, angle = angle, fill_value = fill_value)
        return rotated_img, rotated_err

def crop_image(image: np.ndarray, 
              target_coord: Tuple[Union[float, int], Union[float, int]], 
              new_target_coord: Tuple[Union[float, int], Union[float, int]],
              fill_value: Union[float, None] = np.nan,
              image_err: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    以源位置为中心裁剪图像
    
    Parameters
    ----------
    image : 输入图像
    target_coord : (column, row)源在原图中的坐标
    new_target_coord : (column, row)源在新图中的期望坐标
    fill_value : 填充值
    """
    col, row = target_coord
    new_col, new_row = new_target_coord
    
    # 计算新图像大小
    new_size = (2 * new_row + 1, 2 * new_col + 1)
    new_image = np.full(new_size, fill_value)
    
    # 计算裁剪范围
    start_col = col - new_col
    start_row = row - new_row
    
    # 复制有效区域
    valid_region = np.s_[
        max(0, start_row):min(image.shape[0], start_row + new_size[0]),
        max(0, start_col):min(image.shape[1], start_col + new_size[1])
    ]
    new_valid_region = np.s_[
        max(0, -start_row):min(new_size[0], image.shape[0]-start_row),
        max(0, -start_col):min(new_size[1], image.shape[1]-start_col)
    ]
    
    new_image[new_valid_region] = image[valid_region]

    if image_err is None:
        return new_image
    else:
        cropped_err = crop_image(image_err, target_coord, new_target_coord, fill_value=fill_value)
        return new_image, cropped_err


def align_images(images: List[np.ndarray], 
                target_coords: List[Tuple[Union[float, int], Union[float, int]]],
                new_target_coord: Tuple[Union[float, int], Union[float, int]],
                fill_value: Union[float, None] = np.nan,
                image_err: Optional[List[np.ndarray]] = None) ->  Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    对齐一系列图像
    """
    if image_err is None:
        return [crop_image(img, coord, new_target_coord, fill_value) 
                for img, coord in zip(images, target_coords)]
    else:
        aligned_images = []
        aligned_errs = []
        for image, coord, err in zip(images, target_coords, image_err):
            aligned_image, aligned_err = crop_image(image, coord, new_target_coord, fill_value, err)
            aligned_images.append(aligned_image)
            aligned_errs.append(aligned_err)
        return aligned_images, aligned_errs

def stack_images(images: List[np.ndarray], 
                 method: str = 'median',
                 image_err: Optional[List[np.ndarray]] = None,
                 median_err_params: Optional[dict] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    叠加图像
    
    Parameters
    ----------
    images : 图像列表
    method : 'median' 或 'mean'
    median_err_params: dict, optional
        median_err_params for ErrorPropagation.median: method ('mean' or 'std'), mask (True/False)
    """
    if method not in ['median', 'mean']:
        raise ValueError("method must be 'median' or 'mean'")
    if method == 'mean':
        warnings.warn("Some pixels may be not well exposed, please check the exposure map with sum_exposure_map.")
    if image_err is None:
        if method == 'median':
            return np.nanmedian(images, axis=0)
        elif method == 'mean':
            return np.nanmean(images, axis=0)
    else:
        if method == 'mean':
            mean_image, mean_error = ErrorPropagation.mean(images, image_err, axis=0, ignore_nan=True)
            return mean_image, mean_error
        elif method == 'median':
            median_image, median_error = ErrorPropagation.median(images, image_err, axis=0, ignore_nan=True,
                                                                 **median_err_params)
            return median_image, median_error

def shrink_valid_image(image: np.ndarray, shrink_pixels: int = 2, mask: Optional[np.ndarray] = None, speed: str = 'normal') -> np.ndarray:
    """
    将图像中的有效区域向内收缩指定像素数，
    也就是将原始的 np.nan 区域 (或mask) 向外扩展 shrink_pixels 个像素。
    
    参数:
        image (np.ndarray): 输入的二维图像数组，必须包含 np.nan 表示无效区域
        shrink_pixels (int): 收缩像素数量（即扩展 NaN 的像素数量）

    返回:
        np.ndarray: 处理后的图像，扩展后的 NaN 区域
    """
    if image.ndim != 2:
        raise ValueError("输入必须是二维图像")

    # 构造 nan 区域掩码
    if mask is None:
        shrink_mask = np.isnan(image)
    else:
        shrink_mask = mask

    # 将 NaN 区域扩展（向内侵蚀有效区域）
    expanded_mask = expand_shrink_region(shrink_mask, radius=shrink_pixels, method='expand', speed=speed)

    # 创建新的图像副本
    new_image = image.copy()
    new_image[expanded_mask] = np.nan

    return new_image

def sum_exposure_map(exposure_maps: Optional[List[np.ndarray]] = None,
                     images: Optional[List[np.ndarray]] = None,
                     exposures: Optional[List[float]] = None) -> np.ndarray:
    """
    叠加曝光图
    """
   # if isinstance(exposure_maps, type(None)):
    if exposure_maps is None:
        exposure_maps = []
        for image, exposure in zip(images, exposures):
            image_copy = image.copy()
            image_copy[~np.isnan(image_copy)] = 1. # TODO: check pixels with values as 0
            exposure_maps.append(image_copy * exposure)
    #else: # for Swift which has exposure map fits files
    #    pass
    summed_exposure_map = np.nansum(exposure_maps, axis=0)
    summed_exposure_map = np.nan_to_num(summed_exposure_map, 0)
    return summed_exposure_map
    

class ImageDistanceCalculator:
    @staticmethod
    def calc_distance(coords1, coords2, wcs=None, scale=None):
        """计算两点间距离"""
        # 直接计算像素距离
        pixel_dist = np.sqrt((coords2[0] - coords1[0])**2 + 
                           (coords2[1] - coords1[1])**2)
        
        if wcs is None:
            if scale is None:
                return pixel_dist
            else:
                return pixel_dist * scale
            
        # 使用WCS转换
        sky1 = wcs.pixel_to_world(coords1[0], coords1[1])
        sky2 = wcs.pixel_to_world(coords2[0], coords2[1])
        return sky1.separation(sky2)
        
    @staticmethod
    def from_edges(image, coords, distance_method='max', return_coords=False, wcs=None, scale=None):
        """计算到边的距离
        
        Args:
            max_distance: True返回最大距离，False返回最小距离
        """
        n_rows, n_cols = image.shape
        col, row = coords
    
        edges = [
            (col, 0),
            (col, n_rows),
            (0, row),
            (n_cols, row)
        ]

        distances = [(ImageDistanceCalculator.calc_distance(coords, edge), edge) for edge in edges]
        if distance_method == 'max':
            dist, edge = max(distances)
        elif distance_method == 'min':
            dist, edge = min(distances)

        if return_coords:
            return edge
        return ImageDistanceCalculator.calc_distance(coords, edge, wcs, scale)
        
    @staticmethod
    def from_corners(image, coords, distance_method='max', return_coords=False, scale=None, wcs=None):
        """计算到角点的距离
        
        Args:
            max_distance: True返回最大距离，False返回最小距离
        """
        n_rows, n_cols = image.shape
        col, row = coords

        corners = [
            (0, 0),
            (0, n_rows),
            (n_cols, 0),
            (n_cols, n_rows)
        ]

        distances = [(ImageDistanceCalculator.calc_distance(coords, corner), corner) for corner in corners]
        if distance_method == 'max':
            dist, corner = max(distances)
        elif distance_method == 'min':
            dist, corner = min(distances)

        if return_coords:
            return corner
        return ImageDistanceCalculator.calc_distance(coords, corner, wcs, scale)

    @staticmethod
    def max_distance_from_valid_pixels(image: np.ndarray, 
                                     coords: Tuple[Union[float, int], Union[float, int]], 
                                     ) -> float:
        """计算图像中非nan像素到指定点的最大距离
        
        Parameters
        ----------
        image : np.ndarray
            输入图像
        coords : tuple
            参考点坐标 (col, row)
        wcs : object, optional
            WCS对象，用于天球坐标转换
        scale : float, optional
            像素尺度，用于将像素距离转换为实际距离
            
        Returns
        -------
        float
            最大距离（像素单位，或根据scale/wcs转换后的单位）
        """
        # 创建坐标网格
        rows, cols = np.indices(image.shape)
        
        # 计算每个像素到参考点的距离
        col_diff = cols - coords[0]
        row_diff = rows - coords[1]
        distances = np.sqrt(col_diff**2 + row_diff**2)
        
        # 将nan像素对应的距离设为nan
        distances[np.isnan(image)] = np.nan
        
        # 计算最大距离
        max_dist = np.nanmax(distances)
            
        return max_dist

class DistanceMap:
    """处理图像中像素到指定中心点距离的类"""
    
    def __init__(self, image: np.ndarray, center: tuple):
        self.image = image
        self.center_col, self.center_row = center
        
        # 直接在初始化时计算距离图
        rows, cols = np.indices(self.image.shape)
        self.dist_map = np.sqrt(
            (cols - self.center_col)**2 + 
            (rows - self.center_row)**2
        )
        
    def get_distance_map(self) -> np.ndarray:
        """计算每个像素到中心的距离"""
        return self.dist_map
    
    def get_range_mask(self, inner_radius: float, outer_radius: float) -> np.ndarray:
        """
        获取指定距离范围内的像素掩膜
        
        Parameters
        ----------
        inner_radius : float
            内半径
        outer_radius : float
            外半径
            
        Returns
        -------
        np.ndarray
            布尔掩膜，在指定范围内的像素为True
        """
        return (self.dist_map >= inner_radius) & (self.dist_map < outer_radius)
    
    def get_index_map(self, step) -> np.ndarray:
        """获取距离的索引图"""            
        #index_map = np.round(self.dist_map / step).astype(int)
        #index_map = np.maximum(index_map, 1)
        index_map = np.floor(self.dist_map / step).astype(int)
        return index_map

def calc_radial_profile(image: np.ndarray, 
                       center: tuple, 
                       step: float,
                       image_err: Optional[np.ndarray] = None,
                       bad_pixel_mask: Optional[np.ndarray] = None,
                       start: Optional[float] = None,
                       end: Optional[float] = None,
                       method: str = 'median',
                       median_err_params: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """计算径向profile及其误差

    Parameters
    ----------
    image : np.ndarray
        输入图像
    center : tuple
        中心点坐标 (col, row)
    step : float
        环宽度
    image_err : np.ndarray, optional
        图像误差数组
    bad_pixel_mask : np.ndarray, optional
        坏像素掩模，True表示被mask的像素
    start : float, optional
        起始半径，默认为0
    end : float, optional
        结束半径，默认为图像中心到角落的距离
    method : str
        计算方法，'median'或'mean'或'max'

    Returns
    -------
    radii : np.ndarray
        半径数组
    values : np.ndarray
        对应的profile值
    errors : np.ndarray, optional
        如果提供了image_err，返回对应的误差值
    """
    # 处理输入图像和掩模
    image = mask_image(image, bad_pixel_mask)
    #image_err = mask_image(image_err, bad_pixel_mask) if not isinstance(image_err, type(None)) else None
    image_err = mask_image(image_err, bad_pixel_mask) if image_err is not None else None
    # 处理中心坐标
    if isinstance(center, PixCoord):
        center_coord = center
    else:
        center_coord = PixCoord(x=center[0], y=center[1])
    
    # 设置起始和结束半径
    start = start if start is not None else 0
    if end is None:
        end = ImageDistanceCalculator.max_distance_from_valid_pixels(image, center)
    
    # 初始化结果列表
    radii_range = np.arange(start, end, step)
    radii = []
    values = []
    errors = [] if image_err is not None else None

    # 计算每个半径处的值
    for r in radii_range:
        # 创建区域
        if r == 0:
            region = CirclePixelRegion(
                center=center_coord,
                radius=step
            )
        else:
            region = CircleAnnulusPixelRegion(
                center=center_coord,
                inner_radius=r,
                outer_radius=r + step
            )

        # 获取区域内的有效像素值
        mask = region.to_mask()
        region_pixels = mask.get_values(image)
        valid_pixels = region_pixels[~np.isnan(region_pixels)]

        if len(valid_pixels) == 0:
            continue

        if image_err is not None:
            region_errors = mask.get_values(image_err)
            valid_errors = region_errors[~np.isnan(region_pixels)]

        radii.append(r+step/2)
        if method == 'median':
            if image_err is not None:
                value, error = ErrorPropagation.median(valid_pixels, valid_errors, axis=None, ignore_nan=True, 
                                                       **median_err_params)
                values.append(value)
                errors.append(error)
            else:
                value = np.nanmedian(valid_pixels)
                values.append(value)
        else:  # mean
            if image_err is not None:
                value, error = ErrorPropagation.mean(valid_pixels, valid_errors, axis=None, ignore_nan=True)
                values.append(value)
                errors.append(error)
            else:
                value = np.nanmean(valid_pixels)
                values.append(value)

    # 转换为numpy数组并返回结果
    radii = np.array(radii)
    values = np.array(values)
    if image_err is not None:
        errors = np.array(errors)
        return radii, values, errors
    return radii, values

def bin_image(image: np.ndarray, 
              block_size: Union[int, Tuple[int, int]], 
              image_err: Optional[np.ndarray] = None,
              method: str = 'mean',
              median_err_params: Optional[dict] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    对图像进行binning操作
    
    Parameters
    ----------
    image : np.ndarray
        输入图像
    block_size : int or tuple of int
        binning的块大小。如果是整数，则在两个方向使用相同的大小；
        如果是元组，则分别指定(y, x)方向的块大小
    method : str, optional
        用于binning的函数，默认为np.nanmean
    image_err : np.ndarray, optional
        图像误差数组
        
    Returns
    -------
    np.ndarray or tuple of np.ndarray
        binning后的图像，如果提供了image_err，则同时返回binning后的误差
    """
    if method == 'mean':
        func = np.nanmean
        def error_func(a_err):
            _, error = ErrorPropagation.mean(a_err, a_err, axis=None, ignore_nan=True)
            return error
        
    elif method == 'median':
        func = np.nanmedian
        def error_func(a_err):
            _, error = ErrorPropagation.median(a_err, a_err, axis=None, ignore_nan=True, **median_err_params)
            return error
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    if isinstance(block_size, int):
        block_size = (block_size, block_size)
        
    binned_image = block_reduce(image, block_size, func=func)
    
    if image_err is None:
        return binned_image
    else:
        if ErrorPropagation.check_consistency(image, image_err):
            binner_err = block_reduce(image_err, block_size, func=error_func)
            return binned_image, binner_err
        else:
            raise ValueError("image and image_err must have consistent shapes and NaN positions")

def smooth_image(image: np.ndarray, kernel_size: int = 3, method: str = 'gaussian',
                 image_err: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    对图像进行平滑处理
    
    Parameters
    ----------
    image : np.ndarray
        输入图像
    kernel_size : int
        卷积核大小
    method : str
        平滑方法，'gaussian'
    """
    if method == 'gaussian':
        kernel = Gaussian2DKernel(kernel_size)
        smoothed_image = convolve(image, kernel)
        if image_err is None:
            return smoothed_image
        else:
            if ErrorPropagation.check_consistency(image, image_err):
                smoothed_err = np.sqrt(convolve(image_err**2, kernel**2))
                return smoothed_image, smoothed_err
            else:
                raise ValueError("image and image_err must have consistent shapes and NaN positions")
    else:
        raise ValueError(f"Unsupported method: {method}")

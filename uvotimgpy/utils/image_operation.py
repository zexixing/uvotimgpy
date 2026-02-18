from astropy.io import fits
import numpy as np
#from scipy.ndimage import shift, rotate
from skimage.transform import rotate
from typing import List, Tuple, Union, Optional, Sequence
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from regions import CircleAnnulusPixelRegion, CirclePixelRegion, PixCoord
from scipy.interpolate import interp1d
from uvotimgpy.base.math_tools import ErrorPropagation
from uvotimgpy.base.region import mask_image, RegionStatistics, expand_shrink_region, create_sector_region, create_sector_region_from_map
import warnings
from astropy.wcs import FITSFixedWarning
from astropy.nddata import block_reduce
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve
from scipy.ndimage import uniform_filter, gaussian_filter
warnings.filterwarnings('ignore', category=FITSFixedWarning)

class DS9Converter:
    def __init__(self):
        """
        初始化对象
        """
        pass
    @staticmethod
    def round_to_int(x) -> int:
        return int(np.floor(x + 0.5))
    
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
        python_column, python_row : int or float
            Python数组中的索引（从0开始）
            python_column对应ds9_x（array的第二个索引）
            python_row对应ds9_y（array的第一个索引）
        """
        if to_int:
            # DS9坐标范围[m-0.5, m+0.5)对应整数m
            ds9_out_x = DS9Converter.round_to_int(ds9_x)
            ds9_out_y = DS9Converter.round_to_int(ds9_y)
        else:
            ds9_out_x = ds9_x
            ds9_out_y = ds9_y
        
        # Python索引从0开始
        python_column = ds9_out_x - 1
        python_row = ds9_out_y - 1
        
        return python_column, python_row
    
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
        """
        # DS9坐标从1开始
        ds9_x = python_column + 1
        ds9_y = python_row + 1
        
        if to_int:
            ds9_x = DS9Converter.round_to_int(ds9_x)
            ds9_y = DS9Converter.round_to_int(ds9_y)
            
        return ds9_x, ds9_y

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
                rotate_center: Tuple[Union[float, int], Union[float, int]], 
                angle: float,
                fill_value: Union[float, None] = np.nan,
                image_err: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    以源位置为中心旋转图像
    
    Parameters
    ----------
    image : 输入图像
    rotate_center : (col, row)源的坐标
    angle : 旋转角度（度）
    fill_value : 填充值
    """
    if image.dtype.byteorder not in ('=', '<'):
        image = image.byteswap().view(image.dtype.newbyteorder('<'))
    rotated_img =  rotate(image, 
                          -angle,
                          center=rotate_center,
                          preserve_range=True,
                          mode='constant',
                          cval=fill_value,    # 指定填充值
                          clip=True)
    if image_err is None:
        return rotated_img
    else:
        rotated_err = rotate_image(image = image_err, target_coord = rotate_center, angle = angle, fill_value = fill_value)
        return rotated_img, rotated_err

def crop_image(image: np.ndarray, 
              old_target_coord: Tuple[Union[float, int], Union[float, int]], 
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
    col, row = old_target_coord
    new_col, new_row = new_target_coord

    if isinstance(new_col, float):
        new_col = round(new_col)
    if isinstance(new_row, float):
        new_row = round(new_row)
    if isinstance(col, float):
        col = round(col)
    if isinstance(row, float):
        row = round(row)
    # 计算新图像大小
    new_size = (2 * new_row + 1, 2 * new_col + 1)
    if isinstance(fill_value, int):
        fill_value = float(fill_value)
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
        cropped_err = crop_image(image_err, old_target_coord, new_target_coord, fill_value=fill_value)
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

def images_by_exposure(images: List[np.ndarray], 
                        exposure_list: Optional[Union[List[np.ndarray], List[float]]],
                        method: str = 'divide') -> List[np.ndarray]:
    """
    将图像除以曝光图
    """
    if all(isinstance(expo, np.ndarray) for expo in exposure_list):
        for img, expo in zip(images, exposure_list):
            img[(expo / np.max(expo)) < 0.99] = np.nan
    #exposure_stack = np.sum(exposure_list, axis=0)
    if method == 'divide':
        images = [img / expo for img, expo in zip(images, exposure_list)]
    elif method == 'multiply':
        images = [img * expo for img, expo in zip(images, exposure_list)]
    else:
        raise ValueError("method must be 'divide' or 'multiply'")
    return images
        

def stack_images(images: List[np.ndarray], 
                 method: str = 'median',
                 image_err: Optional[List[np.ndarray]] = None,
                 median_err_params: Optional[dict] = {'method':'mean', 'mask':True},
                 verbose: bool = False,
                 input_unit: Optional[str] = None,
                 exposure_list: Optional[Union[List[np.ndarray], List[float]]] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    叠加图像
    
    Parameters
    ----------
    images : 图像列表
    method : 'median' 或 'mean' 或 'sum'
    median_err_params: dict, optional
        median_err_params for ErrorPropagation.median: method ('mean' or 'std'), mask (True/False)
    """
    if method not in ['median', 'mean', 'sum']:
        raise ValueError("method must be 'median' or 'mean' or 'sum'")
    if method == 'mean' or method == 'sum':
        if verbose:
            warnings.warn("Some pixels may be not well exposed, please check the exposure map with sum_exposure_map().")
    all_nan = np.all(np.isnan(images), axis=0)
    if method == 'mean':
        if input_unit == 'count' and exposure_list is not None:
            images = images_by_exposure(images, exposure_list, method='divide')
            if image_err is not None:
                image_err = images_by_exposure(image_err, exposure_list, method='divide')
        if image_err is not None:
            mean_image, mean_error = ErrorPropagation.mean(images, image_err, axis=0, ignore_nan=True)
            mean_image[all_nan] = np.nan
            mean_error[all_nan] = np.nan
            return mean_image, mean_error
        else:
            result = np.nanmean(images, axis=0)
            result[all_nan] = np.nan
            return result
    elif method == 'median':
        if input_unit == 'count' and exposure_list is not None:
            images = images_by_exposure(images, exposure_list, method='divide')
            if image_err is not None:
                image_err = images_by_exposure(image_err, exposure_list, method='divide')
        if image_err is not None:
            median_image, median_error = ErrorPropagation.median(images, image_err, axis=0, ignore_nan=True,
                                                                 **median_err_params)
            median_image[all_nan] = np.nan
            median_error[all_nan] = np.nan
            return median_image, median_error
        else:
            result = np.nanmedian(images, axis=0)
            result[all_nan] = np.nan
            return result
    elif method == 'sum':
        if input_unit == 'count/s' and exposure_list is not None:
            images = images_by_exposure(images, exposure_list, method='multiply')
            if image_err is not None:
                image_err = images_by_exposure(image_err, exposure_list, method='multiply')
        if image_err is not None:
            sum_image, sum_error = ErrorPropagation.sum(images, image_err, axis=0, ignore_nan=True)
            sum_image[all_nan] = np.nan
            sum_error[all_nan] = np.nan
            return sum_image, sum_error
        else:
            result = np.nansum(images, axis=0)
            result[all_nan] = np.nan
            return result

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

class GeometryMap:
    """处理图像中像素到指定中心点距离的类"""
    
    def __init__(self, image: np.ndarray, center: tuple):
        #self.image = image
        #self.center_col, self.center_row = center
        #
        ## 直接在初始化时计算距离图
        #rows, cols = np.indices(self.image.shape)
        #self.dist_map = np.sqrt(
        #    (cols - self.center_col)**2 + 
        #    (rows - self.center_row)**2
        #)
        height, width = image.shape

        # 创建像素坐标网格
        cols = np.arange(width)
        rows = np.arange(height)
        col_grid, row_grid = np.meshgrid(cols, rows)

        # 计算每个像素到中心的距离
        self.center_col, self.center_row = center
        self.dcol = col_grid - self.center_col
        self.drow = row_grid - self.center_row
        
    def get_distance_map(self) -> np.ndarray:
        """计算每个像素到中心的距离"""
        return np.sqrt(self.dcol**2 + self.drow**2)
    
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
        self.dist_map = np.sqrt(self.dcol**2 + self.drow**2)
        return (self.dist_map >= inner_radius) & (self.dist_map < outer_radius)
    
    def get_index_map(self, step) -> np.ndarray:
        """获取距离的索引图"""            
        #index_map = np.round(self.dist_map / step).astype(int)
        #index_map = np.maximum(index_map, 1)
        self.dist_map = np.sqrt(self.dcol**2 + self.drow**2)
        index_map = np.floor(self.dist_map / step).astype(int)
        return index_map

    def get_angle_map(self) -> np.ndarray:
        """获取每个像素相对于中心的角度"""
        angle_rad = np.arctan2(self.drow, self.dcol)
        angle_deg = np.degrees(angle_rad)
        return (angle_deg + 270) % 360 # 0度=上, 90度=左, 180度=下, 270度=右

def build_edge_list(start: float,
                    end: float,
                    step: float = 1.0,
                    power: float = 0.0):
    """
    Vectorized bin construction using
        width_i = step * (1 + i^beta) / 2

    power = 0: linear (width = step always)
    power > 0: increasing widths
    """
    if end <= start:
        raise ValueError("end must > start")
    if step <= 0:
        raise ValueError("step must > 0")

    full_span = end - start

    # --- (1) 粗略估计最大需要多少 bin （线性下的上限） ---
    # 因为 width_i >= step/2
    max_bins = int(np.ceil(full_span / (step / 2)))  
    # 通常 max_bins 会比实际有效的多，但不会影响性能

    # --- (2) 一次性生成 i = 0..max_bins 的全部 width ---
    i = np.arange(max_bins, dtype=float)[1:]

    #if beta == 0.0:
    #    widths = np.full_like(i, step)
    #else:
    widths = step * (1 + i**power) / 2

    # --- (3) 累积和 ---
    cumsum = widths.cumsum()

    # --- (4) 找到所有 bin 都能完全包含的范围 ---
    valid = cumsum <= full_span
    if not np.any(valid):
        # 一个 bin 也放不下，退化为单 bin
        return np.array([start, end], dtype=float)

    last = np.where(valid)[0][-1]  # 最后的有效 bin index

    # --- (5) 构造 edge_list ---
    widths_used = widths[: last + 1]
    edge_offsets = np.concatenate(([0.0], widths_used.cumsum()))
    edge_list = start + edge_offsets

    return edge_list

#def calc_radial_profile(image: np.ndarray, 
#                       center: tuple, 
#                       step: float,
#                       edge_list: Optional[List[float]] = None,
#                       image_err: Optional[np.ndarray] = None,
#                       bad_pixel_mask: Optional[np.ndarray] = None,
#                       sector_pa: Optional[float] = None,
#                       sector_span: Optional[float] = None,
#                       start: Optional[float] = None,
#                       end: Optional[float] = None,
#                       method: str = 'median',
#                       median_err_params: Optional[dict] = {'method':'mean', 'mask':True},
#                       power: float = 1.0) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
#    """计算径向profile及其误差
#
#    Parameters
#    ----------
#    image : np.ndarray
#        输入图像
#    center : tuple
#        中心点坐标 (col, row)
#    step : float
#        环宽度
#    edge_list : List[float], optional
#        边缘列表
#    image_err : np.ndarray, optional
#        图像误差数组
#    bad_pixel_mask : np.ndarray, optional
#        坏像素掩模，True表示被mask的像素
#    start : float, optional
#        起始半径，默认为0
#    end : float, optional
#        结束半径，默认为图像中心到角落的距离
#    method : str
#        计算方法，'median'或'mean'或'max'
#
#    Returns
#    -------
#    radii : np.ndarray
#        半径数组
#    values : np.ndarray
#        对应的profile值
#    errors : np.ndarray, optional
#        如果提供了image_err，返回对应的误差值
#    """
#    # 处理输入图像和掩模
#    if sector_pa is not None and sector_span is not None:
#        sector_region = create_sector_region(center, sector_pa, sector_span, image.shape, radius=end)
#        out_sector_mask = ~sector_region
#        if bad_pixel_mask is None:
#            bad_pixel_mask = out_sector_mask
#        else:
#            bad_pixel_mask = bad_pixel_mask | out_sector_mask
#    image = mask_image(image, bad_pixel_mask)
#    #image_err = mask_image(image_err, bad_pixel_mask) if not isinstance(image_err, type(None)) else None
#    image_err = mask_image(image_err, bad_pixel_mask) if image_err is not None else None
#    # 处理中心坐标
#    if isinstance(center, PixCoord):
#        center_coord = center
#    else:
#        center_coord = PixCoord(x=center[0], y=center[1])
#    
#    # 设置起始和结束半径
#    start = start if start is not None else 0
#    if end is None:
#        end = ImageDistanceCalculator.max_distance_from_valid_pixels(image, center)
#    
#    # 初始化结果列表
#    if edge_list is None and step is not None:
#        #edge_list = np.arange(start, end+step, step)
#        edge_list = np.asarray(build_edge_list(start, end, step, power=power))
#    radii = []
#    values = []
#    errors = [] if image_err is not None else None
#
#    # 计算每个半径处的值
#    for i, r in enumerate(edge_list[:-1]):
#        # 创建区域
#        r_next = edge_list[i+1]
#        if r == 0:
#            region = CirclePixelRegion(
#                center=center_coord,
#                radius=r_next
#            )
#            radii.append(r)
#        else:
#            region = CircleAnnulusPixelRegion(
#                center=center_coord,
#                inner_radius=r,
#                outer_radius=r_next
#            )
#            radii.append((r+r_next)/2)
#
#        # 获取区域内的有效像素值
#        mask = region.to_mask()
#        region_pixels = mask.get_values(image)
#        valid_pixels = region_pixels[~np.isnan(region_pixels)]
#
#        if len(valid_pixels) == 0:
#            values.append(np.nan)
#            if image_err is not None:
#                errors.append(np.nan)
#            continue
#
#        if image_err is not None:
#            region_errors = mask.get_values(image_err)
#            valid_errors = region_errors[~np.isnan(region_pixels)]
#
#        if method == 'median':
#            if image_err is not None:
#                value, error = ErrorPropagation.median(valid_pixels, valid_errors, axis=None, ignore_nan=True, 
#                                                       **median_err_params)
#                values.append(value)
#                errors.append(error)
#            else:
#                value = np.nanmedian(valid_pixels)
#                values.append(value)
#        elif method == 'mean':
#            if image_err is not None:
#                value, error = ErrorPropagation.mean(valid_pixels, valid_errors, axis=None, ignore_nan=True)
#                values.append(value)
#                errors.append(error)
#            else:
#                value = np.nanmean(valid_pixels)
#                values.append(value)
#
#    # 转换为numpy数组并返回结果
#    radii = np.array(radii)
#    values = np.array(values)
#    if image_err is not None:
#        errors = np.array(errors)
#        return radii, values, errors
#    return radii, values
    
def calc_radial_profile(image: np.ndarray, 
                       center: tuple, 
                       step: float,
                       edge_list: Optional[List[float]] = None,
                       image_err: Optional[np.ndarray] = None,
                       bad_pixel_mask: Optional[np.ndarray] = None,
                       sector_pa: Optional[float] = None,
                       sector_span: Optional[float] = None,
                       start: Optional[float] = None,
                       end: Optional[float] = None,
                       method: str = 'median',
                       median_err_params: Optional[dict] = {'method':'mean', 'mask':True},
                       power: float = 1.0,
                       distance_map: Optional[np.ndarray] = None,
                       angle_map: Optional[np.ndarray] = None,
                       ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """计算径向profile及其误差

    Parameters
    ----------
    image : np.ndarray
        输入图像
    center : tuple
        中心点坐标 (col, row)
    step : float
        环宽度
    edge_list : List[float], optional
        边缘列表
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
    if distance_map is None or angle_map is None:
        geometry_mapper = GeometryMap(image, center)
        if distance_map is None:
            distance_map = geometry_mapper.get_distance_map()
        if angle_map is None:
            angle_map = geometry_mapper.get_angle_map()
    if sector_pa is not None and sector_span is not None:
        #sector_region = create_sector_region(center, sector_pa, sector_span, image.shape, radius=end, )
        sector_region = create_sector_region_from_map(distance_map, angle_map, sector_pa, sector_span, radius=end)
        out_sector_mask = ~sector_region
        if bad_pixel_mask is None:
            bad_pixel_mask = out_sector_mask
        else:
            bad_pixel_mask = bad_pixel_mask | out_sector_mask
    value_center = image[center[1], center[0]]
    if image_err is not None:
        error_center = image_err[center[1], center[0]]
    else:
        error_center = None
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
    if edge_list is None and step is not None:
        #edge_list = np.arange(start, end+step, step)
        edge_list = np.asarray(build_edge_list(start, end, step, power=power))

    n_bin = len(edge_list) - 1
    # 计算每个 bin 的代表半径（默认是边界中点；若最内层从 0 开始则强制为 0）
    radii = 0.5 * (edge_list[:-1] + edge_list[1:])
    if edge_list[0] == 0:
        radii[0] = 0.0

    # 拉平成一维，建立有效像素 mask
    radius_flat = distance_map.ravel()
    image_flat = image.ravel()
    finite = np.isfinite(image_flat)
    # 只考虑半径在 edge_list 范围内的像素
    r_min, r_max = edge_list[0], edge_list[-1]
    radius_in = (radius_flat >= r_min) & (radius_flat <= r_max)
    valid = finite & radius_in
    if image_err is not None:
        image_err_flat = image_err.ravel()
        finite_err = np.isfinite(image_err_flat)
        #valid &= finite_err
    else:
        image_err_flat = None
    if not np.any(valid):
        values = np.full(n_bin, np.nan, dtype=float)
        errors = np.full(n_bin, np.nan, dtype=float) if image_err is not None else None
        if errors is not None:
            return radii, values, errors
        else:
            return radii, values

    r_flat = radius_flat[valid]
    v_flat = image_flat[valid]
    if image_err_flat is not None:
        e_flat = image_err_flat[valid]
    else:
        e_flat = None

    # 将像素按半径分配到 bin
    bin_idx = np.digitize(r_flat, edge_list) - 1   # 0 到 n_bin-1
    # 防御性裁剪（极端边界）
    good_bins = (bin_idx >= 0) & (bin_idx < n_bin)
    r_flat = r_flat[good_bins]
    v_flat = v_flat[good_bins]
    if e_flat is not None:
        e_flat = e_flat[good_bins]
    bin_idx = bin_idx[good_bins]
    # 初始化输出
    values = np.full(n_bin, np.nan, dtype=float)
    errors = np.full(n_bin, np.nan, dtype=float) if e_flat is not None else None
    if edge_list[0] == 0:
        values[0] = value_center
        if e_flat is not None:
            errors[0] = error_center
    if edge_list[0] == 0:
        loop_range = range(1, n_bin)
    else:
        loop_range = range(n_bin)
    # 按 bin 逐个计算（median/mean/max），这里仍是 python loop，
    for i in loop_range:
        sel = (bin_idx == i)
        if not np.any(sel):
            continue
        pix = v_flat[sel]
        if e_flat is not None:
            errs = e_flat[sel]
        else:
            errs = None
#        if radii[i] == 0:
#            print('hi')
#            values[i] = value_center
#            if errs is not None:
#                errors[i] = error_center
        if method == 'median':
            if errs is not None:
                # 保留你原来的误差传播逻辑
                value, error = ErrorPropagation.median(
                    pix, errs, axis=None, ignore_nan=True, **median_err_params
                )
                values[i] = value
                errors[i] = error
            else:
                values[i] = np.nanmedian(pix)
        elif method == 'mean':
            if errs is not None:
                value, error = ErrorPropagation.mean(
                    pix, errs, axis=None, ignore_nan=True
                )
                values[i] = value
                errors[i] = error
            else:
                values[i] = np.nanmean(pix)
        elif method == 'min':
            values[i] = np.nanmin(pix)
        else:
            raise ValueError(f"Unknown method '{method}', must be 'median' or 'mean'.")
    if errors is not None:
        return radii, values, errors
    else:
        return radii, values

def calc_radial_profiles_sectors(
    image: np.ndarray,
    center: tuple,
    step: float,
    sector_pa_list: Sequence[float],
    sector_span: Union[float, Sequence[float]],
    edge_list: Optional[Sequence[float]] = None,
    image_err: Optional[np.ndarray] = None,
    bad_pixel_mask: Optional[np.ndarray] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
    method: str = "median",
    median_err_params: Optional[dict] = {'method':'mean', 'mask':True},
    power: float = 1.0,
    verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    对多个 sector_pa 计算各自的 radial profile

    Parameters
    ----------
    sector_pa_list : sequence of float
        每个扇区的 position angle
    sector_span : float or sequence of float
        若为 float/int，则所有扇区用同一个 span；
        若为 sequence，则长度必须与 sector_pa_list 一致。
    其他参数：
        与 calc_radial_profile 一致，edge_list 若为 None 则内部统一生成。

    Returns
    -------
    radii : (n_rad,) ndarray
    values : (n_sector, n_rad) ndarray
        每一行对应一个 sector 的 profile
    errors : (n_sector, n_rad) ndarray or None
    """
    sector_pa_list = np.asarray(sector_pa_list, dtype=float)
    n_sector = len(sector_pa_list)

    # 规范化 sector_span
    if np.isscalar(sector_span):
        sector_span_list = np.full(n_sector, float(sector_span))
    else:
        sector_span_list = np.asarray(sector_span, dtype=float)
        if len(sector_span_list) != n_sector:
            raise ValueError("sector_span 长度必须与 sector_pa_list 一致。")
    # 预计算 radius_map / theta_map（所有扇区共用）
    geometry_mapper = GeometryMap(image, center)
    distance_map = geometry_mapper.get_distance_map()
    angle_map = geometry_mapper.get_angle_map()
    if bad_pixel_mask is not None:
        image = mask_image(image, bad_pixel_mask)
        image_err = mask_image(image_err, bad_pixel_mask) if image_err is not None else None
    # 如果没给 edge_list，先按整个图像生成一个（所有 sector 共用）
    if edge_list is None:
        # 这里直接用你上面封装好的 build_edge_list
        start_val = 0.0 if start is None else start
        # 先暂时求 end（用一次 max_distance）
        if end is None:
            end_val = ImageDistanceCalculator.max_distance_from_valid_pixels(image, center)
        else:
            end_val = end
        edge_list = build_edge_list(start_val, end_val, step, power=power)

    edge_list = np.asarray(edge_list)

    profile_dict = {}

    for pa, span in zip(sector_pa_list, sector_span_list):
        result = calc_radial_profile(
            image=image,
            center=center,
            step=step,                  # 不再用于生成 edge_list，但保持接口一致
            edge_list=edge_list,        # 固定同一组 edge_list
            image_err=image_err,
            bad_pixel_mask=None,
            sector_pa=pa,
            sector_span=span,
            start=start,
            end=end,
            method=method,
            median_err_params=median_err_params,
            power=power,
            distance_map=distance_map,
            angle_map=angle_map,
        )
        if image_err is not None:
            radii, vals, errs = result
            profile_dict[pa] = {'radii': radii, 'values': vals, 'errors': errs}
        else:
            radii, vals = result
            profile_dict[pa] = {'radii': radii, 'values': vals}
        if verbose:
            print(f'sector {pa} done')
    return profile_dict


def bin_image(image: np.ndarray, 
              block_size: Union[int, Tuple[int, int]], 
              image_err: Optional[np.ndarray] = None,
              method: str = 'mean',
              median_err_params: Optional[dict] = {'method':'mean', 'mask':True}) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    对图像进行binning操作
    
    Parameters
    ----------
    image : np.ndarray
        输入图像
    block_size : int or tuple of int
        binning的块大小。如果是整数，则在两个方向使用相同的大小；
        如果是元组，则分别指定(col, row)方向的块大小
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
        def error_func(a_err, axis=None):
            _, error = ErrorPropagation.mean(a_err, a_err, axis=axis, ignore_nan=True)
            return error
        
    elif method == 'median':
        func = np.nanmedian
        def error_func(a_err, axis=None):
            _, error = ErrorPropagation.median(a_err, a_err, axis=axis, ignore_nan=True, **median_err_params)
            return error
    elif method == 'sum':
        func = np.nansum
        def error_func(a_err, axis=None):
            _, error = ErrorPropagation.sum(a_err, a_err, axis=axis, ignore_nan=True)
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

def bin_coordinate(pixel_coord, block_size):
    """
    将原图像素坐标 pixel_coord 映射到 block_reduce 之后的 binned 图像坐标。

    参数
    ----------
    pixel_coord: (col, row)
    block_size : tuple(int, int) or int
        每个方向上的 block 大小，形式为 (bcol, brow)。

    返回
    ----------
    (colb, rowb) : tuple(int, int)
        binned 图像中的像素坐标。
    """
    col, row = pixel_coord
    if isinstance(block_size, int):
        bcol, brow = block_size, block_size
    else:
        bcol, brow = block_size
    rowb = row // brow
    colb = col // bcol
    return (colb, rowb)

def get_original_coordinate_range(pixel_coord, block_size):
    """
    给定 binned 图像的坐标 (yb, xb)，返回对应的原图像素范围。

    参数
    ----------
    pixel_coord : (col, row)
        binned 图像坐标。
    block_size : tuple(int, int) or int
        每个方向上的 block 大小，形式为 (bcol, brow)。

    返回
    ----------
    (colrange, rowrange) : tuple(range, range)
        原图中对应的行列范围。
    """
    if isinstance(block_size, int):
        bcol, brow = block_size, block_size
    else:
        bcol, brow = block_size
    col, row = pixel_coord
    colrange = range(col * bcol, (col + 1) * bcol)
    rowrange = range(row * brow, (row + 1) * brow)
    return colrange, rowrange

def tile_image(image: np.ndarray, tile_size: int, func=np.median, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    将图像按tile分块，每块用该块的中位数值填充。
    Parameters
    ----------
    image : np.ndarray
        输入图像
    tile_size : int
        tile大小
    func : callable
        填充函数, np.median or np.mean ...
    mask : np.ndarray, optional
        掩膜
    """
    image_tiled = image.copy()
    ny, nx = image.shape
    if mask is not None:
        image_tiled[mask] = np.nan
    for y0 in range(0, ny, tile_size):
        for x0 in range(0, nx, tile_size):
            y1 = min(y0 + tile_size, ny)
            x1 = min(x0 + tile_size, nx)
            tile = image_tiled[y0:y1, x0:x1]
            
            # 提取非 NaN 有效值
            valid_values = tile[~np.isnan(tile)]
            if valid_values.size > 0:
                filled_val = func(valid_values)
                image_tiled[y0:y1, x0:x1] = filled_val
                # 可选：tile 中的 NaN 保持为 NaN，不做替换
            else:
                # 整个 tile 都是 NaN，保持为 NaN
                continue
    return image_tiled

def smooth_image(image: np.ndarray, size: int = 3, method: str = 'gaussian_filter',
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
        平滑方法，'convolve', 'gaussian_filter', 'boxcar', 'fft'
    """
    if image_err is not None:
        consistency = ErrorPropagation.check_consistency(image, image_err)
        if not consistency:
            raise ValueError("image and image_err must have consistent shapes and NaN positions")
    if method == 'convolve':
        kernel = Gaussian2DKernel(size)
        smoothed_image = convolve(image, kernel)
        mask = ~np.isfinite(image)
        smoothed_image[mask] = np.nan
        if image_err is None:
            return smoothed_image
        else:
            smoothed_err = np.sqrt(convolve(image_err**2, kernel**2))
            smoothed_err[mask] = np.nan
            return smoothed_image, smoothed_err
    elif method == 'gaussian_filter':
        smoothed_image = gaussian_filter(image, sigma=size)
        if image_err is None:
            return smoothed_image
        else:
            smoothed_err = gaussian_filter(image_err, sigma=size)
            return smoothed_image, smoothed_err
    elif method == 'boxcar':
        mask = ~np.isfinite(image)
        image[mask] = 0.0
        smoothed_image = uniform_filter(image, size=size)
        smoothed_image[mask] = np.nan
        if image_err is None:
            return smoothed_image
        else:
            image_err[mask] = 0.0
            smoothed_err = uniform_filter(image_err, size=size)
            smoothed_err[mask] = np.nan
            return smoothed_image, smoothed_err
    elif method == 'fft':
        mask = ~np.isfinite(image)
        image[mask] = 0.0
        kernel = Gaussian2DKernel(size)
        kernel = kernel.array.astype(np.float32)
        smoothed_image = fftconvolve(image, kernel, mode='same')
        smoothed_image[mask] = np.nan
        if image_err is None:
            return smoothed_image
        else:
            image_err[mask] = 0.0
            smoothed_err = fftconvolve(image_err, kernel, mode='same')
            smoothed_err[mask] = np.nan
            return smoothed_image, smoothed_err
        

def profile_to_image(profile_r, profile_value, distance_map=None, radius=None, fill_value=np.nan,
                     start_r = None, start_value = None):
    """
    将径向profile转换为图像
    fill_value: np.nan, 0, 'extrapolate'
    """
    if start_r is not None:

        if start_r < profile_r[0]:
            profile_r = np.insert(profile_r, 0, start_r)
            profile_value = np.insert(profile_value, 0, start_value)
        elif start_r > profile_r[0]:
            profile_value = profile_value[profile_r >= start_r]
            profile_r = profile_r[profile_r >= start_r]
        else:
            pass
    profile_interp = interp1d(profile_r, profile_value, 
                              bounds_error=False, 
                              fill_value=fill_value)
    if (distance_map is None) and (radius is not None):
        radius = int(radius)
        center = (radius, radius)
        shape = (2*radius+1, 2*radius+1)
        image = np.zeros(shape)
        distance_calculator = GeometryMap(image, center)
        distance_map = distance_calculator.get_distance_map()
    elif distance_map is not None:
        pass
    elif (distance_map is not None) and (radius is not None):
        raise ValueError("distance_map and radius cannot be provided at the same time")
    else:
        raise ValueError("distance_map or radius must be provided")
    image = profile_interp(distance_map)
    return image

def center_of_image(image: np.ndarray):
    H = len(image)
    W = len(image[0])
    center_col = (W - 1) / 2
    center_row = (H - 1) / 2
    return center_col, center_row

def upscale_mean_fill(
    image: np.ndarray,
    n: int,
    conserve_flux: bool = False,) -> np.ndarray:
    """
    Enlarge a 2D image by an integer factor n using local-mean block filling.

    Each original pixel becomes an n×n block filled with the mean of its
    surrounding neighborhood (here: the pixel value itself, so equivalent
    to constant block replication). This matches typical astronomical
    area-style upscaling.

    Parameters
    ----------
    image : 2D ndarray
        Input image.
    n : int
        Upscaling factor (n >= 1).
    conserve_flux : bool, optional
        If True, divide the result by n^2 so that total sum of pixels
        is conserved.

    Returns
    -------
    upscaled : 2D ndarray
        Enlarged image of shape (ny*n, nx*n).
    """
    if n < 1 or int(n) != n:
        raise ValueError("n must be a positive integer")

    img = np.asarray(image, dtype=float)

    if img.ndim != 2:
        raise ValueError("image must be 2D")

    # Block replication (fast, no loops)
    up = np.repeat(np.repeat(img, n, axis=0), n, axis=1)

    if conserve_flux:
        up /= n * n

    return up
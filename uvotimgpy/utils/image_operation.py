from astropy.io import fits
import numpy as np
#from scipy.ndimage import shift, rotate
from skimage.transform import rotate
from typing import List, Tuple, Union, Optional
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from regions import CircleAnnulusPixelRegion, CirclePixelRegion, PixCoord
from uvotimgpy.base.unit_tools import convert_sequence_to_array
from uvotimgpy.base.region import mask_image
import warnings
from astropy.wcs import FITSFixedWarning
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
    target_coord : (row, column)源的坐标
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
    new_size = (2 * new_row + 1, 2 * new_col + 1)
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
                if hasattr(pixel_dist, 'unit') and hasattr(scale, 'unit'):
                    return pixel_dist*scale
                else:
                    warnings.warn("Ambiguous units for scale or distance.")
                    return getattr(pixel_dist, 'value', pixel_dist)*getattr(scale, 'value', scale)
            
        # 对于WCS转换，需要float值
        try:
            col1, row1 = coords1[0].to('pixel').value, coords1[1].to('pixel').value
            col2, row2 = coords2[0].to('pixel').value, coords2[1].to('pixel').value
        except AttributeError:
            col1, row1 = coords1
            col2, row2 = coords2
            
        sky1 = wcs.pixel_to_world(col1, row1)
        sky2 = wcs.pixel_to_world(col2, row2)
        return sky1.separation(sky2)
        
    @staticmethod
    def from_edges(image, coords, distance_method='max', return_coords=False, wcs=None, scale=None):
        """计算到边的距离
        
        Args:
            max_distance: True返回最大距离，False返回最小距离
        """
        n_rows, n_cols = image.shape
        col, row = coords
        has_units = hasattr(col, 'unit')
    
        if has_units:
            edges = [
                (col, 0*row.unit),
                (col, n_rows*row.unit),
                (0*col.unit, row),
                (n_cols*col.unit, row)
            ]
        else:
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
        has_units = hasattr(col, 'unit')

        if has_units:
            corners = [
                (0*col.unit, 0*col.unit),
                (0*col.unit, n_rows*row.unit),
                (n_cols*col.unit, 0*row.unit),
                (n_cols*col.unit, n_rows*row.unit)
            ]
        else:
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
        index_map = np.round(self.dist_map / step).astype(int)
        index_map = np.maximum(index_map, 1)
        return index_map

class RadialProfile:
    """使用astropy.regions测量图像的径向profile"""
    
    def __init__(self, image: Union[np.ndarray, u.Quantity], center: tuple, step: float,
                 bad_pixel_mask: Optional[np.ndarray] = None,
                 start: Optional[float] = None,
                 end: Optional[float] = None,
                 method: str = 'median'):
        """
        Parameters
        ----------
        image : np.ndarray
            输入图像
        center : tuple
            中心点坐标 (col, row)
        step : float
            环宽度
        bad_pixel_mask : np.ndarray, optional
            坏像素掩模，True表示被mask的像素
        start : float, optional
            起始半径，默认为0
        end : float, optional
            结束半径，默认为图像中心到角落的距离
        method : str
            计算方法，'median'或'mean'
        """
        self.image = mask_image(image, bad_pixel_mask)
        self.center = PixCoord(x=center[0], y=center[1])
        self.step = step
        self.method = method
        
        # 设置起始和结束半径
        self.start = start if start is not None else 0
        if end is None:
            rows, cols = image.shape
            self.end = np.sqrt((rows/2)**2 + (cols/2)**2)
        else:
            self.end = end
            
    def get_radial_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """计算径向profile
        
        Returns
        -------
        radii : np.ndarray
            半径数组
        values : np.ndarray
            对应的profile值
        """
        # 生成半径数组
        radii_range = np.arange(self.start, self.end, self.step)
        radii = []
        values = []
        
        # 对每个半径计算统计量
        for r in radii_range:
            if r == 0:
                # 使用CirclePixelRegion处理中心区域
                region = CirclePixelRegion(
                    center=self.center,
                    radius=self.step
                )
            else:
                # 使用CircleAnnulusPixelRegion处理环形区域
                region = CircleAnnulusPixelRegion(
                    center=self.center,
                    inner_radius=r,
                    outer_radius=r + self.step
                )
                
            mask = region.to_mask()
            data = mask.multiply(self.image)
            
            # 去除被mask的像素
            valid_data = data[data != 0]
            if len(valid_data) > 0:
                radii.append(r)
                if self.method == 'median':
                    values.append(np.nanmedian(valid_data))
                else:
                    values.append(np.nanmean(valid_data))
                
        return np.array(radii), convert_sequence_to_array(values)
    
def test_image_operation():
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

        img = rotate_image(img, target_coord=(col, row), angle=angle, fill_value=np.nan)
        img_list.append(img)

    new_target_coord_ds9 = (100,100)
    col, row = DS9Converter.ds9_to_coords(new_target_coord_ds9[0], new_target_coord_ds9[1])[2:]
    img_list = align_images(img_list, target_list, (col, row))

    img_a = img_list[0]
    img_b = img_list[1]
    diff = img_a - img_b
    mask_pos = diff > 0.05
    mask_neg = diff < -0.05

    filled_a = img_a.copy()
    filled_b = img_b.copy()

    filled_a[mask_pos] = img_b[mask_pos]
    filled_b[mask_neg] = img_a[mask_neg]

    from uvotimgpy.base.visualizer import MaskInspector
    inspector = MaskInspector(img_a, mask_pos)
    inspector.show_comparison(vmin=0,vmax=2)

if __name__ == '__main__':
    pass
    # test_image_operation()
    #img = fits.open('/Users/zexixing/Downloads/30.fits')[1].data
    #selector = ApertureSelector(img,0,2)
    #aperture = selector.get_apertures()
    #print(aperture)
    #from astropy.wcs import WCS
    #wcs = WCS(fits.open('/Users/zexixing/Downloads/30.fits')[1].header)
    #dist = ImageDistanceCalculator.calc_distance((623,1131), (576,1142), wcs)
    #print(dist.arcsec)
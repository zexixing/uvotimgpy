from typing import List, Union, Callable, Any, Sequence, Dict
import numpy as np
import astropy.units as u
from astropy.units import quantity_input

import ast
import inspect
import textwrap

from typing import Any, List, Union, Tuple
import numpy as np
from astropy import units as u

import ast
import inspect
import textwrap
from astropy import units as u
import numpy as np

from typing import Union, List, Any, Optional

def simplify_unit(quantity: u.Quantity) -> u.Quantity:
    """简化Quantity的单位表示"""
    if not isinstance(quantity, u.Quantity):
        return quantity
        
    # 尝试转换为基本单位
    try:
        return quantity.decompose()
    except:
        # 如果无法分解，保持原样
        return quantity

class UnitPropagator:
    """单位传播计算器：分析函数中的运算并追踪物理单位的传播"""

        # 需要展开array的函数：作用于单个array的所有元素
    array_ops = {
            # 保持单位不变
            'preserve': {
                np.sum, np.mean, np.average, np.median, np.min, np.max,
                np.percentile, np.nansum, np.nanmean, np.nanmedian, np.nanmin, np.nanmax,
                np.cumsum,
                # 排序和形状操作
                np.sort,
                # 数组组合
                np.concatenate, np.stack, np.vstack, np.hstack, np.dstack,
                # 形状操作
                np.reshape, np.ravel,
                # 移动std到这里
                np.std, np.nanstd
            },
            # 单位相乘
            'multiply': {
                np.prod, np.nanprod, np.cumprod,
                # 添加矩阵运算
                np.dot, np.matmul, np.inner, np.outer
            },
            # 单位平方
            'square': {
                np.var, np.nanvar,  # 只有方差是平方单位
                np.ptp  # peak to peak = max - min
            }
        }
        
        # 不需要展开array的函数：作用于多个array之间对应元素
    element_ops = {
            # 返回无量纲
            'dimensionless': {
                np.exp, np.log, np.log10, np.log2, np.log1p,
                np.sin, np.cos, np.tan, np.arcsin, np.arccos, np.arctan,
                np.sinh, np.cosh, np.tanh, np.arcsinh, np.arccosh, np.arctanh,
                np.greater, np.less, np.equal, np.not_equal,
                np.greater_equal, np.less_equal,
                np.isfinite, np.isinf, np.isnan
            },
            # 保持单位
            'preserve': {
                np.negative, np.positive, np.absolute, np.abs, np.fabs,
                np.around, np.round, np.rint, np.floor, np.ceil, np.trunc,
                np.gradient, np.diff  # 保持单位的导数
            },
            # 基本运算
            'basic': {
                np.add, np.subtract, np.multiply, np.divide,
                np.true_divide, np.floor_divide, np.remainder, np.mod
            },
            # 幂运算
            'power': {
                np.power, np.sqrt, np.square, np.cbrt
            }
        }
    @staticmethod
    def _is_unit_array(arg: Any) -> bool:
        """检查是否是单位数组"""
        if isinstance(arg, (list, np.ndarray)):
            try:
                return isinstance(arg[0][0], (u.UnitBase, u.CompositeUnit))
            except (IndexError, TypeError):
                return False
        elif isinstance(arg, u.Quantity):
            return True
        return False
    
    @staticmethod
    def _get_unit(arg: Any) -> Union[u.Unit, float]:
        """获取参数的单位或值"""
        if isinstance(arg, u.Quantity):
            return arg.unit
        elif isinstance(arg, (int, float)):
            return float(arg)
        elif isinstance(arg, (list, np.ndarray)):
            if len(arg) > 0:
                if isinstance(arg[0], u.Quantity):
                    return arg[0].unit
                elif isinstance(arg[0], (list, np.ndarray)) and isinstance(arg[0][0], (u.UnitBase, u.CompositeUnit)):
                    return arg[0][0]
        return u.dimensionless_unscaled

    @staticmethod
    def _get_array_info(arg: Any) -> Tuple[u.Unit, tuple, bool]:
        """获取数组的单位、形状和是否为单位数组"""
        if isinstance(arg, u.Quantity):
            return arg.unit, arg.shape, False
        elif isinstance(arg, (list, tuple)):
            if len(arg) > 0:
                if isinstance(arg[0], u.Quantity):
                    return arg[0].unit, arg[0].shape, False
                elif isinstance(arg[0], (list, tuple)):
                    first_elem = arg[0]
                    while isinstance(first_elem, (list, tuple)) and len(first_elem) > 0:
                        first_elem = first_elem[0]
                    if isinstance(first_elem, (u.Quantity, u.UnitBase, u.CompositeUnit)):
                        return first_elem.unit if isinstance(first_elem, u.Quantity) else first_elem, first_elem.shape if isinstance(first_elem, u.Quantity) else (), False
        elif isinstance(arg, np.ndarray):
            if arg.dtype == np.dtype('object'):
                first_elem = arg.flat[0]
                if isinstance(first_elem, (u.Quantity, u.UnitBase, u.CompositeUnit)):
                    return first_elem.unit if isinstance(first_elem, u.Quantity) else first_elem, first_elem.shape if isinstance(first_elem, u.Quantity) else (), False
        return u.dimensionless_unscaled, (), True

    @staticmethod
    def _expand_shape(unit: u.Unit, shape: tuple) -> Union[u.Unit, List]:
        """根据形状扩展单位"""
        if not shape:  # 标量
            return unit
        if len(shape) == 1:  # 一维数组
            return [unit] * shape[0]
        # 多维数组：递归扩展
        return [UnitPropagator._expand_shape(unit, shape[1:]) for _ in range(shape[0])]

    @staticmethod
    def _flatten_units(unit_struct: Union[u.Unit, List]) -> List[u.Unit]:
        """将嵌套的单位结构展平"""
        if not isinstance(unit_struct, list):
            return [unit_struct]
        
        flattened = []
        for item in unit_struct:
            if isinstance(item, list):
                flattened.extend(UnitPropagator._flatten_units(item))
            else:
                flattened.append(item)
        return flattened

    @staticmethod
    def _process_array_op(func: callable, *args, **kwargs) -> u.Unit:
        """处理数组操作"""
        # 对于矩阵运算，需要处理两个输入
        if func in {np.dot, np.matmul, np.inner, np.outer}:
            if len(args) >= 2:
                unit1, shape1, _ = UnitPropagator._get_array_info(args[0])
                unit2, shape2, _ = UnitPropagator._get_array_info(args[1])
                # 矩阵运算的单位是两个输入单位的乘积
                return unit1 * unit2
        
        # 其他数组操作的处理保持不变
        unit, shape, is_unit_array = UnitPropagator._get_array_info(args[0])
        axis = kwargs.get('axis', None)
        
        if is_unit_array:
            if func in UnitPropagator.array_ops['multiply']:
                power = np.prod(shape) if axis is None else shape[axis]
                return unit ** power
            elif func in UnitPropagator.array_ops['square']:
                return unit ** 2
            return unit
        else:
            expanded_units = UnitPropagator._expand_shape(unit, shape)
            flat_units = UnitPropagator._flatten_units(expanded_units)
            
            for category, funcs in UnitPropagator.array_ops.items():
                if func in funcs:
                    if category == 'preserve':
                        return unit
                    elif category == 'multiply':
                        if axis is None:
                            return unit ** len(flat_units)
                        else:
                            return unit ** shape[axis]
                    elif category == 'square':
                        return unit ** 2
            return unit

    @staticmethod
    def _process_element_op(func: callable, units: List[u.Unit]) -> u.Unit:
        """处理元素间运算的函数"""
        print(f"\n=== _process_element_op ===")
        print(f"Function: {func}")
        print(f"Input units: {units}")
        
        if not units:
            return u.dimensionless_unscaled

        base_unit = units[0]
        result_unit = base_unit

        for category, funcs in UnitPropagator.element_ops.items():
            if func in funcs:
                if category == 'basic':
                    if func in {np.multiply, np.prod}:
                        for unit in units[1:]:
                            if isinstance(unit, (u.Unit, u.CompositeUnit)):
                                result_unit = result_unit * unit
                            elif isinstance(unit, (int, float)):
                                continue
                        print(f"Multiply result unit: {result_unit}")
                        return result_unit
                    elif func in {np.divide, np.true_divide, np.floor_divide}:
                        for unit in units[1:]:
                            if isinstance(unit, (u.Unit, u.CompositeUnit)):
                                result_unit = result_unit / unit
                            elif isinstance(unit, (int, float)):
                                continue
                        return result_unit
                    elif func in {np.add, np.subtract, np.remainder, np.mod}:
                        return base_unit
                elif category == 'dimensionless':
                    return u.dimensionless_unscaled
                elif category == 'preserve':
                    return base_unit
                elif category == 'power':
                    if func == np.sqrt:
                        return base_unit ** 0.5
                    elif func == np.square:
                        return base_unit ** 2
                    elif func == np.cbrt:
                        return base_unit ** (1/3)
                    elif func == np.power:
                        if len(units) > 1:
                            power = units[1]
                            if isinstance(power, u.Unit):
                                if power.is_equivalent(u.dimensionless_unscaled):
                                    power = power.to(u.dimensionless_unscaled).value
                            return base_unit ** power
                        return base_unit
        return base_unit

    @staticmethod
    def propagate(func: Union[callable, str], *args, **kwargs) -> u.Unit:
        """计算函数结果的单位
        
        Parameters
        ----------
        func : Union[callable, str]
            要分析的函数或函数名
        args : tuple
            位置参数
        simplify_units : bool, optional
            是否简化单位，默认False
        kwargs : dict
            关键字参数
            
        Returns
        -------
        Unit
            计算结果的单位
        """
        # 如果输入是字符串，尝试从numpy获取对应函数
        if isinstance(func, str):
            try:
                func = getattr(np, func)
            except AttributeError:
                raise ValueError(f"未找到numpy函数: {func}")

        # 检查函数是否在预定义的操作集中
        for category, funcs in UnitPropagator.array_ops.items():
            if func in funcs:
                result = UnitPropagator._process_array_op(func, *args, **kwargs)
                return result

        # 检查元素操作
        for category, funcs in UnitPropagator.element_ops.items():
            if func in funcs:
                result = UnitPropagator._process_element_op(func, [UnitPropagator._get_unit(arg) for arg in args])
                return result

        # 处理自定义函数
        try:
            unit_args = []
            for i, arg in enumerate(args):
                if isinstance(arg, (int, float)):
                    unit_args.append(arg)
                elif isinstance(arg, u.Quantity):
                    if np.isscalar(arg.value):
                        unit_args.append(1.0 * arg.unit)
                    else:
                        unit_args.append(arg)
                elif isinstance(arg, list) and all(isinstance(x, u.Quantity) for x in arg):
                    test_array = np.array([x.value for x in arg])
                    unit_args.append(test_array * arg[0].unit)
                else:
                    unit_args.append(1.0 * u.dimensionless_unscaled)

            result = func(*unit_args, **kwargs)

            if isinstance(result, u.Quantity):
                unit = result.unit
                return unit
            return u.dimensionless_unscaled
        except Exception as e:
            raise ValueError(f"单位计算错误: {str(e)}")
    
def get_common_unit(*args):
    """获取多个参数的共同单位
    
    Parameters
    ----------
    args : sequence
        输入序列，可以包含带不同单位的Quantity
        
    Returns
    -------
    Unit or None
        共同单位（第一个遇到的单位），如果没有单位则返回None
        
    Raises
    ------
    ValueError
        如果序列中包含不兼容的单位
    """
    # 处理空输入
    if not args:
        return None
        
    # 如果输入是单个元组，解包它
    if len(args) == 1:
        if isinstance(args[0], (tuple, list)):
            args = args[0]
        elif isinstance(args[0], u.Quantity):
            # 如果是单个Quantity，直接返回其单位
            return args[0].unit
    
    # 找到第一个带单位的参数
    first_quantity = next((arg for arg in args if isinstance(arg, u.Quantity)), None)
    if first_quantity is None:
        return None
        
    base_unit = first_quantity.unit
    
    # 检查所有带单位的参数是否兼容
    for arg in args:
        if isinstance(arg, u.Quantity):
            if not arg.unit.physical_type == base_unit.physical_type:
                raise ValueError(f"Incompatible units: {arg.unit} and {base_unit}")
    
    return base_unit

def convert_to_common_unit(*args, target_unit=None):
    """将多个值转换为相同单位
    
    Parameters
    ----------
    args : sequence
        需要转换的值
    target_unit : str, optional
        目标单位，如果为None则自动判断共同单位
        
    Returns
    -------
    tuple
        转换后的参数序列
    """
    # 处理空输入
    if not args:
        return args
        
    # 处理单个参数的情况
    if len(args) == 1:
        if isinstance(args[0], (tuple, list)):
            args = args[0]
        elif isinstance(args[0], u.Quantity):
            # 如果是单个Quantity，直接返回
            if target_unit:
                return args[0].to(target_unit)
            return args[0]
    
    base_unit = target_unit if target_unit else get_common_unit(args)
    if base_unit is None:
        return args
        
    converted_args = []
    for val in args:
        try:
            if hasattr(val, 'unit') and val.unit.physical_type == base_unit.physical_type:  # 已有单位的值
                converted_args.append(val.to(base_unit))
            else:  # 无单位的值
                converted_args.append(u.Quantity(val, base_unit))
        except:  # 无法处理的类型保持原样
            converted_args.append(val)
            
    return converted_args[0] if len(converted_args) == 1 else tuple(converted_args)

def convert_sequence_to_array(sequence: Union[List, tuple]) -> Union[np.ndarray, u.Quantity]:
    """Convert a sequence (list, tuple) to an array with units if applicable
    
    Parameters
    ----------
    sequence : Union[List, Tuple]
        Sequence of numbers or Quantities
        
    Returns
    -------
    Union[np.ndarray, Quantity] 
        - numpy array for regular numbers
        - Quantity array for sequences containing Quantities
        - Original input if it's already a Quantity
    
    Raises
    ------
    ValueError
        If units are inconsistent when Quantities are provided
    """
    # Return original if already a Quantity or not a sequence
    if isinstance(sequence, u.Quantity) or not isinstance(sequence, (list, tuple)):
        return sequence
    
    # Get common unit if any elements are Quantities
    unit = get_common_unit(sequence)
    
    if unit is not None:
        # Convert all elements to the common unit using convert_to_common_unit
        try:
            converted = convert_to_common_unit(sequence)
            # 确保converted是一个序列
            if not isinstance(converted, (tuple, list)):
                converted = [converted]
            # 提取值并创建数组
            values = [q.value if isinstance(q, u.Quantity) else q for q in converted]
            return np.array(values) * unit
        except Exception as e:
            # 对于无法处理的类型，抛出错误
            raise ValueError(f"All elements must be compatible with unit {unit}: {str(e)}")
    
    # If no units involved, return regular numpy array
    return np.array(sequence)

class QuantitySeparator:
    @staticmethod
    def convert_sequences(args: tuple, kwargs: dict) -> tuple[tuple, dict, list]:
        """转换所有序列参数并获取单位列表
        
        Parameters
        ----------
        args : tuple
            位置参数
        kwargs : dict
            关键字参数
            
        Returns
        -------
        tuple
            (处理后的args, 处理后的kwargs, 单位列表)
        """
        args = list(args)
        sequence_unit_list = []

        if kwargs is None:
            kwargs = {}
        
        # 处理位置参数
        for i in range(len(args)):
            if isinstance(args[i], (list, tuple)):
                converted = convert_sequence_to_array(args[i])
                if isinstance(converted, u.Quantity):
                    sequence_unit_list.append(converted.unit)
                else:
                    sequence_unit_list.append(None)
                args[i] = converted
            else:
                # 处理非序列类型
                if isinstance(args[i], u.Quantity):
                    sequence_unit_list.append(args[i].unit)
                else:
                    sequence_unit_list.append(None)
        
        # 处理关键字参数
        for key, value in kwargs.items():
            if isinstance(value, (list, tuple)):
                converted = convert_sequence_to_array(value)
                if isinstance(converted, u.Quantity):
                    sequence_unit_list.append(converted.unit)
                else:
                    sequence_unit_list.append(None)
                kwargs[key] = converted
            else:
                # 处理非序列类型
                if isinstance(value, u.Quantity):
                    sequence_unit_list.append(value.unit)
                else:
                    sequence_unit_list.append(None)
            
        return tuple(args), kwargs, sequence_unit_list
    
    @staticmethod
    def process_args(args: tuple, kwargs: dict, sequence_unit_list) -> tuple[list, dict, dict]:
        """处理参数，构建单位字典"""
        unit_dict = {}
        processed_args = []

        if kwargs is None:
            kwargs = {}
        
        # 处理位置参数
        for i, arg in enumerate(args):
            arg_name = f'arg_{i}'
            if isinstance(arg, u.Quantity):
                sequence_unit = sequence_unit_list[i]
                if sequence_unit and arg.unit.physical_type == sequence_unit.physical_type:
                    unit_dict[arg_name] = sequence_unit
                    processed_args.append(arg.to(sequence_unit).value)
                else:
                    unit_dict[arg_name] = arg.unit
                    processed_args.append(arg.value)
            else:
                processed_args.append(arg)
                
        # 处理关键字参数
        processed_kwargs = {}
        for j, (key, arg) in enumerate(kwargs.items()):
            if isinstance(arg, u.Quantity):
                sequence_unit = sequence_unit_list[len(args) + j]
                if sequence_unit and arg.unit.physical_type == sequence_unit.physical_type:
                    unit_dict[key] = sequence_unit
                    processed_kwargs[key] = arg.to(sequence_unit).value
                else:
                    unit_dict[key] = arg.unit
                    processed_kwargs[key] = arg.value
            else:
                processed_kwargs[key] = arg
                
        return processed_args, processed_kwargs, unit_dict
    
    @staticmethod
    def get_unit_dict(args: tuple, kwargs: dict = None) -> dict:
        """获取参数的单位字典
        
        Parameters
        ----------
        args : tuple or list
            位置参数
        kwargs : dict, optional
            关键字参数，默认为None
            
        Returns
        -------
        dict
            单位字典
        """
        if kwargs is None:
            kwargs = {}
            
        args, kwargs, sequence_unit_list = QuantitySeparator.convert_sequences(args, kwargs)
        processed_args, processed_kwargs, unit_dict = QuantitySeparator.process_args(
            args, kwargs, sequence_unit_list)
        return unit_dict

class QuantityWrapper:
    """用于包装numpy函数以支持quantity并自动追踪单位传播的类"""
    
    # 可以使用Quantity属性方法的numpy函数映射
    _quantity_methods = {
        'mean': 'mean',
        'std': 'std',
        'min': 'min',
        'max': 'max',
        'sum': 'sum',
        'median': 'median',
        'var': 'var',
        'ptp': 'ptp',  # peak to peak
        'round': 'round',
        'clip': 'clip',
        'conj': 'conj',
        'conjugate': 'conjugate'
    }
    
    @staticmethod
    def _apply_unit(result: Any, output_unit: u.Unit, propagated_unit: u.Unit,
                    simplify_units: bool) -> Any:
        """应用单位到结果
        
        Parameters
        ----------
        result : Any
            计算结果
        output_unit : Unit or None
            指定的输出单位
        propagated_unit : Unit or None
            传播的单位
        sequence_unit : Unit or None
            序列的单位
        simplify_units : bool
            是否简化单位
            
        Returns
        -------
        Any
            带单位的结果
        """
        if output_unit is not None:
            if propagated_unit and propagated_unit.physical_type == output_unit.physical_type:
                result = (result * propagated_unit).to(output_unit)
            else:
                result = result * output_unit
        elif propagated_unit is not None:
            result = result * propagated_unit
        if simplify_units and isinstance(result, u.Quantity):
            result = simplify_unit(result)
        return result
    
    @staticmethod
    def _try_quantity_input(func: Callable, *args, output_unit: Union[u.Unit, None] = None, 
                           **kwargs) -> tuple[Any, bool]:
        """尝试使用astropy的quantity_input处理
        
        Parameters
        ----------
        func : Callable
            要包装的函数
        args : tuple
            位置参数
        output_unit : Unit or None
            指定的输出单位
        kwargs : dict
            关键字参数
            
        Returns
        -------
        tuple[Any, bool]
            (结果, 是否成功处理)
        """
        try:
            # 对于numpy ufuncs，直接尝试调用
            if isinstance(func, np.ufunc):
                result = func(*args, **kwargs)
            else:
                # 对于其他函数，使用quantity_input包装
                @quantity_input
                def wrapped_func(*a, **kw):
                    return func(*a, **kw)
                result = wrapped_func(*args, **kwargs)
            
            # 如果指定了输出单位，尝试转换
            if output_unit is not None and isinstance(result, u.Quantity):
                try:
                    result = result.to(output_unit)
                except:
                    return None, False
                
            return result, True
            
        except Exception as e:
            return None, False
    
    @staticmethod
    def _try_quantity_method(func: Callable, args: tuple, 
                           output_unit: Union[u.Unit, None] = None,
                           **kwargs) -> tuple[Any, bool]:
        """尝试使用Quantity的内置方法"""
        try:
            # 检查是否可以使用Quantity方法
            if (func.__name__ in QuantityWrapper._quantity_methods and 
                len(args) == 1 and isinstance(args[0], u.Quantity)):
                
                # 获取对应的Quantity方法名
                method_name = QuantityWrapper._quantity_methods[func.__name__]
                method = getattr(args[0], method_name)
                
                # 获取方法的参数
                import inspect
                method_params = inspect.signature(method).parameters
                
                # 过滤出方法支持的关键字参数
                supported_kwargs = {}
                for key, value in kwargs.items():
                    if key in method_params:
                        supported_kwargs[key] = value
                
                # 调用方法，传入支持的参数
                result = method(**supported_kwargs)
                
                # 如果指定了输出单位，进行转换
                if output_unit is not None:
                    result = result.to(output_unit)
                    
                return result, True
                
        except Exception as e:
            #print(f"Method error: {e}")  # 可选的调试信息
            pass
            
        return None, False
    
    @staticmethod
    def wrap(func: Callable, *args, output_unit: Union[u.Unit, None] = None,
            simplify_units: bool = False, force_numpy: bool = False, **kwargs) -> Any:
        """包装numpy函数以支持quantity并自动追踪单位传播"""
        # 转换序列并获取第一个单位
        args, kwargs, sequence_unit_list = QuantitySeparator.convert_sequences(args, kwargs)
        
        # 检查是否有quantity
        all_args = list(args)
        for key, value in kwargs.items():
            all_args.append(value)
            
        has_quantity = any(isinstance(arg, u.Quantity) for arg in all_args)
        if not has_quantity and output_unit is None:
            return func(*args, **kwargs)
            
        # 如果没有强制使用numpy，尝试使用Quantity方法
        if not force_numpy:
            result, success = QuantityWrapper._try_quantity_method(
                func, args, output_unit, **kwargs)  # 传递kwargs
            if success:
                print('success')
                return simplify_unit(result) if simplify_units else result
        
        # 如果不能使用Quantity方法，使用原有的处理方式
        # 先尝试使用quantity_input
        result, success = QuantityWrapper._try_quantity_input(
            func, *args, output_unit=output_unit, **kwargs)
        if success:
            print('success')
            return simplify_unit(result) if simplify_units else result
            
        # 如果quantity_input失败，使用原有的处理方式
        processed_args, processed_kwargs, unit_dict = QuantitySeparator.process_args(
            args, kwargs, sequence_unit_list)
            
        # 获取传播单位 - 直接使用UnitPropagator.propagate
        propagated_unit = UnitPropagator.propagate(func, *args, **kwargs)
            
        # 计算结果
        result = func(*processed_args, **processed_kwargs)
        
        # 应用单位
        print('Use my own method')
        result = QuantityWrapper._apply_unit(result, output_unit, propagated_unit, simplify_units)
        return simplify_unit(result) if simplify_units else result

# 为了保持向后兼容，可以定义一个函数作为类方法的别名
def quantity_wrap(func: Callable, *args, **kwargs) -> Any:
    """quantity_wrap的别名，调用QuantityWrapper.wrap"""
    return QuantityWrapper.wrap(func, *args, **kwargs)

def example_usage():
    """测试unit_tools模块的功能"""
    
    def test_trigonometric():
        """测试三角函数"""
        print("\nTesting trigonometric functions:")
        print("---------------------------------")
        
        # 角度输入
        angles_deg = [30, 45, 60] * u.deg
        print(f"Sin of {angles_deg}: {quantity_wrap(np.sin, angles_deg)}")
        print(f"Cos of {angles_deg}: {quantity_wrap(np.cos, angles_deg)}")
        
        # 弧度输入
        angles_rad = [0, np.pi/4, np.pi/2] * u.rad
        print(f"Sin of {angles_rad}: {quantity_wrap(np.sin, angles_rad)}")

        angles = [30*u.deg, 45*u.deg, 60*u.deg]
        print(f"Sin of {angles}: {quantity_wrap(np.sin, angles)}")
    
    def test_mixed_units():
        """测试混合单位"""
        print("\nTesting mixed units:")
        print("--------------------")
        
        # 长度混合
        lengths = [1 * u.km, 2000 * u.m, 300000 * u.cm]
        print(f"Original lengths: {lengths}")
        print(f"Mean length: {quantity_wrap(np.mean, lengths)}")
        print(f"Mean length in meters: {quantity_wrap(np.mean, lengths, output_unit=u.m)}")
        
        # 时间混合
        times = [1 * u.hour, 3600 * u.s, 120 * u.minute]
        print(f"Mean time: {quantity_wrap(np.mean, times)}")
    
    def test_array_operations():
        """测试数组操作"""
        print("\nTesting array operations:")
        print("-----------------------")
        
        arr1 = [1, 2, 3] * u.m
        arr2 = [4, 5, 6] * u.s
        print(f"Array multiplication: {quantity_wrap(np.multiply, arr1, arr2)}")
        
        # 矩阵运算
        matrix = np.array([[1, 2], [3, 4]]) * u.m
        vector = np.array([2, 1]) * u.s
        print(f"Matrix-vector multiplication: {quantity_wrap(np.dot, matrix, vector)}")
    
    def test_statistical():
        """测试统计函数"""
        print("\nTesting statistical functions:")
        print("-----------------------------")
        
        values = [10, 20, 30, 40, 50] * u.kg
        print(f"Values: {values}")
        print(f"Mean: {quantity_wrap(np.mean, values)}")
        print(f"Std: {quantity_wrap(np.std, values)}")
        print(f"Max: {quantity_wrap(np.max, values)}")
        
        # 带权重的平均
        weights = [1, 2, 3, 4, 5]  # 无单位权重
        print(f"Weighted mean: {quantity_wrap(np.average, values, weights=weights)}")
    
    def test_math_operations():
        """测试数学运算"""
        print("\nTesting mathematical operations:")
        print("-------------------------------")
        
        # 指数运算 - 修改输入为平方单位
        value = 100 * u.m**2  # 改为平方米
        print(f"Square root of {value}: {quantity_wrap(np.sqrt, value)}")
        
        # 绝对值
        values = [-1, 2, -3] * u.m
        print(f"Absolute values of {values}: {quantity_wrap(np.abs, values)}")
        
        # 对数（无单位）
        ratios = [0.1, 1, 10] * u.dimensionless_unscaled
        print(f"Log of {ratios}: {quantity_wrap(np.log10, ratios)}")
    
    def test_mixed_quantity_number():
        """测试量纲和无量纲混合"""
        print("\nTesting mixed Quantity and number operations:")
        print("-----------------------------------------")
        
        # 数组乘以标量
        arr = [1, 2, 3] * u.m
        scalar = 2  # 无单位
        print(f"{arr} * {scalar} = {quantity_wrap(np.multiply, arr, scalar)}")
        
        # 混合数组运算
        arr1 = [1, 2, 3] * u.m
        arr2 = [4, 5, 6]  # 无单位
        print(f"{arr1} * {arr2} = {quantity_wrap(np.multiply, arr1, arr2)}")
    
    def test_unit_conversion():
        """测试单位转换"""
        print("\nTesting unit conversion:")
        print("----------------------")
        
        # 速度转换
        velocity = [10, 20, 30] * u.km/u.s
        print(f"Original velocity: {velocity}")
        print(f"Converted to m/s: {quantity_wrap(np.mean, velocity, output_unit=u.m/u.s)}")
        
    
    def test_simplify_units():
        """测试单位简化功能"""
        print("\n=== 测试单位简化 ===")
        
        # 创建一些复杂单位的数据
        velocity = [10, 20] * (u.km/u.hour)
        acceleration = [1, 2] * (u.km/u.hour/u.s)
        energy = [100, 200] * (u.kg * u.km**2 / u.hour**2)
        
        # 测试 QuantityWrapper.wrap
        print("\n测试 QuantityWrapper.wrap:")
        # 不简化单位
        result = quantity_wrap(np.multiply, velocity, acceleration)
        print(f"不简化单位: {result.unit}")
        
        # 简化单位
        result = quantity_wrap(np.multiply, velocity, acceleration, simplify_units=True)
        print(f"简化单位: {result.unit}")
        
        # 测试复杂运算
        print("\n测试复杂运算:")
        # 不简化单位
        result = quantity_wrap(np.multiply, energy, velocity)
        print(f"不简化单位: {result.unit}")
        
        # 简化单位
        result = quantity_wrap(np.multiply, energy, velocity, simplify_units=True)
        print(f"简化单位: {result.unit}")
    
    def test_sum_behavior():
        """测试quantity_wrap对np.sum的处理"""
        print("\n=== 测试np.sum的行为 ===")
        
        # 创建测试数据
        arr1 = np.array([[0.04, 0.04],
                         [0.04, 0.04]]) * u.m**2
        arr2 = np.array([[0.04, 0.16],
                         [0.36, 0.64]]) * u.m**2
        test_list = [arr1, arr2]
        
        print("\n原始数据:")
        print(f"arr1:\n{arr1}")
        print(f"arr2:\n{arr2}")
        
        print("\n直接使用numpy:")
        # 先堆叠再求和
        stacked = np.stack([arr1.value, arr2.value])
        print(f"Stacked shape: {stacked.shape}")
        numpy_sum = np.sum(stacked, axis=0) * u.m**2
        print(f"Numpy sum:\n{numpy_sum}")
        
        print("\n使用quantity_wrap:")
        # 直接使用quantity_wrap
        wrapped_sum = quantity_wrap(np.sum, test_list, axis=0)
        print(f"Wrapped sum:\n{wrapped_sum}")
        
        print("\n使用quantity_wrap和stack:")
        # 先堆叠再使用quantity_wrap
        stacked_arrays = quantity_wrap(np.stack, test_list, axis=0)
        print(f"Stacked shape: {stacked_arrays.shape}")
        wrapped_stacked_sum = quantity_wrap(np.sum, stacked_arrays, axis=0)
        print(f"Wrapped stacked sum:\n{wrapped_stacked_sum}")
    
    def test_matrix_multiply():
        """测试矩阵乘法的单位处理"""
        print("\n=== 测试矩阵乘法 ===")
        
        # 创建测试数据
        A = np.array([[1, 2], [3, 4]]) * u.m
        B = np.array([[2, 0], [0, 2]]) * u.s
        
        print("\n测试 np.multiply (逐元素乘法):")
        result1 = quantity_wrap(np.multiply, A, B)
        print(f"A * B =\n{result1}")
        
        print("\n测试 np.matmul (矩阵乘法):")
        result2 = quantity_wrap(np.matmul, A, B)
        print(f"A @ B =\n{result2}")
        
    
    # 运行所有测试
    test_simplify_units()
    test_trigonometric()
    test_mixed_units()
    test_array_operations()
    test_statistical()
    test_math_operations()
    test_mixed_quantity_number()
    test_unit_conversion()
    test_sum_behavior()
    test_matrix_multiply()


def test_image_operations():
    
    # 创建测试数据
    img1 = np.array([[1, 2], [3, 4]]) * u.count
    img2 = np.array([[2, 3], [4, 5]]) * u.count
    imgs = [img1, img2]  # 图像序列
    scale_factor = 2.5   # 无单位标量
    bias = 10 * u.count  # 有单位标量
    
    print("\n=== 基本图像运算测试 ===")
    # 1. 图像加减
    add_unit = UnitPropagator.propagate(np.add, img1, img2)
    print(f"图像相加单位: {add_unit}")
    
    sub_unit = UnitPropagator.propagate(np.subtract, img1, img2)
    print(f"图像相减单位: {sub_unit}")
    
    # 2. 缩放和偏移
    scale_unit = UnitPropagator.propagate(np.multiply, img1, scale_factor)
    print(f"图像缩放单位: {scale_unit}")
    
    bias_unit = UnitPropagator.propagate(np.add, img1, bias)
    print(f"图像加偏移单位: {bias_unit}")
    
    print("\n=== 统计运算测试 ===")
    # 3. 单幅图像统计
    mean_unit = UnitPropagator.propagate(np.mean, img1)
    print(f"均值单位: {mean_unit}")
    
    std_unit = UnitPropagator.propagate(np.std, img1)
    print(f"标准差单位: {std_unit}")
    
    # 4. 多幅图像统计
    def stack_median(imgs):
        """计算图像序列的逐像素中值"""
        return np.median(imgs, axis=0)
    
    median_unit = UnitPropagator.propagate(stack_median, imgs)
    print(f"图像序列中值单位: {median_unit}")
    
    print("\n=== 图像形状操作测试 ===")
    # 5. 形状操作
    reshape_unit = UnitPropagator.propagate(np.reshape, img1)
    print(f"重整形状单位: {reshape_unit}")
    
    stack_unit = UnitPropagator.propagate(np.vstack, imgs)
    print(f"堆叠图像单位: {stack_unit}")
    
    # 6. 复合运算
    def normalize_image(img, background):
        """减去背景并归一化"""
        return (img - background) / background
    
    norm_unit = UnitPropagator.propagate(normalize_image, img1, bias)
    print(f"\n归一化图像单位: {norm_unit}")

def test_unit_propagation():
    """测试UnitPropagator对矩阵乘法的单位传播"""
    print("\n=== 测试UnitPropagator单位传播 ===")
    
    # 创建测试数据
    A = np.array([[1, 2], [3, 4]]) * u.m
    B = np.array([[2, 0], [0, 2]]) * u.m
    
    print("\n方法1: 在函数定义中指定axis:")
    def multiply_func1(*values):
        """在函数内部指定axis=0"""
        print(f"multiply_func1输入: {values}")
        result = np.prod(values, axis=0)
        print(f"multiply_func1结果: {result}")
        return result
    
    print("\n方法2: 通过kwargs传入axis:")
    def multiply_func2(*values, **kwargs):
        """通过kwargs接收axis参数"""
        print(f"multiply_func2输入: {values}")
        print(f"multiply_func2 kwargs: {kwargs}")
        result = np.prod(values, **kwargs)
        print(f"multiply_func2结果: {result}")
        return result
    
    print("\n方法3: 不指定axis:")
    def multiply_func3(*values):
        """不指定axis，让numpy自动处理"""
        print(f"multiply_func3输入: {values}")
        result = np.prod(values)
        print(f"multiply_func3结果: {result}")
        return result
    
    # 测试方法1
    print("\n测试方法1:")
    unit1 = UnitPropagator.propagate(multiply_func1, A, B)
    print(f"方法1传播单位: {unit1}")
    
    # 测试方法2
    print("\n测试方法2:")
    unit2 = UnitPropagator.propagate(multiply_func2, A, B, axis=0)
    print(f"方法2传播单位: {unit2}")
    
    # 测试方法3
    print("\n测试方法3:")
    unit3 = UnitPropagator.propagate(multiply_func3, A, B)
    print(f"方法3传播单位: {unit3}")
    
    # 直接调用测试
    print("\n直接调用测试:")
    result1 = multiply_func1(A, B)
    result2 = multiply_func2(A, B, axis=0)
    result3 = multiply_func3(A, B)
    print(f"方法1结果单位: {getattr(result1, 'unit', None)}")
    print(f"方法2结果单位: {getattr(result2, 'unit', None)}")
    print(f"方法3结果单位: {getattr(result3, 'unit', None)}")

if __name__ == '__main__':
    test_image_operations()
    example_usage()
    test_unit_propagation()
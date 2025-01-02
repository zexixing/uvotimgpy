from typing import List, Union, Callable, Any, Sequence, Dict
import numpy as np
import astropy.units as u

import ast
import inspect
import textwrap

import ast
import inspect
import textwrap
from astropy import units as u
import numpy as np

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
    
    def __init__(self, simplify_units: bool = True):
        """
        Parameters
        ----------
        simplify_units : bool, optional
            是否自动简化单位，默认为True
        """
        self.simplify_units = simplify_units
    
    @classmethod
    def propagate_units(cls, func, unit_dict, simplify_units: bool = None):
        """分析函数并返回结果的单位"""
        source = inspect.getsource(func)
        source = textwrap.dedent(source)
        
        tree = ast.parse(source)
        func_body = tree.body[0].body
        
        if isinstance(func_body[-1], ast.Return):
            return_expr = func_body[-1].value
            result_unit = cls._eval_node(return_expr, unit_dict)
            
            if result_unit is not None and (simplify_units or simplify_units is None):
                return simplify_unit(result_unit)  # 直接使用独立的simplify_unit函数
            return result_unit
        return None

    # numpy函数的单位传播规则字典
    _np_unit_rules = {
        # 1. 基本数学运算：返回None（无量纲）
        'exp': lambda x: None,
        'log': lambda x: None,
        'log10': lambda x: None,
        
        # 2. 三角函数：返回None（无量纲）
        'sin': lambda x: None,
        'cos': lambda x: None,
        'tan': lambda x: None,
        'arcsin': lambda x: None,
        'arccos': lambda x: None,
        'arctan': lambda x: None,
        'sinh': lambda x: None,
        'cosh': lambda x: None,
        'tanh': lambda x: None,
        
        # 3. 基本运算：保持或组合单位
        'sqrt': lambda x: x**0.5 if x is not None else None,
        'abs': lambda x: x,
        'multiply': lambda x, y: x * y if x is not None and y is not None else None,
        'prod': lambda x, y: x * y if x is not None and y is not None else None,
        'divide': lambda x, y: x / y if x is not None and y is not None else None,
        
        # 4. 统计函数：保持或修改单位
        'sum': lambda x: x,
        'mean': lambda x: x,
        'median': lambda x: x,
        'min': lambda x: x,
        'max': lambda x: x,
        'std': lambda x: x,
        'var': lambda x: x**2,
        'average': lambda x, weights=None: x,
        
        # 5. 矩阵运算：组合单位
        'dot': lambda x, y: x * y if x is not None and y is not None else None,
        'matmul': lambda x, y: x * y if x is not None and y is not None else None,
        
        # 6. 数组操作：保持单位
        'concatenate': lambda x: x,
        'stack': lambda x: x,
        'vstack': lambda x: x,
        'hstack': lambda x: x,
        
        # 7. 概率和比较函数：返回None（无量纲）
        'prob': lambda x: None,
        'cdf': lambda x: None,
        'pdf': lambda x: None,
        'pmf': lambda x: None,
        'greater': lambda x, y: None,
        'less': lambda x, y: None,
        'equal': lambda x, y: None,
    }

    # 可以添加一些辅助方法来简化规则的使用
    @staticmethod
    def _get_unit_rule(func_name: str) -> Callable:
        """获取函数的单位传播规则"""
        return UnitPropagator._np_unit_rules.get(func_name)

    @staticmethod
    def _is_dimensionless_func(func_name: str) -> bool:
        """检查函数是否返回无量纲结果"""
        rule = UnitPropagator._get_unit_rule(func_name)
        if rule:
            # 使用None作为测试输入，如果返回None则为无量纲函数
            return rule(u.m) is None
        return False

    @staticmethod
    def _is_unit_preserving_func(func_name: str) -> bool:
        """检查函数是否保持输入单位"""
        rule = UnitPropagator._get_unit_rule(func_name)
        if rule:
            test_unit = u.m
            return rule(test_unit) == test_unit
        return False

    @staticmethod
    def _eval_binary_op(node, units):
        """
        评估二元运算符的单位传播
        
        参数：
        node (ast.BinOp): AST二元运算节点
        units (dict): 当前的单位字典
        
        返回：
        astropy.units.Unit: 运算结果的单位
        """
        left = UnitPropagator._eval_node(node.left, units)
        right = UnitPropagator._eval_node(node.right, units)
        
        # 处理加法和减法：需要相同单位
        if isinstance(node.op, (ast.Add, ast.Sub)):
            if left is None or right is None:
                return None
            # 检查单位是否兼容
            if left.is_equivalent(right):
                return left
            else:
                raise ValueError(f"Cannot add/subtract incompatible units {left} and {right}")
        
        # 处理乘法：单位相乘
        elif isinstance(node.op, ast.Mult):
            if left is None:
                return right
            if right is None:
                return left
            return left * right
        
        # 处理除法：单位相除    
        elif isinstance(node.op, ast.Div):
            if left is None:
                return 1/right if right is not None else None
            if right is None:
                return left
            return left / right
        
        # 处理幂运算：指数必须无单位    
        elif isinstance(node.op, ast.Pow):
            if right is None:
                if left is not None:
                    # 获取指数的数值
                    if isinstance(node.right, ast.Num):
                        return left ** node.right.n
                    else:
                        raise ValueError("Cannot evaluate non-numeric exponent")
                return None
            else:
                raise ValueError("Exponent cannot have units")
                
        return None

    @staticmethod
    def _eval_node(node, units):
        """
        递归评估AST节点的单位
        
        参数：
        node (ast.AST): AST节点
        units (dict): 当前的单位字典
        
        返回：
        astropy.units.Unit: 节点表达式的单位
        """
        # 处理变量名
        if isinstance(node, ast.Name):
            return units.get(node.id)
        
        # 处理数字字面量    
        elif isinstance(node, ast.Num):
            return None
        
        # 处理二元运算    
        elif isinstance(node, ast.BinOp):
            return UnitPropagator._eval_binary_op(node, units)
        
        # 处理函数调用    
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                # 直接函数调用 (如 exp(x))
                func_name = node.func.id
                return UnitPropagator._handle_function_call(func_name, node.args, units)
            elif isinstance(node.func, ast.Attribute):
                # 带命名空间的函数调用 (如 np.exp(x))
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id in ['np', 'numpy']:
                        func_name = node.func.attr
                        return UnitPropagator._handle_function_call(func_name, node.args, units)
        
        return None

    @staticmethod
    def _handle_function_call(func_name, args, units):
        """处理函数调用的单位传播"""
        rule = UnitPropagator._get_unit_rule(func_name)
        if rule is None:
            return None
            
        args_units = [UnitPropagator._eval_node(arg, units) for arg in args]
        
        # 处理无参数的情况
        if not args_units:
            return None
            
        # 处理特殊函数
        if func_name == 'average':
            # 检查权重是否无量纲
            if len(args_units) > 1 and args_units[1] is not None:
                raise ValueError("Weights in average function should be dimensionless")
            return rule(args_units[0])
            
        # 处理单参数函数
        if len(args_units) == 1:
            return rule(args_units[0])
            
        # 处理二元运算
        if len(args_units) == 2:
            return rule(args_units[0], args_units[1])
            
        # 处理多参数情况
        result_unit = args_units[0]
        for unit in args_units[1:]:
            if func_name in ['multiply', 'prod']:
                result_unit = result_unit * unit if unit is not None else result_unit
            elif func_name == 'divide':
                result_unit = result_unit / unit if unit is not None else result_unit
            elif UnitPropagator._is_unit_preserving_func(func_name):
                # 对于保持单位的函数，确保所有参数单位相同
                if unit is not None and not unit.is_equivalent(result_unit):
                    raise ValueError(f"All arguments to {func_name} must have compatible units")
        
        return result_unit

    
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
    def _convert_sequences(args: tuple, kwargs: dict) -> tuple[tuple, dict, u.Unit]:
        """转换所有序列参数并获取第一个单位
        
        Parameters
        ----------
        args : tuple
            位置参数
        kwargs : dict
            关键字参数
            
        Returns
        -------
        tuple
            (处理后的args, 处理后的kwargs, 第一个遇到的单位)
        """
        args = list(args)
        sequence_unit = None
        
        # 处理位置参数
        for i in range(len(args)):
            if isinstance(args[i], (list, tuple)):
                converted = convert_sequence_to_array(args[i])
                if isinstance(converted, u.Quantity) and sequence_unit is None:
                    sequence_unit = converted.unit
                args[i] = converted
        
        # 处理关键字参数
        for key, value in kwargs.items():
            if isinstance(value, (list, tuple)):
                converted = convert_sequence_to_array(value)
                if isinstance(converted, u.Quantity) and sequence_unit is None:
                    sequence_unit = converted.unit
                kwargs[key] = converted
                
        return tuple(args), kwargs, sequence_unit
    
    @staticmethod
    def _process_args(args: tuple, kwargs: dict, sequence_unit: u.Unit) -> tuple[list, dict, dict]:
        """处理参数，构建单位字典
        
        Parameters
        ----------
        args : tuple
            位置参数
        kwargs : dict
            关键字参数
        sequence_unit : Unit
            基准单位
            
        Returns
        -------
        tuple
            (处理后的args值列表, 处理后的kwargs值字典, 单位字典)
        """
        unit_dict = {}
        processed_args = []
        
        # 处理位置参数
        for i, arg in enumerate(args):
            arg_name = f'arg_{i}'
            if isinstance(arg, u.Quantity):
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
        for key, arg in kwargs.items():
            if isinstance(arg, u.Quantity):
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
    def _get_propagated_unit(func: Callable, unit_dict: dict, 
                           all_args: list, simplify_units: bool) -> u.Unit:
        """获取传播的单位
        
        Parameters
        ----------
        func : Callable
            要调用的函数
        unit_dict : dict
            单位字典
        all_args : list
            所有参数列表
        simplify_units : bool
            是否简化单位
            
        Returns
        -------
        Unit or None
            传播后的单位
        """
        # 创建包装函数
        def wrapped_func(*wrapped_args, **wrapped_kwargs):
            return func(*wrapped_args, **wrapped_kwargs)
            
        # 获取函数名
        func_name = func.__name__
        if hasattr(func, '__module__') and 'numpy' in func.__module__:
            func_name = func.__name__
            
        # 尝试使用UnitPropagator分析单位
        propagated_unit = UnitPropagator.propagate_units(wrapped_func, unit_dict, simplify_units)
        
        # 如果分析失败，使用预定义规则
        if propagated_unit is None and func_name in UnitPropagator._np_unit_rules:
            base_units = [arg.unit for arg in all_args if isinstance(arg, u.Quantity)]
            if base_units:
                propagated_unit = UnitPropagator._np_unit_rules[func_name](*base_units)
                if simplify_units:
                    propagated_unit = simplify_unit(propagated_unit)
        return propagated_unit
    
    @staticmethod
    def _apply_unit(result: Any, output_unit: u.Unit, propagated_unit: u.Unit,
                   sequence_unit: u.Unit, simplify_units: bool) -> Any:
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
            if sequence_unit and sequence_unit.physical_type == output_unit.physical_type:
                result = (result * sequence_unit).to(output_unit)
            else:
                result = result * output_unit
        elif propagated_unit is not None:
            result = result * propagated_unit
        elif sequence_unit is not None:
            result = result * sequence_unit
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
                           output_unit: Union[u.Unit, None] = None) -> tuple[Any, bool]:
        """尝试使用Quantity的内置方法
        
        Parameters
        ----------
        func : Callable
            要调用的函数
        args : tuple
            位置参数
        output_unit : Unit or None
            指定的输出单位
            
        Returns
        -------
        tuple[Any, bool]
            (结果, 是否成功使用Quantity方法)
        """
        try:
            # 检查是否可以使用Quantity方法
            if (func.__name__ in QuantityWrapper._quantity_methods and 
                len(args) == 1 and isinstance(args[0], u.Quantity)):
                
                # 获取对应的Quantity方法名
                method_name = QuantityWrapper._quantity_methods[func.__name__]
                result = getattr(args[0], method_name)()
                
                # 如果指定了输出单位，进行转换
                if output_unit is not None:
                    result = result.to(output_unit)
                    
                return result, True
                
        except Exception as e:
            pass
            
        return None, False
    
    @staticmethod
    def wrap(func: Callable, *args, output_unit: Union[u.Unit, None] = None,
            simplify_units: bool = False, force_numpy: bool = False, **kwargs) -> Any:
        """包装numpy函数以支持quantity并自动追踪单位传播
        
        Parameters
        ----------
        func : Callable
            要包装的函数
        args : tuple
            位置参数
        output_unit : Unit or None, optional
            指定的输出单位
        simplify_units : bool, optional
            是否简化单位，默认False
        force_numpy : bool, optional
            是否强制使用numpy函数而不是Quantity方法，默认False
        kwargs : dict
            关键字参数
        """
        # 转换序列并获取第一个单位
        args, kwargs, sequence_unit = QuantityWrapper._convert_sequences(args, kwargs)
        
        # 检查是否有quantity
        all_args = list(args) + list(kwargs.values())
        has_quantity = any(isinstance(arg, u.Quantity) for arg in all_args)
        if not has_quantity and output_unit is None:
            return func(*args, **kwargs)
            
        # 如果没有强制使用numpy，尝试使用Quantity方法
        if not force_numpy:
            result, success = QuantityWrapper._try_quantity_method(
                func, args, output_unit)
            if success:
                print('success')
                return result
        
        # 如果不能使用Quantity方法，使用原有的处理方式
        # 先尝试使用quantity_input
        result, success = QuantityWrapper._try_quantity_input(
            func, *args, output_unit=output_unit, **kwargs)
        if success:
            print('success')
            return result
        
        print(f'Using my own codes')
            
        # 如果quantity_input失败，使用原有的处理方式
        processed_args, processed_kwargs, unit_dict = QuantityWrapper._process_args(
            args, kwargs, sequence_unit)
            
        # 获取传播单位
        propagated_unit = QuantityWrapper._get_propagated_unit(
            func, unit_dict, all_args, simplify_units)
            
        # 计算结果
        result = func(*processed_args, **processed_kwargs)
        
        # 应用单位
        return QuantityWrapper._apply_unit(
            result, output_unit, propagated_unit, sequence_unit, simplify_units)

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
        
        # 指数运算
        value = 100 * u.m
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
        
        # 温度转换
        temp = [0, 100] * u.deg_C
        print(f"Original temperature: {temp}")
        print(f"Converted to Kelvin: {quantity_wrap(np.mean, temp, output_unit=u.K)}")
    
    # 运行所有测试
    test_trigonometric()
    test_mixed_units()
    test_array_operations()
    test_statistical()
    test_math_operations()
    test_mixed_quantity_number()
    test_unit_conversion()

if __name__ == "__main__":
    example_usage()
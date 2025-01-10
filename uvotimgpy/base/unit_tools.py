from typing import List, Union, Callable, Any, Sequence, Dict, Tuple, Optional
import numpy as np
import astropy.units as u
from astropy.units import quantity_input
import ast
import inspect
from functools import reduce


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
    class NodeVisitor(ast.NodeVisitor):
        def __init__(self):
            self.nodes = []

        def visit_Return(self, node):
            self.nodes.append(('return', node))
            self.generic_visit(node)

        def visit_Assign(self, node):
            self.nodes.append(('assign', node))
            self.generic_visit(node)

        def visit_Call(self, node):
            self.nodes.append(('call', node))
            self.generic_visit(node)

        def visit_BinOp(self, node):
            self.nodes.append(('binop', node))
            self.generic_visit(node)

    @staticmethod
    def _get_unit_from_arg(arg: Any) -> Union[u.Unit, List[u.Unit]]:
        if isinstance(arg, (int, float)):
            return u.dimensionless_unscaled
        elif isinstance(arg, u.Quantity) and not isinstance(arg, np.ndarray):
            return arg.unit
        elif isinstance(arg, u.Quantity) and isinstance(arg, np.ndarray):
            return [arg.unit for x in arg]
        elif isinstance(arg, (list, tuple, np.ndarray)):
            return [UnitPropagator._get_unit_from_arg(x) for x in arg]
        else:
            return u.dimensionless_unscaled

    @staticmethod
    def _get_units_from_args(*args) -> List[Union[u.Unit, List[u.Unit]]]:
        """从所有参数获取单位列表"""
        return [UnitPropagator._get_unit_from_arg(arg) for arg in args]

    def _is_uniform_unit_array(arg) -> bool:
        """
        检查是否是由相同 unit 组成的 array

        Parameters
        ----------
        arg : Any
            要检查的参数

        Returns
        -------
        bool
            如果是由相同 unit 组成的 array 返回 True，否则返回 False

        Examples
        --------
        >>> arr1 = np.array([u.m * u.s, u.m * u.s], dtype=object)
        >>> _is_uniform_unit_array(arr1)  # True

        >>> arr2 = np.array([u.m * u.s, u.m], dtype=object)
        >>> _is_uniform_unit_array(arr2)  # False

        >>> arr3 = np.array([1, 2, 3])
        >>> _is_uniform_unit_array(arr3)  # False
        """
        # 检查是否是 numpy array
        if not isinstance(arg, np.ndarray):
            return False

        # 检查 dtype 是否是 object
        if arg.dtype != object:
            return False

        # 如果是空数组，返回 False
        if arg.size == 0:
            return False

        # 获取第一个元素作为参考
        first_unit = arg.flat[0]

        # 检查第一个元素是否是 unit
        if not isinstance(first_unit, (u.Unit, u.CompositeUnit, u.IrreducibleUnit)):
            return False

        # 检查所有元素是否都是 unit 且与第一个相同
        return all(isinstance(x, (u.Unit, u.CompositeUnit, u.IrreducibleUnit)) 
                  and x == first_unit for x in arg.flat)

    @staticmethod
    def _try_compute_unit(func: callable, units: List[Union[u.Unit, List[u.Unit]]], **kwargs) -> u.Unit:
        """尝试计算函数结果的单位"""
        # 特殊处理 sqrt 函数
        if func == np.sqrt:
            if len(units) == 1:
                result = units[0] ** 0.5
                return result
        result = UnitPropagator._special_func(func, *units, **kwargs)
        if result:
            return result
        else:
            try:
                if len(units) == 1 and isinstance(units[0], list):
                    result = func(units[0], **kwargs)
                else:
                    result = func(*units, **kwargs)     
                if isinstance(result, (u.Unit, u.CompositeUnit, u.IrreducibleUnit)):
                    return result
                elif isinstance(result, u.Quantity):
                    return result.unit
                elif UnitPropagator._is_uniform_unit_array(result):
                    return result[0]
                else:
                    return u.dimensionless_unscaled

            except Exception as e:
                # 如果失败，尝试样本值方法
                try:
                    sample_args = []
                    for unit in units:
                        if isinstance(unit, list):
                            sample_args.append([1 * i if i != u.dimensionless_unscaled else 1*u.dimensionless_unscaled for i in unit])
                        else:
                            sample_args.append(1 * unit if unit != u.dimensionless_unscaled else 1*u.dimensionless_unscaled)  
                except Exception as e:
                    raise ValueError(f"无法计算单位: {str(e)}")
                try:
                    if len(units) == 1 and isinstance(units[0], list):
                        result = func(sample_args[0], **kwargs)
                    else:
                        result = func(*sample_args, **kwargs)
                    return getattr(result, 'unit', u.dimensionless_unscaled)
                except Exception as e:
                    try: 
                        sample_args = [u.Quantity(i) for i in sample_args]
                        result = func(*sample_args, **kwargs)
                        if isinstance(result, np.ndarray):
                            if isinstance(result.flat[0], u.Quantity):
                                if all(q == result.flat[0] for q in result.flat):
                                    return getattr(result.flat[0], 'unit', u.dimensionless_unscaled)
                    except Exception as e:
                        raise ValueError(f"无法计算单位: {str(e)}")

    @staticmethod
    def _eval_binop(op, left_unit, right_unit) -> u.Unit:
        """计算二元运算的单位"""

        if isinstance(op, ast.Add) or isinstance(op, ast.Sub):
            if left_unit != right_unit:
                raise ValueError(f"加减运算的单位必须相同: {left_unit} vs {right_unit}")
            return left_unit

        elif isinstance(op, ast.Mult):
            return left_unit * right_unit

        elif isinstance(op, ast.Div):
            return left_unit / right_unit

        elif isinstance(op, ast.Pow):
            # 处理幂运算
            if right_unit != u.dimensionless_unscaled:
                raise ValueError("指数必须是无量纲的")
            else:
                power = right_unit.value
                # 否则假设是单位为1的量
                return left_unit ** power

        else:
            raise ValueError(f"不支持的运算符: {type(op).__name__}")

    @staticmethod
    def _eval_node(node: ast.AST, local_dict: dict, unit_dict: dict) -> u.Unit:
        """计算节点的单位"""
        if isinstance(node, (ast.Constant, ast.Num)):  # ast.Num 是为了兼容旧版本
            # 存储原始数值
            unit = u.dimensionless_unscaled
            unit.value = node.value if hasattr(node, 'value') else node.n
            return unit

        elif isinstance(node, ast.Name):
            # 优先从unit_dict中获取单位
            if node.id in unit_dict:
                result = unit_dict[node.id]
            else:
                result = UnitPropagator._get_unit_from_arg(local_dict.get(node.id))
            return result

        elif isinstance(node, ast.Call):
            # 获取函数
            if isinstance(node.func, ast.Name):
                func = local_dict.get(node.func.id)
                if func is None and node.func.id == 'reduce':
                    func = reduce
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'np':
                    func = getattr(np, node.func.attr, None)
            else:
                raise ValueError(f"不支持的函数调用形式: {ast.dump(node.func)}")

            if func is None:
                raise ValueError(f"未找到函数: {ast.dump(node.func)}")

            # 计算参数的单位
            arg_units = [UnitPropagator._eval_node(arg, local_dict, unit_dict) for arg in node.args]

            # 处理特殊统计函数
            result = UnitPropagator._special_func(func, *arg_units)
            if result:
                return result
            # 计算函数结果的单位
            result = UnitPropagator._try_compute_unit(func, arg_units)
            return result

        elif isinstance(node, ast.BinOp):
            left_unit = UnitPropagator._eval_node(node.left, local_dict, unit_dict)
            right_unit = UnitPropagator._eval_node(node.right, local_dict, unit_dict)
            result = UnitPropagator._eval_binop(node.op, left_unit, right_unit)
            return result

        elif isinstance(node, ast.List):
            return [UnitPropagator._eval_node(elt, local_dict, unit_dict) for elt in node.elts]

        elif isinstance(node, ast.Attribute):
            # 处理属性访问，比如 u.m
            if isinstance(node.value, ast.Name):
                if node.value.id == 'u':
                    # 如果是 astropy.units 的单位
                    try:
                        result = getattr(u, node.attr)
                        return result
                    except AttributeError:
                        raise ValueError(f"未知的单位: u.{node.attr}")
                else:
                    base_obj = local_dict.get(node.value.id)
                    if base_obj is not None:
                        result = UnitPropagator._get_unit_from_arg(getattr(base_obj, node.attr, None))
                        return result
            # 处理 np.xxx 的情况
            elif isinstance(node.value, ast.Name) and node.value.id == 'np':
                return u.dimensionless_unscaled
        else:
            raise ValueError(f"不支持的节点类型: {type(node).__name__}")
    @staticmethod
    def _special_func(func: callable, *args, **kwargs) -> u.Unit:
        if func in [np.mean, np.median, np.std, np.vstack, np.sum, np.stack]:
            if len(args) == 1 and isinstance(args[0], list):
                unit_list = args[0]
                # 检查是否所有单位都相同
                if all(u == unit_list[0] for u in unit_list):
                    if isinstance(unit_list[0], list):
                        return unit_list[0][0]
                    else:
                        return unit_list[0]
        if func in [np.reshape]:
            if len(args) == 1:
                if isinstance(args[0], list):
                    return args[0][0]
                else:
                    return args[0]
        return False
    
    @staticmethod
    def propagate(func: callable, *args, **kwargs) -> u.Unit:
        """计算函数结果的单位"""
        try:
            # 如果是numpy函数或其他内置函数，直接计算
            if isinstance(func, np.ufunc) or (hasattr(func, '__module__') and 
                (func.__module__.startswith('numpy') or func.__module__ == 'builtins')):
                units = UnitPropagator._get_units_from_args(*args)
                return UnitPropagator._try_compute_unit(func, units, **kwargs)

            # 获取函数源码
            source = inspect.getsource(func)

            # 处理缩进问题
            lines = source.splitlines()
            if lines:
                # 找到第一行非空白字符的缩进级别
                first_line = lines[0]
                indent = len(first_line) - len(first_line.lstrip())
                # 去除所有行的相同缩进
                dedented_lines = [line[indent:] if line.startswith(' ' * indent) else line for line in lines]
                source = '\n'.join(dedented_lines)
            tree = ast.parse(source)

            if isinstance(tree.body[0], ast.FunctionDef):
                func_body = tree.body[0].body

                # 创建本地变量字典，用于存储变量和其单位
                local_dict = {}
                unit_dict = {}  # 新增：专门存储单位的字典

                # 绑定参数
                arg_names = [arg.arg for arg in tree.body[0].args.args]
                for name, value in zip(arg_names, args):
                    local_dict[name] = value
                    unit_dict[name] = UnitPropagator._get_unit_from_arg(value)

                local_dict['np'] = np
                local_dict['u'] = u

                # 收集并处理所有节点
                visitor = UnitPropagator.NodeVisitor()
                visitor.visit(tree)

                for node_type, node in visitor.nodes:
                    if node_type == 'assign':
                        # 处理赋值语句
                        target_name = node.targets[0].id
                        value_unit = UnitPropagator._eval_node(node.value, local_dict, unit_dict)  # 添加unit_dict参数
                        unit_dict[target_name] = value_unit  # 存储到单位字典中
                    elif node_type == 'return':
                        # 使用单位字典查找变量的单位
                        if isinstance(node.value, ast.Name):
                            var_name = node.value.id
                            if var_name in unit_dict:
                                return unit_dict[var_name]
                        return UnitPropagator._eval_node(node.value, local_dict, unit_dict)

        except Exception as e:
            print(f"错误: {str(e)}")
            raise
    
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

def is_quantity_collection(values):
    if isinstance(values, u.Quantity) and np.isscalar(values.value):
        return False
    result =  (isinstance(values, (list, np.ndarray)) and 
            len(values) > 1 and 
            hasattr(values[0], 'unit'))
    return result

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
    def _contains_quantity(arg) -> bool:
        """
        检查参数是否包含 Quantity

        Parameters
        ----------
        arg : Any
            要检查的参数

        Returns
        -------
        bool
            如果参数包含 Quantity 返回 True，否则返回 False
        """
        if isinstance(arg, u.Quantity):
            return True
        elif isinstance(arg, (list, tuple, np.ndarray)):
            return any(isinstance(x, u.Quantity) or 
                      (isinstance(x, (list, tuple, np.ndarray)) and 
                       QuantityWrapper._contains_quantity(x)) for x in arg)
        return False

    @staticmethod
    def wrap(func: Callable, *args, output_unit: Union[u.Unit, None] = None,
            simplify_units: bool = False, force_numpy: bool = False, **kwargs) -> Any:
        """包装numpy函数以支持quantity并自动追踪单位传播"""

        # 检查是否有quantity  
        has_quantity = any(QuantityWrapper._contains_quantity(arg) for arg in args)
        if not has_quantity and output_unit is None:
            return func(*args, **kwargs)
            
        # 尝试使用Quantity方法
        if not force_numpy:
            result, success = QuantityWrapper._try_quantity_method(
                func, args, output_unit, **kwargs)  # 传递kwargs
            if success:
                print('success')
                return simplify_unit(result) if simplify_units else result
            
        # 如果不能使用Quantity方法，使用原有的处理方式
        # 先尝试使用quantity_input
        if len(args) == 1:
            if is_quantity_collection(args[0]):
                if args[0][0].unit.physical_type == 'angle' or func in [np.abs, np.log10]:
                    result, success = QuantityWrapper._try_quantity_input(
                        func, *args, output_unit=output_unit, **kwargs)
                else:
                    success = False
            else:
                result, success = QuantityWrapper._try_quantity_input(
                    func, *args, output_unit=output_unit, **kwargs)
        elif all(not is_quantity_collection(arg) for arg in args):
            result, success = QuantityWrapper._try_quantity_input(
                func, *args, output_unit=output_unit, **kwargs)
        else:
            success = False

        if success:
            print('success')
            return simplify_unit(result) if simplify_units else result
        
        def process_arg(arg):
            if isinstance(arg, u.Quantity):
                return arg.value
            elif isinstance(arg, (list, tuple)):
                # 检查是否所有元素都是Quantity
                if all(isinstance(x, u.Quantity) for x in arg):
                    # 检查是否所有单位相同
                    units = [x.unit for x in arg]
                    if not all(u == units[0] for u in units):
                        raise ValueError("列表中的元素必须具有相同的单位")
                    # 提取值
                    return [x.value for x in arg]
                else:
                    # 如果不是所有元素都是Quantity，则原样返回
                    return arg
            else:
                return arg
        processed_args = tuple(process_arg(arg) for arg in args)

        # 获取传播单位 - 直接使用UnitPropagator.propagate
        propagated_unit = UnitPropagator.propagate(func, *args, **kwargs)
            
        # 计算结果
        result = func(*processed_args, **kwargs)
        
        # 应用单位
        print('Use my own method')
        print(result, propagated_unit)
        result = QuantityWrapper._apply_unit(result, output_unit, propagated_unit, simplify_units)
        return result

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

        #angles = [30*u.deg, 45*u.deg, 60*u.deg]
        #print(f"Sin of {angles}: {quantity_wrap(np.sin, angles)}")
    
    
    def test_array_operations():
        """测试数组操作"""
        print("\nTesting array operations:")
        print("-----------------------")
        
        arr1 = [1, 2, 3] * u.m
        arr2 = [4, 5, 6] * u.s
        print(f"Array multiplication: {quantity_wrap(np.multiply, arr1, arr2)}")
        
        # 矩阵运算
        #matrix = np.array([[1, 2], [3, 4]]) * u.m
        #vector = np.array([2, 1]) * u.s
        #print(f"Matrix-vector multiplication: {quantity_wrap(np.dot, matrix, vector)}")
    
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
        arr2 = [4, 5, 6] * u.dimensionless_unscaled  # 无单位
        print(f"{arr1} * {arr2} = {quantity_wrap(np.multiply, arr1, arr2)}") # TODO
    
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
        print(f"Wrapped sum:\n{wrapped_sum}") # TODO
        
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
        
        
    
    # 运行所有测试
    test_simplify_units()
    test_trigonometric()
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
    reshape_unit = UnitPropagator.propagate(np.reshape, img1, newshape=(2,2))
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
    """测试UnitPropagator对数组和复合函数的处理"""
    print("\n=== 测试数组操作 ===")
    
    # 创建测试数据
    A = np.array([[1, 2], [3, 4]]) * u.m
    B = np.array([[2, 0], [0, 2]]) * u.s
    
    # 1. 测试直接的数组操作
    print("\n直接数组操作:")
    sum_unit = UnitPropagator.propagate(np.sum, A)
    print(f"np.sum单位: {sum_unit}")  # 应该是 m
    
    mean_unit = UnitPropagator.propagate(np.mean, A)
    print(f"np.mean单位: {mean_unit}")  # 应该是 m
    
    # 2. 测试复合函数
    print("\n复合函数:")
    def composite_func1(x):
        """包含sum的复合函数"""
        return np.sum(x) / 2
    
    def composite_func2(x, y):
        """包含多个数组操作的复合函数"""
        #return np.sum(np.prod([x,y],axis=0)) / np.mean(y)
        return np.prod([x,y])
    
    comp1_unit = UnitPropagator.propagate(composite_func1, A)
    print(f"composite_func1单位: {comp1_unit}")  # 应该是 m
    
    comp2_unit = UnitPropagator.propagate(composite_func2, A, B)
    print(f"composite_func2单位: {comp2_unit}")  # 应该是 m
    
    # 3. 测试带axis参数的操作
    
    # 4. 测试更复杂的复合函数
    print("\n复杂复合函数:")
    def complex_func(x, y):
        """包含多个数组操作和中间计算的复合函数"""
        intermediate = np.sum(x * y, axis=0)  # 按列求和
        return np.mean(intermediate) / np.std(y)
    
    complex_unit = UnitPropagator.propagate(complex_func, A, B)
    print(f"complex_func单位: {complex_unit}")  # 应该是 m

def test_quantity_wrap():
    """测试quantity_wrap对复合函数的处理"""
    print("\n=== 测试复合函数 ===")
    
    # 创建测试数据
    A = np.array([[1, 2], [3, 4]]) * u.m
    B = np.array([[2, 0], [0, 2]]) * u.s
    
    def composite_func2(x, y):
        """包含多个数组操作的复合函数"""
        return np.sum(np.prod([x,y],axis=0)) / np.mean(y)
    
    # 1. 直接调用函数
    print("\n直接调用:")
    try:
        direct_result = composite_func2(A, B)
        print(f"结果: {direct_result}")
        print(f"单位: {getattr(direct_result, 'unit', None)}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 2. 使用quantity_wrap
    print("\n使用quantity_wrap:")
    wrapped_result = QuantityWrapper.wrap(composite_func2, A, B)
    print(f"结果: {wrapped_result}")
    print(f"单位: {getattr(wrapped_result, 'unit', None)}")
    
    # 3. 分步测试
    print("\n分步测试:")
    # 测试乘法
    mul_result = QuantityWrapper.wrap(np.multiply, A, B)
    print(f"x * y = {mul_result}")
    print(f"乘法单位: {getattr(mul_result, 'unit', None)}")
    
    # 测试求和
    sum_result = QuantityWrapper.wrap(np.sum, mul_result)
    print(f"sum(x * y) = {sum_result}")
    print(f"求和单位: {getattr(sum_result, 'unit', None)}")
    
    # 测试平均值
    mean_result = QuantityWrapper.wrap(np.mean, B)
    print(f"mean(y) = {mean_result}")
    print(f"平均值单位: {getattr(mean_result, 'unit', None)}")
    
    # 测试除法
    print(sum_result,mean_result)
    div_result = QuantityWrapper.wrap(np.divide, sum_result, mean_result)
    print(f"sum(x * y) / mean(y) = {div_result}")
    print(f"最终单位: {getattr(div_result, 'unit', None)}")

if __name__ == '__main__':
    test_image_operations()
    #test_unit_propagation()
    #test_quantity_wrap()
    #example_usage()
from typing import List, Union, Callable, Any, Sequence, Dict
import numpy as np
import astropy.units as u

def get_common_unit(*args):
    """获取多个参数的共同单位
    
    Parameters
    ----------
    args : sequence
        
    Returns
    -------
    Unit or None
        共同单位，如果没有单位则返回None
    """
    # 找到第一个带单位的参数
    base_unit = next((arg.unit for arg in args 
                     if isinstance(arg, u.Quantity)), None)
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
    base_unit = target_unit if target_unit else get_common_unit(*args)
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

def list_to_array_quantity(quantities: List) -> Union[np.ndarray, u.Quantity]:
    """Convert a list to an array
    
    Parameters
    ----------
    quantities : List
        List of numbers or Quantities
        
    Returns
    -------
    Union[np.ndarray, Quantity] 
        numpy array for regular numbers, or Quantity array for Quantities
    
    Raises
    ------
    ValueError
        If units are inconsistent when Quantities are provided
    """
    # Check if input contains any Quantity objects
    has_quantities = any(isinstance(q, u.Quantity) for q in quantities)
    
    if not has_quantities:
        # If no Quantities, return regular numpy array
        return np.array(quantities)
        
    # If there are Quantities, check consistency
    if not all(isinstance(q, u.Quantity) for q in quantities):
        print(list)
        raise ValueError("If any element is a Quantity, all elements must be Quantity objects")
        
    if not all(q.unit.physical_type == quantities[0].unit.physical_type for q in quantities):
        raise ValueError("All units must have the same physical type")
        
    return u.Quantity(quantities)

def process_sequence(sequence):
    """处理序列类型的输入
    
    Parameters
    ----------
    sequence : list, tuple, or array
        输入序列
        
    Returns
    -------
    tuple
        (处理后的序列, 单位)
    """
    if not isinstance(sequence, (list, tuple, np.ndarray)):
        return sequence
    
    if isinstance(sequence, u.Quantity):
        return sequence
    
    # 获取序列的共同单位
    unit = get_common_unit(*sequence)
    
    if unit is not None:
        # 如果有单位，去掉所有值的单位
        processed_seq = [
            arg.to(unit).value if isinstance(arg, u.Quantity) else arg 
            for arg in sequence
        ]
        return processed_seq*unit
    
    return sequence
    
def quantity_wrap(func: Callable, 
                 *args,
                 output_unit: Union[u.Unit, None] = None,
                 only_use_number: bool = False,
                 **kwargs) -> Any:
    """
    包装numpy函数以支持quantity
    
    Parameters
    ----------
    func : Callable
        要调用的numpy函数
    args : sequence
        位置参数
    kwargs : dict
        关键字参数
    output_unit : Unit or None
        期望的输出单位。如果为None，会尝试自动推断
    only_use_number : bool
        是否仅用输入的数字进行计算
    
    Returns
    -------
    result : array or Quantity
        计算结果，如果指定了output_unit则带单位
    """
    args = list(args)
    for i in range(len(args)):
        if isinstance(args[i], (list, tuple, np.ndarray)):
            args[i] = process_sequence(args[i])
    args = tuple(args)

    for key, value in kwargs.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            kwargs[key] = process_sequence(value)

    # 检查所有参数中是否有quantity
    all_args = list(args) + list(kwargs.values())
    has_quantity = any(isinstance(arg, u.Quantity) for arg in all_args)
    
    if not has_quantity and output_unit is None:
        return func(*args, **kwargs)
    
    # 获取第一个quantity的单位作为基准单位
    base_unit = None
    for arg in all_args:
        if isinstance(arg, u.Quantity):
            base_unit = arg.unit
            break
    
    # 处理位置参数
    processed_args = []
    for arg in args:
        if isinstance(arg, u.Quantity):
            if not only_use_number:
                processed_args.append(arg.to(base_unit).value)
            else:
                processed_args.append(arg.value)
        else:
            processed_args.append(arg)
    
    # 处理关键字参数
    processed_kwargs = {}
    for key, arg in kwargs.items():
        if isinstance(arg, u.Quantity):
            if not only_use_number:
                processed_kwargs[key] = arg.to(base_unit).value
            else:
                processed_kwargs[key] = arg.value
        else:
            processed_kwargs[key] = arg
    
    # 调用函数
    result = func(*processed_args, **processed_kwargs)
    
    # 处理输出单位
    if output_unit is not None and only_use_number:
        return result * output_unit
    elif output_unit is None and only_use_number:
        return result
    elif output_unit is not None and base_unit is not None:
        try:
            return (result * base_unit).to(output_unit)
        except:
            return result * output_unit
    elif base_unit is not None:
        return result * base_unit
    elif output_unit is not None:
        return result * output_unit
    else:
        return result

# Usage example:
if __name__ == "__main__":
    a = 100*u.mJy
    b = 2*u.Jy
    result = quantity_wrap(np.prod, [a,b])
    print(result)
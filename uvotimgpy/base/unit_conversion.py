from typing import List, Union, Callable, Any, Sequence, Dict
import numpy as np
import astropy.units as u

class QuantityConverter:
    """Utility class for converting astropy.units.Quantity objects"""
    
    @staticmethod
    def list_to_array(quantities: List) -> Union[np.ndarray, u.Quantity]:
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
            
        if not all(q.unit == quantities[0].unit for q in quantities):
            raise ValueError("All quantities must have the same unit")
            
        return u.Quantity(quantities)


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
    # Create some test data
    data = [
        1.0 * u.adu,
        2.0 * u.adu,
        3.0 * u.adu
    ]
    
    # Convert to array
    converter = QuantityConverter()
    array = converter.list_to_array(data)
    print(f"Converted array: {array}")
    print(f"Array unit: {array.unit}")

    print(QuantityConverter.list_to_array([]))

result = quantity_wrap(
    np.arange,
    1, 2,
    step=0.1,
    output_unit=u.meter,
    only_use_number=False
)
print(result)
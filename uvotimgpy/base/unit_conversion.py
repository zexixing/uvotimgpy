from typing import List, Union
import numpy as np
import astropy.units as u

class QuantityConverter:
    """Utility class for converting arrays of astropy.units.Quantity objects"""
    
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
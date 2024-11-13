from typing import List
import numpy as np
from astropy import units

class QuantityConverter:
    """Utility class for converting arrays of astropy.units.Quantity objects"""
    
    @staticmethod
    def list_to_array(quantities: List[units.Quantity]) -> units.Quantity:
        """Convert a list of Quantities to a Quantity array
        
        Parameters
        ----------
        quantities : List[Quantity]
            List of Quantities with the same unit
            
        Returns
        -------
        Quantity
            The converted array
        
        Raises
        ------
        ValueError
            If the list is empty or units are inconsistent
        """
        if not quantities:
            raise ValueError("Empty list provided")
            
        # Check if all elements are Quantities with consistent units
        if not all(isinstance(q, units.Quantity) for q in quantities):
            raise ValueError("All elements must be Quantity objects")
            
        if not all(q.unit == quantities[0].unit for q in quantities):
            raise ValueError("All quantities must have the same unit")
            
        return units.Quantity(quantities)

# Usage example:
if __name__ == "__main__":
    # Create some test data
    data = [
        1.0 * units.adu,
        2.0 * units.adu,
        3.0 * units.adu
    ]
    
    # Convert to array
    converter = QuantityConverter()
    array = converter.list_to_array(data)
    print(f"Converted array: {array}")
    print(f"Array unit: {array.unit}")
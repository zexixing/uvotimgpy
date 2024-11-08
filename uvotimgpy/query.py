from astropy.coordinates import SkyCoord
import astropy.units as u
from difflib import get_close_matches
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # Suppress astropy warnings

class StarCoordinateQuery:
    """
    A class to query star coordinates from local database and online services
    """
    
    def __init__(self):
        """
        Initialize with a dictionary of known stars
        Coordinates stored in degrees, FK5 J2000 system
        """
        self.known_stars = {
            'HD 12345': {'ra': 123.456*u.deg, 'dec': 45.678*u.deg},
            'SAO 12345': {'ra': 234.567*u.deg, 'dec': 56.789*u.deg},
            'TYC 1234-5678-1': {'ra': 345.678*u.deg, 'dec': 67.890*u.deg},
            'P177D': {'ra':239.80666667*u.deg, 'dec':47.61163889*u.deg}, # 15 59 13.579 +47 36 41.91 13.49mag 
            'P041C': {'ra':222.99166667*u.deg, 'dec':71.7215*u.deg}, # 14 51 57.980 +71 43 17.39 12.16mag
            'P330E': {'ra':247.891042*u.deg, 'dec':30.146417*u.deg}, # 16 31 33.813 +30 08 46.40 12.92mag
        }
        self.name_lookup = {name.lower(): name for name in self.known_stars}
    
    def _fuzzy_match(self, target_name, min_score=0.6):
        """
        Perform fuzzy matching on target name against known star names
        
        Parameters:
        -----------
        target_name : str
            Name of the target to search for
        min_score : float
            Minimum similarity score (0-1) for matching
            
        Returns:
        --------
        str or None
            Matched star name if found, None otherwise
        """
        matches = get_close_matches(
            target_name.lower(), 
            self.name_lookup.keys(), 
            n=1, 
            cutoff=min_score
        )
        return self.name_lookup[matches[0]] if matches else None
    
    def _query_online(self, target_name):
        """
        Query star coordinates from online astronomical database
        
        Parameters:
        -----------
        target_name : str
            Name of the target to search online
            
        Returns:
        --------
        SkyCoord or None
            SkyCoord object in FK5 if found, None if query fails
        """
        try:
            coords = SkyCoord.from_name(target_name, frame='fk5')
            return coords
        except:
            return None
    
    def get_coordinates(self, target_name):
        """
        Get coordinates through local database or online query
        
        Parameters:
        -----------
        target_name : str
            Name of the target star
        output_format : str
            'deg' for degree values with units
            'hmsdms' for sexagesimal format (HH:MM:SS.SS +DD:MM:SS.SS)
            
        Returns:
        --------
        tuple or str or None
            If output_format='deg': (ra, dec) coordinates with units
            If output_format='hmsdms': string in sexagesimal format
            None if target not found
        """
        coords = None
        
        # 1. Try exact match in local database
        if target_name in self.known_stars:
            star_info = self.known_stars[target_name]
            coords = SkyCoord(ra=star_info['ra'], 
                            dec=star_info['dec'], 
                            frame='fk5',
                            equinox='J2000')
            
        # 2. Try fuzzy match in local database
        if coords is None:
            matched_name = self._fuzzy_match(target_name)
            if matched_name is not None:
                star_info = self.known_stars[matched_name]
                print(f"Found fuzzy match: {target_name} -> {matched_name}")
                coords = SkyCoord(ra=star_info['ra'], 
                                dec=star_info['dec'],
                                frame='fk5',
                                equinox='J2000')
        
        # 3. Try online query as last resort
        if coords is None:
            coords = self._query_online(target_name)
            if coords is not None:
                print(f"Found coordinates from online database for {target_name}")
        
        if coords is None:
            print(f"No coordinates found for {target_name}")
            return None
            
        # Return coordinates
        return coords 
        #if output_format == 'deg':
        #    return coords.ra, coords.dec
        #elif output_format == 'hmsdms':
        #    return f"{coords.ra.to_string(unit=u.hour, sep=':')} {coords.dec.to_string(unit=u.deg, sep=':')}"
        #else:
        #    raise ValueError("output_format must be 'deg' or 'hmsdms'")


def test_star_coordinate_query():
    """Test function demonstrating the usage of StarCoordinateQuery"""
    query = StarCoordinateQuery()
    
    # Test with various inputs and formats
    print("Testing HD 12345:", query.get_coordinates('HD 12345'))
    print("Testing HD12345:", query.get_coordinates('HD12345'))
    print("Testing Vega:", query.get_coordinates('Vega'))
    print("Testing nonexistent star:", query.get_coordinates('NonexistentStar123'))

if __name__ == "__main__":
    test_star_coordinate_query()
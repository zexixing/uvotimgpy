from astropy.coordinates import SkyCoord
import astropy.units as u
from difflib import get_close_matches
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # Suppress astropy warnings

class StarCoordinateQuery:
    """
    A class to query star coordinates from local database and online services

    Example:
    # Test function demonstrating the usage of StarCoordinateQuery
    query = StarCoordinateQuery()

    # Test with various inputs and formats
    print("Testing HD 12345:", query.get_coordinates('HD 12345'))
    print("Testing HD12345:", query.get_coordinates('HD12345'))
    print("Testing Vega:", query.get_coordinates('Vega'))
    print("Testing nonexistent star:", query.get_coordinates('NonexistentStar123'))
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

class StarCatalogQuery:
    """处理不同星表查询的类"""
    
    def __init__(self, center_sky, radius, mag_limit, verbose=True):
        """
        Parameters
        ----------
        center_sky : SkyCoord
            搜索中心的天球坐标
        radius : Quantity
            搜索半径
        mag_limit : float
            星等上限
        verbose : bool, optional
            是否显示查询状态提醒，默认为True
        """
        self.center_sky = center_sky
        self.radius = radius
        self.mag_limit = mag_limit
        self.verbose = verbose
        
        # 预计算坐标范围
        radius_deg = self.radius.to(u.deg).value
        ra = self.center_sky.ra.deg
        dec = self.center_sky.dec.deg
        
        self.ra_min = ra - radius_deg
        self.ra_max = ra + radius_deg
        self.dec_min = dec - radius_deg
        self.dec_max = dec + radius_deg
        
    def _get_vizier_with_constraints(self, mag_column):
        """创建带有标准约束的Vizier对象"""
        vizier = Vizier(
            column_filters={
                "RAJ2000": f">{self.ra_min} & <{self.ra_max}",
                "DEJ2000": f">{self.dec_min} & <{self.dec_max}",
                mag_column: f"<{self.mag_limit}"
            }
        )
        vizier.ROW_LIMIT = -1
        return vizier
    
    def query_gsc(self):
        """查询GSC 2.3星表"""
        if self.verbose:
            print("Querying GSC 2.3 catalog...")
        vizier = self._get_vizier_with_constraints("Vmag")
        catalogs = vizier.query_constraints(catalog="I/305/out")
        if self.verbose:
            print("GSC 2.3 query completed")
        
        if not catalogs:
            raise ValueError("No objects found in GSC 2.3 for the specified region")
        
        return catalogs[0], 'RAJ2000', 'DEJ2000'
    
    def query_gaia(self):
        """查询GAIA DR3星表"""
        if self.verbose:
            print("Querying Gaia DR3 catalog...")
        vizier = self._get_vizier_with_constraints("Gmag")
        catalogs = vizier.query_constraints(catalog="I/355/gaiadr3")
        if self.verbose:
            print("Gaia DR3 query completed")
        
        if not catalogs:
            raise ValueError("No objects found in Gaia DR3 for the specified region")
            
        stars = catalogs[0]
        stars.rename_column('RA_ICRS', 'ra')
        stars.rename_column('DE_ICRS', 'dec')
        stars.rename_column('Gmag', 'phot_g_mean_mag')
        
        return stars, 'ra', 'dec'
    
    def query_ucac4(self):
        """查询UCAC4星表"""
        if self.verbose:
            print("Querying UCAC4 catalog...")
        vizier = self._get_vizier_with_constraints("Vmag")
        catalogs = vizier.query_constraints(catalog='I/322A/out')
        if self.verbose:
            print("UCAC4 query completed")
        
        if not catalogs:
            raise ValueError("No objects found in UCAC4 for the specified region")
            
        return catalogs[0], 'RAJ2000', 'DEJ2000'
    
    def query_apass(self):
        """查询APASS DR9星表"""
        if self.verbose:
            print("Querying APASS DR9 catalog...")
        vizier = self._get_vizier_with_constraints("Vmag")
        catalogs = vizier.query_constraints(catalog='II/336/apass9')
        if self.verbose:
            print("APASS DR9 query completed")
        
        if not catalogs:
            raise ValueError("No objects found in APASS DR9 for the specified region")
            
        return catalogs[0], 'RAJ2000', 'DEJ2000'
    
    def query_usnob(self):
        """查询USNO-B1.0星表"""
        if self.verbose:
            print("Querying USNO-B1.0 catalog...")
        vizier = self._get_vizier_with_constraints("R1mag")
        catalogs = vizier.query_constraints(catalog='I/284/out')
        if self.verbose:
            print("USNO-B1.0 query completed")
        
        if not catalogs:
            raise ValueError("No objects found in USNO-B1.0 for the specified region")
            
        return catalogs[0], 'RAJ2000', 'DEJ2000'

    def query_simbad(self):
        """查询SIMBAD数据库"""
        if self.verbose:
            print("Querying SIMBAD database...")
        Simbad.add_votable_fields('flux(V)', 'ra(d)', 'dec(d)')
        result = Simbad.query_region(self.center_sky, radius=self.radius)
        if self.verbose:
            print("SIMBAD query completed")
            
        mask = result['FLUX_V'] < self.mag_limit
        stars = result[mask]
        if not len(stars):
            raise ValueError("No objects found in SIMBAD for the specified region")
        stars.rename_column('RA_d', 'ra')
        stars.rename_column('DEC_d', 'dec')
        stars.rename_column('FLUX_V', 'vmag')
        
        return stars, 'ra', 'dec'
    
    def query(self, catalog):
        """根据指定的星表名称进行查询"""
        catalog = catalog.upper()
        query_methods = {
            'GSC': self.query_gsc,
            'GAIA': self.query_gaia,
            'UCAC4': self.query_ucac4,
            'APASS': self.query_apass,
            'USNOB': self.query_usnob,
            'SIMBAD': self.query_simbad,
        }
        
        if catalog not in query_methods:
            raise ValueError(f"Unsupported catalog: {catalog}")
        
        return query_methods[catalog]()

## 测试更小的区域
#center = SkyCoord(ra=180, dec=0, unit='deg')
#radius = 0.1 * u.deg  # 缩小搜索半径到0.1度
#query = StarCatalogQuery(center, radius, mag_limit=15)
#
## 获取查询结果
#stars, ra_key, dec_key = query.query('apass')
## 使用结果
#coords = SkyCoord(ra=stars[ra_key], dec=stars[dec_key])
#print(coords)



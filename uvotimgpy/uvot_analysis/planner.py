import numpy as np
from astropy.time import Time
from uvotimgpy.config import paths
from uvotimgpy.base.math_tools import UnitConverter
from astropy.coordinates import SkyCoord, ICRS, FK5

class RegionPlanner:
    def __init__(self, option_path):
        """
        Initialize RegionPlanner with an option file name
        
        Parameters:
        -----------
        option_name : str
            Name of the option file in the docs directory
        """
        self.option_name = option_path
        self.filter_configs = {
            'uvw1': {'c': 'magenta'},
            'v': {'c': 'green'},
            'b': {'c': 'cyan'},
            'u': {'c': 'blue'},
            # fotd, break
            'v_clocked':{'c':'red','axis':140.5},
            'uv_clocked':{'c':'red','axis':144.5},
            'v_nominal':{'c':'red','axis':148.1},
            'uv_nominal':{'c':'red','axis':151.4},
        }
        self._load_option()
    
    def _load_option(self):
        """Load and parse the option file"""
        ra_list, dec_list = [], []
        start_list, end_list = [], []
        roll_list = []
        
        with open(self.option_name) as f:
            for line in f.readlines():
                info = line.split()
                ra = float(info[8])
                dec = float(info[9])
                start = Time(info[0].replace('-',':')+':'+info[1], format='yday').jd
                end = Time(info[3].replace('-',':')+':'+info[4], format='yday').jd
                roll = (float(info[16])+float(info[17]))/2
                
                ra_list.append(ra)
                dec_list.append(dec)
                start_list.append(start)
                end_list.append(end)
                roll_list.append(roll)

        self.start_list = np.array(start_list)
        self.end_list = np.array(end_list)
        self.ra_list = np.array(ra_list)
        self.dec_list = np.array(dec_list)
        self.roll_list = np.array(roll_list)

        # Calculate rates
        self.mid_list = (self.start_list + self.end_list) / 2
        self.ra_rate = (self.ra_list[1] - self.ra_list[0]) / ((self.mid_list[1] - self.mid_list[0]) * 24 * 3600)
        self.dec_rate = (self.dec_list[1] - self.dec_list[0]) / ((self.mid_list[1] - self.mid_list[0]) * 24 * 3600)
        
        self.ra_init = self.ra_list - self.ra_rate * ((self.mid_list - self.start_list) * 24 * 3600)
        self.dec_init = self.dec_list - self.dec_rate * ((self.mid_list - self.start_list) * 24 * 3600)

    def create_reg(self, delta, obs_config, reg_path, grism=False):
        """
        Create region file based on the loaded options
        
        Parameters:
        -----------
        delta : float
            Delta value from Horizons
        order : list
            List of filter orders for observation
        reg_name : str
            Output region file name
        """
        #aper = 20  # aperture in arcsec
        aper = (100000/UnitConverter.au_to_km(delta))/np.pi*180*60*60
        
        with open(reg_path, 'w') as f:
            # Write header
            f.write('global color=white dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
            f.write('ICRS\n')
            
            # Process each observation
            for i in range(len(self.start_list)):
                base = 0
                ra_init_obs = self.ra_init[i]
                dec_init_obs = self.dec_init[i]
                roll_obs = self.roll_list[i]
                if obs_config == 'middle':
                    self._write_mid_observation(f, i, aper, grism, roll_obs)
                else:
                    self._write_multi_observations(f, obs_config, ra_init_obs, dec_init_obs, roll_obs, base, aper)


    def _write_mid_observation(self, f, i, aper, grism, roll_obs):
        """Write mid-point observation to region file"""
        c = 'cyan'
        ra_obs = self.ra_list[i]
        dec_obs = self.dec_list[i]
        f.write(f'circle({ra_obs},{dec_obs},{aper}") # color={c}\n')
        if grism:
            axis = self.filter_configs[grism]['axis']
            disp = roll_obs - 240.64 + axis
            antidisp = disp + 180
            f.write(f'# vector({ra_obs},{dec_obs},504",{disp}) vector=1\n')
            f.write(f'# vector({ra_obs},{dec_obs},300",{antidisp}) vector=1\n')

    def _write_multi_observations(self, f, obs_config, ra_init_obs, dec_init_obs, roll_obs, base, aper):
        """Write multiple observations for each orbit"""
        for filter_name in obs_config.keys():
            if filter_name in ['break', 'fotd']:
                continue
                
            t = obs_config[filter_name]
            t_obs = base * 2 + t/2
            base += t/2
            
            ra_obs = ra_init_obs + self.ra_rate * t_obs
            dec_obs = dec_init_obs + self.dec_rate * t_obs
            color = self.filter_configs[filter_name]['c']
            
            f.write(f'circle({ra_obs},{dec_obs},{aper}") # color={color}\n')

            if filter_name in ['v_clocked', 'uv_clocked', 'v_nominal', 'uv_nominal']:
                axis = self.filter_configs[filter_name]['axis']
                disp = roll_obs - 240.64 + axis
                antidisp = disp + 180
                f.write(f'# vector({ra_obs},{dec_obs},504",{disp}) vector=1\n')
                f.write(f'# vector({ra_obs},{dec_obs},300",{antidisp}) vector=1\n')

# Example usage:
if __name__ == "__main__":
    
    # Update filter times if needed
    #obs_config = {'uvw1': 200, 'v': 200, 'b': 200, 'u': 200, 'fotd': 300, 'break': 20, 
    #              'v_clocked':1000, 'uv_clocked':1000, 'v_nominal':1000, 'uv_nominal':1000}
    #obs_config = {'fotd': 300, 'break': 20, 
    #              'v': 200, 'break': 20, 'v': 200, 'break': 20,
    #              'uvw1': 200, 'break': 20, 'uvw1': 200, 'break': 20, 'uvw1': 200,
    #              'break': 20, 'uvw1': 200, 'break': 20, 'uvw1': 200}   
    obs_config = 'middle'

    # path
    obs_path = paths.get_subpath(paths.projects, 'C_2025N1', 'obs')
    option_path = paths.get_subpath(obs_path, '2025sep.txt')
    reg_path = paths.get_subpath(obs_path, '2025sep.reg')
    print(option_path)

    # Create region file
    delta = 2.57
    planner = RegionPlanner(option_path)
    planner.create_reg(delta, obs_config, reg_path)
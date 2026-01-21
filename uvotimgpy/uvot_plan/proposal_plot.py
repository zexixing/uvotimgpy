import re
from datetime import datetime
from uvotimgpy.base.file_and_table import dates_to_plot_dates, read_horizons_table
from uvotimgpy.uvot_plan.etc import create_magnitude_calculator
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np  
from sbpy.data import Ephem
from astropy.table import Table
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.axes as AA
from matplotlib.dates import DateFormatter
from mpl_toolkits.axisartist import SubplotHost
from matplotlib.ticker import FixedLocator, FixedFormatter

def orbit_dict_to_mag(orbit_dict, mag_calculator=None):
    if mag_calculator is None:
        mag_list = orbit_dict['Tmag']
    else:
        mag_list = mag_calculator(orbit_dict['r'], orbit_dict['delta'], orbit_dict['date'])
    return mag_list

def get_yearend_list(start, end, month=1, day=1):
    """
    获取日期范围内的所有1月1日
    支持多种日期格式
    """
    # 尝试不同的格式
    formats = [
        '%Y-%m-%dT%H:%M:%S.%f',  # 2023-04-01T00:00:00.000
        '%Y-%m-%dT%H:%M:%S',      # 2023-04-01T00:00:00
        '%Y-%m-%d %H:%M:%S',      # 2023-04-01 00:00:00
        '%Y-%m-%d',               # 2023-04-01
    ]
    
    # 解析开始日期
    for fmt in formats:
        try:
            s = datetime.strptime(start, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"无法解析日期: {start}")
    
    # 解析结束日期
    for fmt in formats:
        try:
            e = datetime.strptime(end, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"无法解析日期: {end}")
    
    # 生成特定月日的列表
    result = []
    for y in range(s.year, e.year + 1):
        try:
            target_date = datetime(y, month, day)
            # 检查是否在范围内
            if s <= target_date <= e:
                result.append(f'{y:04d}-{month:02d}-{day:02d}')
        except ValueError:
            # 处理无效日期（如2月30日）
            pass
    
    return result

def find_elong_bounds(date, elong, threshold=50):
    """
    找到elong<=threshold的连续段，返回每段外侧边界（刚好>threshold的点）
    返回[(段前的点, 段后的点), ...]
    threshold: float, int, or tuple
    """
    segments = []
    in_segment = False
    start = None
    
    for i in range(len(elong)):
        if isinstance(threshold, float) or isinstance(threshold, int):
            in_threshold = elong[i] <= threshold
        elif isinstance(threshold, tuple):
            in_threshold = ((elong[i] <= threshold[0]) or (elong[i] >= threshold[1]))
        if in_threshold:
            if not in_segment:  # 段开始
                # 记录段前的点（上一个>threshold的点）
                if i == 0:
                    start = date[0]  # 第一个点就在段内，用第一个点
                else:
                    start = date[i-1]  # 用前一个点（>threshold）
                in_segment = True
        else:  # elong[i] > threshold
            if in_segment:  # 段结束
                # 记录段后的点（当前这个>threshold的点）
                segments.append((start, date[i]))
                in_segment = False
    
    # 如果最后还在段内
    if in_segment:
        segments.append((start, date[-1]))  # 用最后一个点
    
    # 过滤掉开始和结束相同的tuple
    segments = [(s, e) for s, e in segments if s != e]
    
    return segments

def plot_lightcurve(orbit_dict, mag_list, current_cycle, cycle_dict, obs_dict, perihelion_date=None,
                    yearend_list=None, elong_list=None, xlim=None, ylim=None,
                    text_params=None, elong_threshold=46, save_path=None):
    """
    current_cycle = '21'

    cycle_dict = {
        '19': ['2023-04-01', '2024-04-01'],
        '20': ['2024-04-01', '2025-04-01'],
        '21': ['2025-04-01', '2026-04-01'],
    }
    obs_dict = {
        '19': ['2024-01-22', '2024-03-04'],
        '20': ['2024-04-13','2024-05-20','2024-06-23','2024-07-14','2024-11-20','2025-03-14'],
        '21': ['2025-04-23','2025-06-04', '2025-09-03'],
    }
    yearend_list = ['2023-04-01','2024-01-01','2025-01-01','2026-01-01']
    elong_list = [('2023-09-03', '2023-12-26'), ('2024-07-18', '2024-10-24'), ('2024-11-24', '2025-03-13'), ('2025-11-17', '2026-02-12')]
    xlim = ['2023-04-01', '2026-04-01']
    text_params = {'name': 'C/2023 A3', 'position': (0.9, 0.9)}
    elong_threshold: float, int, or tuple
    """
    # read data
    date = orbit_dict['date']
    date_plot = dates_to_plot_dates(date)
    
    # calculate magnitudes, create interpolation functions 
    if mag_list is None:
        mag_list = orbit_dict['Tmag']
    mag_f = interp1d(date_plot, mag_list, fill_value='extrapolate')
    time2rh_f = interp1d(date_plot, orbit_dict['r'], fill_value='extrapolate')

    yearend_list = get_yearend_list(date[0], date[-1], month=1, day=1)
    yearend_list = dates_to_plot_dates(yearend_list)

    yearmiddle_list = get_yearend_list(date[0], date[-1], month=6, day=30)
    yearname_list = [i.split('-')[0] for i in yearmiddle_list]
    yearmiddle_list = dates_to_plot_dates(yearmiddle_list)

    if perihelion_date is not None:
        perihelion_date = dates_to_plot_dates(perihelion_date)
        perihelion_rh = time2rh_f(perihelion_date)

    for key in cycle_dict:
        cycle_dict[key] = [dates_to_plot_dates(cycle_dict[key][0]), dates_to_plot_dates(cycle_dict[key][1])]
    
    if obs_dict is not None:
        for key in obs_dict:
            obs_dict[key] = [dates_to_plot_dates(obs_date) for obs_date in obs_dict[key]]
    
    if elong_list is not None:
        elong_list_plot = []
        for elong_tuple in elong_list:
            elong_list_plot.append((dates_to_plot_dates(elong_tuple[0]), dates_to_plot_dates(elong_tuple[1])))
        elong_list = elong_list_plot
    else:
        elong = orbit_dict.get('S-O-T', orbit_dict.get('elong'))
        elong_list = find_elong_bounds(date, elong, threshold=elong_threshold)
        print(elong_list)
        elong_list = [(dates_to_plot_dates(elong_tuple[0]), dates_to_plot_dates(elong_tuple[1])) for elong_tuple in elong_list]

    if xlim is not None:
        xlim = (dates_to_plot_dates(xlim[0]), dates_to_plot_dates(xlim[1]))
    else:
        xlim = (date_plot[0], date_plot[-1])

    # plot setting
    fig = plt.figure(figsize=[6,4], dpi=300)
    host = SubplotHost(fig, 111)
    fig.add_subplot(host)
    plt.subplots_adjust(right=0.75, bottom=0.15)
    #host = host_subplot(111, axes_class=AA.Axes)
    #plt.subplots_adjust(right=0.75)
    
    ax_years = host.twiny() # create axis for years
    ax_years2 = host.twiny()
    ax_rh = host.twiny() # create axis for rh
    
    offset = -20 # shift the year axis by an offset of -20
    new_fixed_axis = ax_years.get_grid_helper().new_fixed_axis
    new_fixed_axis2 = ax_years2.get_grid_helper().new_fixed_axis
    ax_years.axis["bottom"] = new_fixed_axis(loc="bottom",axes=ax_years,offset=(0, offset))
    ax_years.axis["bottom"].toggle(all=True)
    ax_years2.axis["bottom"] = new_fixed_axis2(loc="bottom",axes=ax_years2,offset=(0, offset))
    ax_years2.axis["bottom"].toggle(all=True)
    
    offset = 0
    new_fixed_axis = ax_rh.get_grid_helper().new_fixed_axis
    ax_rh.axis["top"] = new_fixed_axis(loc="top",axes=ax_rh,offset=(0, offset))
    ax_rh.axis["top"].toggle(all=True)
    
    host.set_ylabel('Apparent magnitude (V-band)')
    ax_years.set_xlabel('Date')
    ax_rh.set_xlabel('Heliocentric distance (AU)')
    
    ax_rh.tick_params(axis='y', colors='k') # change the color of the rh e3 axis
    #ax_rh.yaxis.label.set_color('red')
    ax_rh.axis["top"].line.set_color('k')
    ax_rh.axis["top"].major_ticks.set_color('k')
    ax_rh.axis["top"].major_ticklabels.set_color('k')
    
    date_form = DateFormatter("%m") # label with only month
    host.xaxis.set_major_formatter(date_form)
    host.xaxis.set_major_locator(mdates.MonthLocator(interval=4)) # set the frequency of date ticks
    host.set_xlim(xlim[0], xlim[1])
    
    ax_rh.set_xticks(host.get_xticks()) # assign the values of E3 rh to the axis ticks and labels
    ax_rh.set_xlim(host.get_xlim())
    ax_rh.set_xticklabels(np.around(time2rh_f(host.get_xticks()),1).astype(str))

    if perihelion_date is not None:
        ax_rh.set_xticks([perihelion_date], minor=True)
        ax_rh.set_xticklabels([np.around(perihelion_rh,1).astype(str)], minor=True)
        ax_rh.axis["top"].minor_ticklabels.set_color('gray')
        ax_rh.axvline(perihelion_date, color='black', lw=1, ls=':',alpha=0.3) #cycle separation line
    
    # 将年中位置设为主刻度（用于标签）
    ax_years.set_xticks(yearmiddle_list)  # 主刻度在年中
    ax_years.set_xlim(host.get_xlim())
    ax_years.set_xticklabels(yearname_list)  # 标签显示在年中
    ax_years.axis["bottom"].major_ticks.set_ticksize(0)  # 隐藏主刻度线
    ax_years.axis["top"].major_ticks.set_ticksize(0)  # 隐藏主刻度线

    ax_years2.set_xticks(yearend_list)  # 主刻度在年末
    ax_years2.set_xlim(host.get_xlim())
    ax_years2.set_xticklabels([])
    ax_years2.axis["top"].major_ticks.set_ticksize(0)  # 隐藏主刻度线

    # 将年末位置设为次刻度（用于显示刻度线）
    #ax_years.set_xticks(yearend_list, minor=True)  # 次刻度在年末
    #ax_years.tick_params(axis='x', which='minor', length=10, width=2, color='black')  # 显示次刻度线

    keys = list(cycle_dict.keys())
    for key in keys[:-1]:
        host.axvline(cycle_dict[key][1], color='k', lw=1, ls='--',alpha=0.3) #cycle separation line
    
    if ylim is not None:
        host.set_ylim(ylim[0], ylim[1]) # set limit for the magnitudes
    
    #plot
    date_plot = np.array(date_plot)
    for key in cycle_dict:
        if key == current_cycle:
            host.plot(date_plot[(date_plot>=cycle_dict[key][0])&(date_plot<cycle_dict[key][1])], mag_f(date_plot[(date_plot>=cycle_dict[key][0])&(date_plot<cycle_dict[key][1])]), 'k-', alpha=1)
        else:
            host.plot(date_plot[(date_plot>=cycle_dict[key][0])&(date_plot<cycle_dict[key][1])], mag_f(date_plot[(date_plot>=cycle_dict[key][0])&(date_plot<cycle_dict[key][1])]), 'k-', alpha=0.4)
    
    if elong_list is not None:
        for elong_tuple in elong_list:
            host.plot(date_plot[(date_plot>=elong_tuple[0])&(date_plot<elong_tuple[1])], mag_f(date_plot[(date_plot>=elong_tuple[0])&(date_plot<elong_tuple[1])]), 'w--',lw=2)

    if obs_dict is not None:
        for key in obs_dict:
            if key == current_cycle:
                if len(obs_dict[key]) > 0:
                    for obs_date in obs_dict[key]:
                        host.plot(obs_date, mag_f(obs_date), 'o',c='k',lw=0,markersize=5)
            else:
                if len(obs_dict[key]) > 0:
                    for obs_date in obs_dict[key]:
                        host.plot(obs_date, mag_f(obs_date), 'o', markeredgecolor='k', markerfacecolor='w', lw=0, markersize=5)
    
    # texts
    for key in cycle_dict:
        ylim = host.get_ylim()
        host.text((cycle_dict[key][0]+cycle_dict[key][1])/2, np.max(ylim) - (np.max(ylim)-np.min(ylim))/20, 'Cycle '+key, fontsize=10, ha='center',va='center')
    
    if text_params is not None:
        host.text(text_params['position'][0], text_params['position'][1], text_params['name'], fontsize=10,color='k', transform=host.transAxes,ha='center',va='center')
    
    # others
    jwst_date = '2025-07-28'
    jwst_date = dates_to_plot_dates(jwst_date)
    host.plot_date(jwst_date, mag_f(jwst_date), fmt='+',c='r',lw=0,markersize=8, label='JWST')
    arrow_style = {'arrowstyle':'-|>','color':'k'}
    host.annotate('JWST visit', xy=(jwst_date, mag_f(jwst_date)-0.1), xytext=(jwst_date,mag_f(jwst_date)-2), arrowprops=arrow_style, va='center',ha='center',size=10,color='k')

    host.invert_yaxis()
    plt.grid(alpha=0.3,ls='--')
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

#orbit_dict = read_horizons_table('/Users/zexixing/Downloads/horizons_results-a3.txt')
#mag_calculator = create_magnitude_calculator([(4.5, 10.0), (6.0, 7.5), (4.5, 10.0)],['2024-01-24', '2025-06-01'])
#mag_list = mag_calculator(orbit_dict['r'], orbit_dict['delta'], orbit_dict['date'])
#cycle_dict = {
#    '19': ['2023-04-01', '2024-04-01'],
#    '20': ['2024-04-01', '2025-04-01'],
#    '21': ['2025-04-01', '2026-04-01'],
#}
#obs_dict = {
#    '19': ['2024-01-22', '2024-03-04'],
#    '20': ['2024-04-13','2024-05-20','2024-06-23','2024-07-14','2024-11-20','2025-03-14'],
#    '21': ['2025-04-23','2025-06-04', '2025-09-03'],
#}
##elong_list = [('2023-09-03', '2023-12-26'), ('2024-07-18', '2024-10-24'), ('2024-11-24', '2025-03-13'), ('2025-11-17', '2026-02-12')]
#plot_lightcurve(orbit_dict, mag_list, current_cycle='21', cycle_dict=cycle_dict, obs_dict=obs_dict)

if __name__ == "__main__":
    # --- example ---
    #orbit_dict = read_horizons_table('/Users/zexixing/Downloads/horizons_results-e1-cycle22.txt')
    #mag_list = orbit_dict['T-mag']
    #mag_calculator = create_magnitude_calculator((7.0, 10.0))
    #mag_list = mag_calculator(orbit_dict['r'], orbit_dict['delta'], orbit_dict['date'])
    #cycle_dict = {
    #    '21': ['2025-04-01', '2026-04-01'],
    #    '22': ['2026-04-01', '2027-04-01'],
    #}
    #obs_dict = {
    #    '21': ['2025-06-17','2025-07-28', '2025-09-04', '2025-10-10', '2026-02-22', '2026-03-25'],
    #    '22': ['2026-04-28', '2026-10-09'], # 1.9, 4.0
    #    #'22': ['2027-01-04'], # 4.0
    #}
    #save_path = '/Users/zexixing/Downloads/cycle22_e1.png'
    #save_path = None
    #plot_lightcurve(orbit_dict, mag_list, current_cycle='22', cycle_dict=cycle_dict, obs_dict=obs_dict, 
    #    perihelion_date='2026-01-20', elong_threshold=46, 
    #    text_params={'name': 'C/2024 E1', 'position': (0.8, 0.9)}, save_path=save_path)

    orbit_dict = read_horizons_table('/Users/zexixing/Downloads/horizons_results-2p.txt')
    mag_list = orbit_dict['T-mag']
    cycle_dict = {
        '5': ['2026-07-01', '2027-06-30'],
    }
    obs_dict = {
        #'21': ['2025-06-17','2025-07-28', '2025-09-04', '2025-10-10', '2026-02-22', '2026-03-25'],
        #'22': ['2026-04-28', '2026-10-09'], # 1.9, 4.0
        #'22': ['2027-01-04'], # 4.0
    }
    save_path = '/Users/zexixing/Downloads/cycle5_2p.png'
    #save_path = None
    plot_lightcurve(orbit_dict, mag_list, current_cycle='5', cycle_dict=cycle_dict, obs_dict=obs_dict, 
        perihelion_date='2027-02-10', elong_threshold=(85, 135), 
        text_params={'name': '2P Encke', 'position': (0.8, 0.9)}, save_path=save_path, ylim=(12, 20))
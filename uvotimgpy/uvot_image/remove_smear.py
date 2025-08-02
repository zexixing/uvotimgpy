from _mypath import thisdir
import numpy as np
from tools import *
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.jplhorizons import Horizons
import math
import os
from aper_phot import limit_loop, get_dist
from before_stack import set_coord

def create_w(hd):
    # create w to find the comet (px, py)
    w = WCS(naxis=2)
    w.wcs.crpix= [hd['TCRPX6'],hd['TCRPX7']]
    w.wcs.cdelt= [hd['TCDLT6'],hd['TCDLT7']]
    w.wcs.crval= [hd['TCRVL6'],hd['TCRVL7']]
    w.wcs.ctype= [hd['TCTYP6'],hd['TCTYP7']]
    return w

def find_comet(w, time, id_num):
    obj = Horizons(id=id_num,location='@swift',epochs=time)
    eph = obj.ephemerides()[0]
    ra = eph["RA"]
    dec = eph["DEC"]
    px, py = w.wcs_world2pix( ra, dec, 1)
    return px, py

def slice_time(hd, num):
    dstart = Time(hd['DATE-OBS'])
    dend = Time(hd['DATE-END'])
    date = []
    tstart = hd['TSTART']
    tend = hd['TSTOP']
    time = []
    # initialize
    #date.append(dstart)
    #time.append(tstart)
    dd = dend - dstart
    dt = tend - tstart
    # loop
    for i in range(0,num+1):
        date.append(dstart+i/num*dd)
        time.append(tstart+i/num*dt)
    # finalize
    #date.append(dend)
    #time.append(tend)
    return date, time

def slice_pix(w, date, id_num):
    num = len(date) - 1
    x_comet = []
    y_comet = []
    for i in range(0, num):
        start = date[i]
        end = date[i+1]
        dt = end-start
        mid = start+1/2*dt
        px, py = find_comet(w, mid.jd, id_num)
        x_comet.append(px)
        y_comet.append(py)
    return x_comet, y_comet

def filt_time(time_list, x_list, y_list, time_array, i):
    # i in range(0, len(time_array)-1)
    start0 = time_array[i]
    end0 = time_array[i+1]
    judge = (time_list>=start0)&(time_list<end0)
    x_list = x_list[judge]
    y_list = y_list[judge]
    return x_list, y_list

def read_evt(evt_ind, filt):
    evt_path = get_path('../data/46P_raw_uvot/'+evt_ind+'/uvot/event/sw'+evt_ind+filt+'w1po_uf.evt.gz')
    hdul = fits.open(evt_path)
    dt = hdul[1].data
    hd = hdul[1].header
    time_list = dt['TIME']
    x_list = dt['X']
    y_list = dt['Y']
    return time_list, x_list, y_list, hd

def mask(slice_img, size, num, exp):#, exp_map):
    import itertools
    import pandas as pd
    import functools
    # mask bad exp
    #exp = np.median(exp_map.argsort()[::-1][0:5])
    #slice_img = slice_img*((exp_map==exp).astype(np.float64))
    # mask bright stars
    src_center = [size, size]
    src_r = 1400
    start = 100
    step = 10
    img_data, cen_pix, i_range, j_range = \
        limit_loop(slice_img, src_center, src_r)
    pos_pix_list = list(itertools.product(i_range,j_range))
    pos_pix_arr = np.array(pos_pix_list)
    #pos_pix_arr = np.empty(len(pos_pix_list), dtype=object)
    #pos_pix_arr[:] = pos_pix_list # list to array (keep tuple)
    #dist_list = np.array(list(map(functools.partial(get_dist, point_2=cen_pix), pos_pix_list))) #TODO:
    dist_list = np.sqrt(np.sum(np.power(pos_pix_arr-src_center,2),axis=1))
    index_list = ((np.array(dist_list)-start)/step).astype(int)
    image_list = img_data[i_range[0]:(i_range[-1]+1),j_range[0]:(j_range[-1]+1)].flatten()
    data = {'distance': dist_list,
            'position': pos_pix_list,
            'index': index_list,
            'image': image_list}
    data = pd.DataFrame(data)
    data = data[data['image']>0]
    data = data[data['distance']>start]
    data = data[data['distance']<src_r]
    data_group = data.groupby('index')
    median_list = {}
    for name,group in data_group:
        median_list[name] = np.median(group['image'])
    # mask
    mask_data = data[data['image']>(0.7*(exp/num))]
    if len(mask_data['position'])!=0:
        posi = np.array(list(mask_data['position']))
        mask_ind = mask_data['index']
        mask_fill = [median_list[key] for key in mask_ind]
        i = posi[:,0]
        j = posi[:,1]
        img_data[i, j] = mask_fill
    return img_data
    
def writeHDR(img_hdr):

    hdr = hdu.header
    hdr['TELESCOP'] = img_hdu.header['TELESCOP']
    hdr['INSTRUME'] = img_hdu.header['INSTRUME']
    hdr['FILTER'] = img_hdu.header['FILTER']
    hdr['COMET'] = obs_log_name.split('_')[0]+' '+obs_log_name.split('_')[-1][:-4]
    hdr['PLATESCL'] = ('1','arcsec/pixel')
    hdr['XPOS'] = f'{size[0]}'
    hdr['YPOS'] = f'{size[1]}'
    hdr['EXPTIME'] = (f'{exp}', '[seconds]')
    hdr['MID_TIME'] = f'{mid_t}'

def stack(evt_ind, filt, id_horizons, size):
    stacked_img = np.zeros((2*size -1,
                            2*size -1))
    stacked_exp = np.zeros((2*size -1,
                            2*size -1))
    time_list, x_list, y_list, hd = read_evt(evt_ind, filt)
    exp = hd['EXPOSURE']
    #num = math.ceil((34.37/196.6*exp)/4) ### calculate the num of slices
    num = math.ceil((11.56/139.5*exp)/4)
    print('num: '+str(num))
    w = create_w(hd)
    date, time = slice_time(hd, num)
    time_array = np.array(time)
    x_comet, y_comet = slice_pix(w, date, id_horizons) # 2 lists
    # for exp map
    exp_path = get_path(evt_ind,filt, to_file=True, map_type='ex')
    exp_hdu = fits.open(exp_path)[1]
    exp_data = exp_hdu.data.T
    exp_hd   = exp_hdu.header
    w_exp = WCS(exp_hd)
    x_comet_exp, y_comet_exp = slice_pix(w_exp, date, id_horizons)
    for i in range(0,num):
        print(i+1)
        x_slice_list, y_slice_list = filt_time(time_list, x_list, y_list, time_array, i)
        x_slice_list = x_slice_list - x_comet[i] + (size-1) + 0.5
        y_slice_list = y_slice_list - y_comet[i] + (size-1) + 0.5
        #pos_list = list(zip(x_slice_list,y_slice_list))
        edges = np.arange(0,2*size,1)
        slice_img = np.histogram2d(x_slice_list, y_slice_list, bins=(edges, edges))[0]
        if filt == 'uvv':
            slice_img = mask(slice_img, size, num, exp)
        stacked_img += slice_img
        # for exp map
        new_exp = set_coord(exp_data,
                            np.array([x_comet_exp[i]-1,y_comet_exp[i]-1]),
                            (size,size))
        stacked_exp += new_exp/num
    #stacked_exp = stacked_exp/np.max(stacked_exp)
    #stacked_exp = np.where((stacked_exp>0)*(stacked_exp<1),0.5,stacked_exp)
    output_name = evt_ind+'_'+filt+'.fits.gz'#'_'+str(int(num))+'.fits.gz'
    output_path = get_path('../docs/smear/'+output_name)
    hdr = fits.Header()
    dt = Time(hd['DATE-OBS']) - Time(hd['DATE-END'])
    mid_t = Time(hd['DATE-OBS']) + 1/2*dt
    hdr['TELESCOP'] = 'Swift'
    hdr['INSTRUME'] = 'UVOT'
    hdr['FILTER'] = filt
    hdr['COMET'] = evt_ind
    hdr['PLATESCL'] = ('0.502','arcsec/pixel')
    hdr['XPOS'] = f'{size}'
    hdr['YPOS'] = f'{size}'
    hdr['EXPTIME'] = (f'{exp}', '[seconds]')
    hdr['MID_TIME'] = f'{mid_t}'
    hdu_img = fits.PrimaryHDU(stacked_img,header=hdr)
    hdu_exp = fits.ImageHDU(stacked_exp)
    hdul = fits.HDUList([hdu_img,hdu_exp])
    hdul.writeto(output_path)
    os.system('mv '+output_path+' '+output_path[:-3])

# smooth
def gaussBlur(img, sigma, H, W, _boundary='fill', _fillvalue=0):
    import cv2 as cv
    from scipy import signal
    gaussKernel_x = cv.getGaussianKernel(W, sigma, cv.CV_64F)
    gaussKernel_x = np.transpose(gaussKernel_x)
    gaussBlur_x = signal.convolve2d(img, gaussKernel_x, mode="same",
                                    boundary=_boundary, fillvalue=_fillvalue)
    gaussKernel_y = cv.getGaussianKernel(H, sigma, cv.CV_64F)
    gaussBlur_xy = signal.convolve2d(gaussBlur_x, gaussKernel_y, mode="same",
                                     boundary=_boundary, fillvalue=_fillvalue)
    return gaussBlur_xy

'''
import math
#obs = {'00094421002':'uvv',
#       '00094422001':'uw1',}
       #'00094425002':'uvv',
       #'00094426001':'uw1',
       #'00094429002':'uvv',
       #'00094430001':'uw1'}

#obs = {'00094318002':'uvv'}
obs =  {
        '00094318002':'uvv',
        '00094318003':'uuu',
        '00094319001':'uw1',
        '00094319004':'uvv',
        '00094319005':'uuu',
        '00094320002':'uw1',
        '00094381002':'uw1',
        '00094381003':'uvv',
        '00094382001':'uuu',
        '00094382004':'uw1',
        '00094382005':'uvv',
        '00094383002':'uuu',
        '00094387002':'uw1',
        '00094387003':'uvv',
        '00094388001':'uuu',
        '00094388004':'uw1',
        '00094388005':'uvv',
        '00094389002':'uuu',
        '00094393002':'uw1',
        '00094393003':'uvv',
        '00094394001':'uuu',
        '00094394004':'uw1',
        '00094394005':'uvv',
        '00094395002':'uuu',
        '00094399002':'uw1',
        '00094399003':'uvv',
        '00094400001':'uuu',
        '00094400004':'uw1',
        '00094400005':'uvv',
        '00094401002':'uuu',
        '00094405002':'uw1',
        '00094405003':'uvv',
        '00094406001':'uuu',
        '00094406004':'uw1',
        '00094406005':'uvv',
        '00094407002':'uuu',
       }
for evt_ind in obs:
    print(evt_ind)
    stack(evt_ind, obs[evt_ind], 90000546, 2000)

hdul = fits.open('/Users/zexixing/Research/swift46P/docs/stack/14_uw1_evt.fits')
import matplotlib.pyplot as plt
img = hdul[0].data
plt.imshow(img,vmin=0,vmax=5)
plt.xlim(1000,3000)
plt.ylim(1000,3000)
import cv2 as cv
#img_blur = cv.GaussianBlur(img, (9, 9), 2,borderType=1)
img_blur = gaussBlur(img, 2, 9, 9, "symm")
fig=plt.figure()
plt.imshow(img_blur,vmin=0,vmax=5)
plt.xlim(1000,3000)
plt.ylim(1000,3000)
plt.show()
'''

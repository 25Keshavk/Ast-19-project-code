#!/usr/bin/env python
# coding: utf-8

# First download this code and put it under your ../ast19/ folder 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np 
import matplotlib.pyplot as plt

# this is just to make the font look prettier
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'

# we are going to deal with fits file, which is a type of data format to store 
# multi-dimensional data the following two lines are used to process 
# the fits file that we will examine
import astropy.io.fits as fits
from get_cubeinfo import get_cubeinfo
import os 
import math


# In[2]:


# current working directory 
cwd = os.getcwd()
print('Current working directory: ', cwd)


# # The first part is the same as code1 where we read in the datacube

# In[3]:


# here we combine all the codes above into a function so that we can call the function when needed 
def read_datacube_information(filename): 
    """
    Process input fitsfile to read flux and coordinate information 
    
    Input: fitsfile of a radio data cube, e.g. filename = './data/CAR_EF04_trim.fits'
    Output: 
    A library of 3D datacube, 2D coordinate (RA/DEC) information, 1D velocity, and a 2D header 
    
    History: 
    SIP2022, AST 19, 06/19/2022
    """
    from get_cubeinfo import get_cubeinfo
    import astropy.io.fits as fits
    
    cube_data = fits.getdata(filename)  # 3D data 
    cube_header = fits.getheader(filename) # 3D header 
    
    header_info = get_cubeinfo(cube_header, returnHeader=True)
    cube_ra2d = header_info[0] # 2D image of right ascension values for the data cube 
    cube_dec2d = header_info[1] # 2D image of declination values for the data cube 
    cube_vhel = header_info[2] # 1D array of the velocity of the data array 
    cube_header2d = header_info[3][0] # a 2D header that we'll need for making maps later 
    
    # let's return a library with all information to use later 
    cube_info = {"cube_data": cube_data, 
              "cube_header2d": cube_header2d, 
              "cube_ra2d": cube_ra2d, 
              "cube_dec2d": cube_dec2d, 
              "cube_vhel": cube_vhel}
    return cube_info 


# In[4]:


filename = cwd+'/data/CAR_EF04_trim.fits'
cube_info = read_datacube_information(filename)


# This is the function from previous code 1 exercise 4 where you extract velocity channel map at a ceratin velocity 

# In[5]:


def extract_vchan_map(obj_v, filename): 
    # chunk 1, process filename and read cube information 
    # return vchan map, actual velocity, and the corresponding pixel coordinates 
    cube_info = read_datacube_information(filename)
    cube_data = cube_info['cube_data']
    cube_ra2d = cube_info['cube_ra2d']
    cube_dec2d = cube_info['cube_dec2d']
    cube_vhel = cube_info['cube_vhel']
    cube_header2d = cube_info['cube_header2d']
    obj_v_pix = np.argmin(np.abs(obj_v - cube_vhel))
    obj_vchan = cube_data[obj_v_pix, :, :]
    
    # actual velocity from the velocity array 
    act_vhel = cube_vhel[obj_v_pix]

    return obj_vchan, act_vhel, obj_v_pix


# In[6]:


# an example 
obj_vchan, act_vhel, act_vhel_pix = extract_vchan_map(-200, filename)


# In[7]:


obj_vchan.shape


# # Implement Source finding algorithm <br>
# Let's try to use a package designed to find point sources (i.e., stars). It's not perfect as you'll see later, but we can try to improve the algorithm by adding more functions later. <br>
# * Installation: https://photutils.readthedocs.io/en/stable/install.html <br>
# * !pip install photutils

# In[8]:


from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture
from astropy.stats import sigma_clipped_stats
from astropy.io import ascii


# First extract a velocity channel map at velocity of obj_v

# In[9]:


obj_v = -230

# act_vhel is the actual velocity in the data 
# act_vhel_pix is the corresponding pixel coordinate for the velocity in the data
obj_vchan, act_vhel, act_vhel_pix = extract_vchan_map(obj_v, filename)


# Build folders in the working directory to save results to 

# In[10]:


import os 
if os.path.isdir(cwd+'/source_finding') == False: 
    os.mkdir(cwd+'/source_finding')
    print('created folder: ', cwd+'/source_finding')
source_dir = '{}/source_finding/vchan_-{:03d}'.format(cwd, abs(int(act_vhel)))
if os.path.isdir(source_dir) == False: 
    os.mkdir(source_dir)
    print('created folder: ', source_dir)


# Get some stats (mean, median, sigma) on this velocity channel using sigma_clipping

# In[11]:


# follow examples on this website: 
# https://photutils.readthedocs.io/en/stable/detection.html

source_img = obj_vchan.copy()

# find out the image statistics using sigma clipping 
clip_sigma = 5 # threshold for sigma clipping for the image statistics, 
               # this value doesn't change mean, median, and std much
img_mean, img_median, img_std = sigma_clipped_stats(source_img, sigma=clip_sigma)  
print('Image stats: mean={:.5f} K, median={:.5f} K, std={:.5f} K'.format(img_mean, 
                                                                         img_median, 
                                                                         img_std))


# 
# Detect point sources in an image using the DAOFIND (Stetson 1987) algorithm

# In[12]:


from photutils.detection import DAOStarFinder

# find Gaussian point-like sources in the image that have FWHMs of around n pixels 
# and have peaks approximately n-sigma above the background. 
# Running this class on the data yields an astropy Table containing the results of the star finder:
phot_fwhm_pixels = 10 # this nubmer is gonna affect the result a lot, try to find the best value 
phot_std_threshold = 8 
daofind = DAOStarFinder(fwhm=phot_fwhm_pixels, 
                        threshold=phot_std_threshold*img_std)


# Generate a image mask to take out region that we don't want to study 

# In[13]:


# there are some really large clouds that we don't need to classify, so let's make a mask to take out those region
# these regions will be plotted as pale red in the following images
def create_mask(source_image): 
    # hand coded, could change later as needed. 
    mask = np.zeros(source_image.shape, dtype=bool)
    mask[350:, 125:] = True # top right 
    mask[260:400, 10:85] = True # middle left 
    mask[220:260, 35:90] = True # middle bottom part 
    mask[190:220, 55:115] = True # middle bottom part 
    mask[160:190, 75:125] = True # middle bottom part 
    mask[80:160, 85:135] = True # middle bottom part 
    return mask

source_mask = create_mask(source_img)


# In[14]:


# use this command to control whether you want to apply the mask or not 
# could set it to True or False and see what changes
use_mask = False


# In[15]:


#  now find the sources 
if use_mask == True: 
    sources = daofind(source_img, mask=source_mask) 
    for col in sources.colnames:  
        sources[col].info.format = '%.4g'  # for consistent table output
    print('Initila run, found {:d} sources, with mask'.format(len(sources)))
else: 
    sources = daofind(source_img) 
    for col in sources.colnames:  
        sources[col].info.format = '%.4g'  # for consistent table output
    print('Initila run, found {:d} sources, without mask'.format(len(sources)))


# Apply a sigma threshold to determine whether a source detection is significant, here we say that the maximum flux of the source has to be 5 sigma higher than the noise std

# In[16]:


# get rid of srouces with peak flux less than n sigma 
nsig_peak_flux = 3
ind_keep = sources['peak'] >= nsig_peak_flux * img_std
sources = sources[ind_keep]
print('Getting rid of peak<5sig sources, found {:d} sources'.format(len(sources)))


# In[17]:


# xcentroid, ycentroid: x/y pixel coordinates for a source in the data cube 
sources.show_in_notebook()


# In[18]:


print(sources.dtype)


# Save the source catalog 

# In[19]:


# Formula
def angular_distance(deca,decb,raa,rab):
    deca = math.radians(deca)
    decb = math.radians(decb)
    raa = math.radians(raa)
    rab = math.radians(rab)
    angular_distance = 2*math.asin(math.sqrt(math.sin((decb-deca)/2)**2+(math.cos(decb)*math.cos(deca)*math.sin((rab-raa)/2)**2)))
    angular_distance = math.degrees(angular_distance)
    return angular_distance


# In[20]:


# now translate the pixel coordiantes of the sources into RA/DEC
from astropy.wcs import WCS 
cube_wcs = WCS(cube_info['cube_header2d'])

# traslate sources pixel coordinates into world coordinates 
sources_ra, sources_dec = cube_wcs.all_pix2world(sources['xcentroid'], 
                                                 sources['ycentroid'], 0)

# add that to the source table 
sources_final = sources.copy()
sources_final['RA_deg'] = np.around(sources_ra, decimals=4)
sources_final['DEC_deg'] = np.around(sources_dec, decimals=4)
sources_final['vhel_km/s'] = np.around(act_vhel, decimals=2)

# get rid of some columns that we don't need 
sources_final.remove_columns(['sky', 'flux', 'mag', 'npix', 
                              'roundness1', 'roundness2', 'sharpness'])
# get rid of duplicate sources
t_d = 0.65 #touching distance ie the distance allowed between source centers
def remove_duplicates(sources_table, distance_allowed):
    #print(len(sources_table))
    source_duplicates = []
    for i_source in np.arange(len(sources_table)):
        ra_a = sources_table['RA_deg'][i_source]
        dec_a = sources_table['DEC_deg'][i_source]
        for another_source in np.arange(len(sources_table)):
            ra_b = sources_table['RA_deg'][another_source]
            dec_b = sources_table['DEC_deg'][another_source]
            distance = angular_distance(dec_a,dec_b,ra_a,ra_b)
            if distance<=distance_allowed:
                if i_source != another_source and i_source < another_source:
                    print('Source', i_source, 'at', ra_a, 'by', dec_a, "&", 'Source', another_source, 'at', ra_b, 'by', dec_b, 'with a magnitude of', distance, 'degrees')
                    source_duplicates.append(i_source)
                    #print(source_duplicates)
                    
                    return source_duplicates
source_duplicates_draft = remove_duplicates(sources_final, t_d)
#print(source_duplicates_draft)
duplicates_list = []
if source_duplicates_draft:
    for x in source_duplicates_draft:
        if x not in duplicates_list:
            duplicates_list.append(x) 
if duplicates_list:
    for i in duplicates_list:
        sources_final.remove_row(i)
#print(len(sources_final))
# save the data for future uses     
table_name = '{}/vchan_-{:03d}_sources.csv'.format(source_dir, abs(int(act_vhel))) 
ascii.write(sources_final, table_name, format='csv', overwrite=True)
print('Table saved to ', table_name)


# Read in table and plot the result 

# In[21]:


from astropy.table import Table 
sources_final = Table.read(table_name, format='ascii')
sources_final.show_in_notebook()


# Now plot the sources on the map, first let's plot the velocity channel map, the mask (if use_mask == True), and the detected sources

# In[22]:


interp = 'none' 
dpi = 250
fs = 16

fig = plt.figure(figsize=(5, 6))
ax = fig.add_subplot(111, projection=cube_wcs)

# plot original image
norm = ImageNormalize(stretch=SqrtStretch())
im = ax.imshow(source_img, origin='lower', vmin=-img_std/2., vmax=6*img_std, 
               cmap=plt.cm.Greys, interpolation=interp)
# overlay the mask 
if use_mask == True: 
    plt.imshow(source_mask, cmap='Reds', origin='lower', alpha=0.2)

# add source 
positions = np.transpose((sources_final['xcentroid'], sources_final['ycentroid']))
apertures = CircularAperture(positions, r=4.)
# plot the detected sources 
apertures.plot(color='r', lw=1.5, alpha=1) 

# add source number to the circles 
for i_source in np.arange(len(sources_final)): 
    xpix = sources_final['xcentroid'][i_source]+3
    ypix = sources_final['ycentroid'][i_source]+2
    ax.text(xpix, ypix, '{}'.format(i_source), color='r', fontsize=fs-8)

# title 
title = 'Velocity = {:.1f} km/s'.format(act_vhel)
ax.set_title(title, fontsize=fs)

# add colrobar 
cb = fig.colorbar(im)
cb.ax.tick_params(labelsize=fs-4)
cb.set_label('Brightness Temperature Tb (K)', fontsize=fs-4)

# Coordinate labels and ticks 
ax.coords[0].set_axislabel('RA (J2000)', fontsize=fs) 
ax.coords[1].set_axislabel('DEC (J2000)', fontsize=fs)
ax.coords[0].set_major_formatter('d.d')
ax.coords[1].set_major_formatter('d.d')
ax.tick_params(labelsize=fs-2, direction='in')
ax.coords[0].display_minor_ticks(True)
ax.coords[1].display_minor_ticks(True)

# let's record the parameters on the image 
ax.text(260, 30, 'phot_fwhm_pixels={:.1f}'.format(phot_fwhm_pixels), 
        horizontalalignment='right', fontsize=8)
ax.text(260, 20, 'phot_std_threshold={:.1f}'.format(phot_std_threshold), 
        horizontalalignment='right', fontsize=8)
ax.text(260, 10, 'nsig_peak_flux={:.1f}'.format(nsig_peak_flux), 
        horizontalalignment='right', fontsize=8)

figure_name = '{}/vchan_-{:03d}.pdf'.format(source_dir, abs(int(act_vhel)))
fig.savefig(figure_name, dpi=dpi)


# A function to plot vchan map for each detected cloud 

# In[23]:


#def contour_fmt(x):
#    s = f"{x:.1f}"
#    if s.endswith("0"):
#        s = f"{x:.0f}"
#    return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"

def plot_vchan_each_source(i_source, source_img, img_std, act_vhel, 
                           fs=11, dpi=250, interp='none', 
                           img_width_npixel=20): 
    # input:img_width_npixel = 20 # plot +/-20 pixels from the center of the detected source 
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, projection=cube_wcs)
    
    # plot original image
    norm = ImageNormalize(stretch=SqrtStretch())
    source_peak = sources_final['peak'][i_source]
    im = ax.imshow(source_img, origin='lower', vmin=-img_std, vmax=1.2*source_peak, 
                   cmap=plt.cm.Greys, interpolation=interp)
    
    # add colrobar 
    cb = fig.colorbar(im, shrink=1, pad=0.02)
    cb.ax.tick_params(labelsize=fs-4)
    cb.set_label('Brightness Temperature Tb (K)', fontsize=fs-4)
    
    # add contour 
    levels = np.asarray([3, 8]) # sigma 
    cs = ax.contour(source_img, levels=levels*img_std, 
               colors=['k', 'm'], linewidths=1)
    fmt = {}
    strs = [r'{}$\sigma$'.format(kk) for kk in levels]
    for l, s in zip(cs.levels, strs):
        fmt[l] = s

    # Label every other level using strings
    ax.clabel(cs, cs.levels, inline=False, fmt=fmt, fontsize=5)
    
    # set xranges and y ranges 
    xmin = int(sources_final['xcentroid'][i_source]) - img_width_npixel
    xmax = int(sources_final['xcentroid'][i_source]) + img_width_npixel
    ymin = int(sources_final['ycentroid'][i_source]) - img_width_npixel
    ymax = int(sources_final['ycentroid'][i_source]) + img_width_npixel
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # add source 
    #positions = np.transpose((sources['xcentroid'][i_source], 
    #                          sources['ycentroid'][i_source]))
    #apertures = CircularAperture(positions, r=4.)
    ## plot the detected sources 
    #apertures.plot(color='r', lw=1, alpha=1) 
    ax.scatter([sources['xcentroid'][i_source]], [sources['ycentroid'][i_source]], 
               marker='+', color='r', lw=2, s=100)
    
    # title 
    # title = 'Velocity = {:.1f} km/s, Source={:d}'.format(act_vhel, i_source)
    title = 'S{:d}, Vel={:.1f} km/s, T map'.format(i_source, act_vhel)
    ax.set_title(title, fontsize=fs)
    
    # Coordinate labels and ticks 
    ax.coords[0].set_axislabel('RA (J2000)', fontsize=fs) 
    ax.coords[1].set_axislabel('DEC (J2000)', fontsize=fs)
    ax.coords[0].set_major_formatter('d.d')
    ax.coords[1].set_major_formatter('d.d')
    ax.tick_params(labelsize=fs-2, direction='in')
    ax.coords[0].display_minor_ticks(True)
    ax.coords[1].display_minor_ticks(True)
    
    
    # fig.tight_layout()
    # a little trick to print name in a format of -0XX or -1XX something, to make movie later 
    #import os 
    #source_dir = './source_finding/vchan_-{:03d}'.format(abs(int(act_vhel)))
    #if os.path.isdir(source_dir) == False: 
    #    os.mkdir(source_dir)
    figure_name = '{}/vchan_-{:03d}_source{:03d}.png'.format(source_dir, abs(int(act_vhel)), i_source)
    fig.savefig(figure_name, dpi=dpi)
    plt.close()
    print('Save to ', figure_name)


# In[24]:


img_width_npixel = 40 # fix for all the source maps 
for i_source in np.arange(len(sources_final)): 
    plot_vchan_each_source(i_source, source_img, img_std, act_vhel, 
                           img_width_npixel=img_width_npixel)
    # break


# # Moment 0 Map 

# In[25]:


from __future__ import print_function

def zero_moment(data, vel, vmin, vmax, calNHI=True):
    '''
    Integrate the cube over a [vmin, vmax] velocity range.

        NHI = 1.823e18 * init_vmin^vmax (Tv dv), where Tv is the pixel values
        of the cube, and dv is the velocity step of the integration.
        check here: http://www.atnf.csiro.au/people/Tobias.Westmeier/tools_hihelpers.php#velocity
    Input:
          data: 3D data cube with dimension of (velocity, coordinate1, coordinate2)
          vel: 1D velocity vector. same size as data.shape[0]
          vmin, vmax: the integration range.
    Return:
          2D image of HI column density map
    History: updated as of 2016.10.03. Yong Zheng @ Columbia Astro
    Updated as of July 15, get rid of filter function for AST19 
    '''

    k2cm2 = 1.823e18  # from K km/s to cm-2. Can be found in the ISM book, Draine 2004

    ind = np.all([vel>=vmin, vel<=vmax], axis=0)
    new_d = data[ind, :, :]
    new_v = vel[ind]

    delv = np.fabs(np.mean(new_v[1:]-new_v[:-1]))  # the velocity step

    #if dofilter == True:
    #    new_d, cube_mask = filter_cube(new_d, fregion)

    colden = (new_d*delv).sum(0)
    if calNHI == True:
        colden = colden*k2cm2

    return colden


# In[26]:


def plot_mom0_each_source(i_source, source_img, img_min,img_max, act_vhel,vmin, vmax,
                           fs=11, dpi=250, interp='none', 
                           img_width_npixel=20, use_log=True): 
    # input:img_width_npixel = 20 # plot +/-20 pixels from the center of the detected source 
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, projection=cube_wcs)
    
    # plot original image
    # norm = ImageNormalize(stretch=SqrtStretch())
    #source_peak = sources_final['peak'][i_source]
    if use_log == False: 
        im = ax.imshow(source_img, origin='lower', vmin=img_min, vmax=img_max, 
                       cmap=plt.cm.Reds, interpolation=interp)
    else: 
        im = ax.imshow(np.log10(source_img), origin='lower', 
                       vmin=np.log10(img_min), vmax=np.log10(img_max), 
                       cmap=plt.cm.Reds, interpolation=interp)
    
    # add colrobar 
    cb = fig.colorbar(im, shrink=1, pad=0.02)
    cb.ax.tick_params(labelsize=fs-4)
    if use_log == False: 
        cb.set_label('Column Density (cm-2)', fontsize=fs-4)
    else: 
        cb.set_label('Log Column Density (cm-2)', fontsize=fs-4)
    
    # add contour
    if use_log == False: 
        levels = [5e18, 1e19, 5e19] # sigma 
    else: 
        levels = [np.log10(5e18), np.log10(1e19), np.log10(5e19)]
    cs = ax.contour(source_img, levels=levels, 
               colors=['steelblue', 'm', 'k'], linewidths=1)
    #fmt = {}
    #strs = [r'{}$\sigma$'.format(kk) for kk in levels]
    #for l, s in zip(cs.levels, strs):
    #    fmt[l] = s

    # Label every other level using strings
    #ax.clabel(cs, cs.levels, inline=False, fmt=fmt, fontsize=5)
    
    # set xranges and y ranges 
    
    xmin = int(sources_final['xcentroid'][i_source]) - img_width_npixel
    xmax = int(sources_final['xcentroid'][i_source]) + img_width_npixel
    ymin = int(sources_final['ycentroid'][i_source]) - img_width_npixel
    ymax = int(sources_final['ycentroid'][i_source]) + img_width_npixel
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # add source 
    #positions = np.transpose((sources['xcentroid'][i_source], 
    #                          sources['ycentroid'][i_source]))
    #apertures = CircularAperture(positions, r=4.)
    ## plot the detected sources 
    #apertures.plot(color='r', lw=1, alpha=1) 
    ax.scatter([sources['xcentroid'][i_source]], [sources['ycentroid'][i_source]], 
               marker='+', color='k', lw=1.5, s=50)
    
    # title 
    title = 'S{:d}, Vel=[{:.1f}, {:.1f}]km/s, NHI Map'.format(i_source, vmin, vmax)
    ax.set_title(title, fontsize=fs)
    
    # Coordinate labels and ticks 
    ax.coords[0].set_axislabel('RA (J2000)', fontsize=fs) 
    ax.coords[1].set_axislabel('DEC (J2000)', fontsize=fs)
    ax.coords[0].set_major_formatter('d.d')
    ax.coords[1].set_major_formatter('d.d')
    ax.tick_params(labelsize=fs-2, direction='in')
    ax.coords[0].display_minor_ticks(True)
    ax.coords[1].display_minor_ticks(True)
    
    
    # fig.tight_layout()
    # a little trick to print name in a format of -0XX or -1XX something, to make movie later 
    #import os 
    #source_dir = './source_finding/vchan_-{:03d}'.format(abs(int(act_vhel)))
    #if os.path.isdir(source_dir) == False: 
    #    os.mkdir(source_dir)
    figure_name = '{}/vchan_-{:03d}_source{:03d}_HI.png'.format(source_dir, abs(int(act_vhel)), i_source)
    fig.savefig(figure_name, dpi=dpi)
    plt.close()
    print('Save to ', figure_name)


# In[27]:


# traslate sources pixel coordinates into world coordinates 

def create_source_graph_mom0(sources_final,cube_info,d_velocity,cube_wcs):

    #source_mask = create_mask(source_img)
        
    for i_source in np.arange(len(sources_final)): 
        xpix = int(sources_final['xcentroid'][i_source])
        ypix = int(sources_final['ycentroid'][i_source])
        
        source_vel = sources_final['vhel_km/s'][i_source]
        int_vmin = source_vel - d_velocity
        int_vmax = source_vel + d_velocity
        #print(vmin)
        #print(vmax)
        m0_img = zero_moment(cube_info['cube_data'],cube_info['cube_vhel'], 
                           int_vmin, int_vmax, calNHI=True)
        m0_mean, m0_median, m0_std = sigma_clipped_stats(m0_img, sigma=3)
        
        # find min and max pixels within +/- 20 pixels of the source 
        npix = 20 
        low_y = np.max([ypix-npix, 0])
        high_y = np.min([ypix+npix, m0_img.shape[0]-1])
        left_x = np.min([0, xpix-npix])
        right_x = np.max([xpix+npix, m0_img.shape[1]-1])
        sub_img = m0_img[low_y:high_y, left_x:right_x]
        img_min = m0_mean
        img_max = np.nanmax(sub_img)
        # plot the source 
        plot_mom0_each_source(i_source, m0_img,img_min,img_max, source_vel,
                              int_vmin, int_vmax, fs=11, dpi=250, interp='none', 
                              img_width_npixel=img_width_npixel, use_log=False)


# In[28]:


d_velocity = 15 # we'll integrate moment maps from source_velocity-d_velocity to source_velocity+d_velocity 
cube_wcs = WCS(cube_info['cube_header2d']) # for projection later 

# read in the source_final data from the previous run 
from astropy.table import Table 
sources_final = Table.read(table_name, format='ascii')

create_source_graph_mom0(sources_final, cube_info, d_velocity , cube_wcs)


# # Moment 1 Map 

# In[29]:


def first_moment(data, vel, vmin, vmax):
    '''
    Flux-weighted velocity, calculated within vmin and vmax.
        fluxwt_vel = init_vmin^vmax (Tv * vel *dv) / init_vmin^vmax(Tv * dv)
        check here: http://www.atnf.csiro.au/people/Tobias.Westmeier/tools_hihelpers.php#velocity
    Input:
          data: 3D data cube with dimension of (velocity, coordinate1, coordinate2)
          vel: 1D velocity vector. same size as data.shape[0]
          vmin, vmax: the integration range.
    Return:
          2D image of flux-weighted velocity
    History: updated as of 2016.10.03. Yong Zheng @ Columbia Astro
    '''

    ind = np.all([vel>=vmin, vel<=vmax], axis=0)
    new_d = data[ind, :, :]
    new_v = np.reshape(vel[ind], (vel[ind].size, 1, 1))

    #if dofilter == True:
    #    # new_d = filter_cube(new_d, fregion)
    #    new_d, cube_mask = filter_cube(new_d, fregion)
    # delv = np.fabs(np.mean(new_v[1:]-new_v[:-1]))  # the velocity step

    top_init = (new_v * new_d).sum(axis=0)  # both top/bottom needs a delv, but they cancel in the end
    bottom_init = new_d.sum(axis=0)       # so ignore delv here

    # to avoid 0 in the denominator
    ind0 = bottom_init==0.
    bottom_init[ind0] = 1.
    fluxwt_vel = top_init/bottom_init

    # get rid of pixels with peculiar velocities after the division
    fluxwt_vel[np.isnan(fluxwt_vel)] = -10000
    ind1 = fluxwt_vel<vmin
    
    ind2 = fluxwt_vel>vmax


    ind12 = np.any([ind1, ind2], axis=0)
    indnan = np.any([ind12, ind0], axis=0)

    fluxwt_vel[indnan] = np.nan

    return fluxwt_vel


# In[30]:


def plot_mom1_each_source(i_source, m1_img, m0_img, act_vhel,vmin, vmax,
                           fs=11, dpi=250, interp='none', 
                           img_width_npixel=20): 
    # input:img_width_npixel = 20 # plot +/-20 pixels from the center of the detected source 
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, projection=cube_wcs)
    
    # plot original image
    im = ax.imshow(m1_img, origin='lower', vmin=vmin-5, vmax=vmax+5, 
                   cmap=plt.cm.coolwarm, interpolation=interp)
    
    # add colrobar 
    cb = fig.colorbar(im, shrink=1, pad=0.02)
    cb.ax.tick_params(labelsize=fs-4)
    cb.set_label('Averaged Velocity (km/s)', fontsize=fs-4)
    
    # add contour
    levels = [5e18, 1e19, 5e19] # sigma 
    cs = ax.contour(m0_img, levels=levels, 
               colors=['steelblue', 'm', 'k'], linewidths=1)
    
    # set xranges and y ranges 
    xmin = int(sources_final['xcentroid'][i_source]) - img_width_npixel
    xmax = int(sources_final['xcentroid'][i_source]) + img_width_npixel
    ymin = int(sources_final['ycentroid'][i_source]) - img_width_npixel
    ymax = int(sources_final['ycentroid'][i_source]) + img_width_npixel
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # add source 
    ax.scatter([sources['xcentroid'][i_source]], [sources['ycentroid'][i_source]], 
               marker='+', color='k', lw=1.5, s=50)
    
    # title 
    title = 'S{:d}, Vel=[{:.1f}, {:.1f}]km/s, V Map'.format(i_source, vmin, vmax)
    ax.set_title(title, fontsize=fs)
    
    # Coordinate labels and ticks 
    ax.coords[0].set_axislabel('RA (J2000)', fontsize=fs) 
    ax.coords[1].set_axislabel('DEC (J2000)', fontsize=fs)
    ax.coords[0].set_major_formatter('d.d')
    ax.coords[1].set_major_formatter('d.d')
    ax.tick_params(labelsize=fs-2, direction='in')
    ax.coords[0].display_minor_ticks(True)
    ax.coords[1].display_minor_ticks(True)
    
    
    # fig.tight_layout()
    # a little trick to print name in a format of -0XX or -1XX something, to make movie later 
    #import os 
    #source_dir = './source_finding/vchan_-{:03d}'.format(abs(int(act_vhel)))
    #if os.path.isdir(source_dir) == False: 
    #    os.mkdir(source_dir)
    figure_name = '{}/vchan_-{:03d}_source{:03d}_aver-vel.png'.format(source_dir, abs(int(act_vhel)), i_source)
    fig.savefig(figure_name, dpi=dpi)
    plt.close()
    print('Save to ', figure_name)


# In[31]:


# traslate sources pixel coordinates into world coordinates 

def create_source_graph_mom1(sources_final,cube_info,d_velocity,cube_wcs):

    #source_mask = create_mask(source_img)
        
    for i_source in np.arange(len(sources_final)): 
        xpix = int(sources_final['xcentroid'][i_source])
        ypix = int(sources_final['ycentroid'][i_source])
        
        source_vel = sources_final['vhel_km/s'][i_source]
        int_vmin = source_vel - d_velocity
        int_vmax = source_vel + d_velocity
        #print(vmin)
        #print(vmax)
        # first_moment(data, vel, vmin, vmax)
        m1_img = first_moment(cube_info['cube_data'],cube_info['cube_vhel'], 
                           int_vmin, int_vmax)
        
        # also calculate m0_img as a mask 
        m0_img = zero_moment(cube_info['cube_data'],cube_info['cube_vhel'], 
                           int_vmin, int_vmax, calNHI=True)
        
        # mask out weird NHI region 
        nan_val = m0_img <= 0
        m1_img[nan_val] = np.nan 
        
        # find min and max pixels within +/- 20 pixels of the source 
        plot_mom1_each_source(i_source, m1_img, m0_img, source_vel,int_vmin, int_vmax,
                           fs=11, dpi=250, interp='none', 
                           img_width_npixel=img_width_npixel)


# In[32]:


create_source_graph_mom1(sources_final,cube_info,d_velocity,cube_wcs)


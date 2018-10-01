#!/usr/bin/env python
#
# Copyright (C) 2017 Joel S. Bader
# You may use, distribute, and modify this code
# under the terms of the Python Software Foundation License Version 2
# available at https://www.python.org/download/releases/2.7/license/
#
import argparse
import os
import fnmatch
import math 
import string
import exifread
import cPickle
import copy 

import numpy as np
import scipy as sp
from scipy import interpolate
from scipy.stats import rankdata
from scipy.stats import pearsonr
#import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as mpcm 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as mlines
from matplotlib.patches import Polygon

from skimage.restoration import unwrap_phase

import statsmodels.api as sm

from PIL import Image, ImageDraw

from scipy import ndimage
from skimage.morphology import disk, dilation, watershed, closing, skeletonize, medial_axis


from sklearn.cluster import KMeans

import logging

logging.basicConfig(format='%(levelname)s %(name)s.%(funcName)s: %(message)s')
logger = logging.getLogger('stack_images')
logger.setLevel(logging.INFO)


MAXK = 21

ANNOTS = ['Normal', 'NAT', 'Tumor']
DAYNUMS = ['0','5']

def get_files_recursive(indir, pattern_str):
    ret = [ ]
    for root, dirnames, filenames in os.walk(indir):
        for filename in fnmatch.filter(filenames, pattern_str):
            ret.append( os.path.join(root, filename) )
    return(ret)

def convert_fullpath_to_dict(fullpath_list):
    base_dict = dict()
    for my_full in fullpath_list:
        (my_root, my_base) = os.path.split(my_full)
        # check that days match
        toks = my_base.split('_')
        
        if my_base in base_dict:
            logger.warn('repeated basename %s fullpath %s %s', my_base, base_dict[my_base], my_full)
            continue
        base_dict[my_base] = my_full
    return(base_dict)

def file_to_points(infile, dim=2):
    #logger.info('reading %dD tuples from %s', dim, infile)
    fp = open(infile, 'r')
    ret_list = [ ]
    for line in fp:
        toks = line.strip().split()
        assert(len(toks) >= dim), 'line too short, expected %d tokens: %s' % (dim, line)
        rec_float = [ float(x) for x in toks[0:dim] ]
        rec_tuple = tuple(rec_float)
        ret_list.append(rec_tuple)
    fp.close()
    nrec = len(ret_list)
    # logger.info('read %d tuples', nrec)
    points = np.asarray(ret_list)
    #logger.info('shape of points: %s', str(points.shape))
    return(points)

def get_contour_length(points):
    m = len(points)
    dim = points.shape[1]
    contour_length = np.zeros(m)
    prev = 0.0
    for i in range(len(points)):
        j = (i - 1) % m
        mysq = 0.0
        for k in range(dim):
            term = points[i][k] - points[j][k]
            mysq += term * term
        delta = math.sqrt(mysq)
        contour_length[i] = prev + delta
        prev = contour_length[i]
    # logger.info('contour length %f', contour_length[m-1])
    return(contour_length)

def get_interpolate(points, contour_in, contour_out):
    # interpolate the contour to a regularly spaced grid
    nrow = len(contour_out)
    ncol = points.shape[1]
    # logger.info('interpolating to %s rows, %s cols', nrow, ncol)
    # create an array with the proper shape
    ret = np.zeros( (nrow, ncol) )
    
    # add the last point as the initial point with distance 0
    c0 = np.zeros(1)
    x = np.concatenate((c0, contour_in))
    p_last = np.array(points[[-1],:]) 
    new_pts = np.concatenate((p_last, points))
    #logger.info('countour %s %s', str(contour_in[0:5]), str(x[0:5]))
    #logger.info('shapes %s %s', str(points.shape), str(p_last.shape))
    #logger.info('orig %s new %s last %s', str(points[0:5,:]), str(new_pts[0:5,:]), str(p_last))
    
    for k in range(ncol):
        y = new_pts[:,k]
        fn = interpolate.interp1d(x, y, kind='linear')
        y_grid = fn(contour_out)
        for i in range(nrow):
            ret[i,k] = y_grid[i]

    return ret

def get_area(points):
    n = len(points) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i,0] * points[j,1]
        area -= points[j,0] * points[i,1]
    area = abs(area) / 2.0
    return area

def get_rfft(points_grid):
    n = points_grid.shape[0]
    dim = points_grid.shape[1]
    nfft = (n/2) + 1
    ret = np.zeros( (nfft, dim), dtype=complex )
    for k in range(dim):
        a = np.fft.rfft(points_grid[:,k])
        #logger.info('assumed shape %s', str(ret.shape))
        #logger.info('actual shape %s', str(a.shape))
        ret[:,k] = a
    # logger.info('rfft start end:\n%s\n%s', str(ret[0:5,:]), str(ret[-5:,:]))
    return(ret)

def get_irfft(points_rfft):
    nk = points_rfft.shape[0]
    dim = points_rfft.shape[1]
    n = 2*(nk - 1)
    ret = np.zeros( (n, dim) )
    for k in range(dim):
        a = np.fft.irfft(points_rfft[:,k])
        #logger.info('assumed shape %s', str(ret.shape))
        #logger.info('actual shape %s', str(a.shape))
        ret[:,k] = a
    # logger.info('irfft start end:\n%s\n%s', str(ret[0:5,:]), str(ret[-5:,:]))
    return(ret)

def get_power(points_rfft):
    nfreq = points_rfft.shape[0]
    dim = points_rfft.shape[1]
    power = np.zeros(nfreq)
    for k in range(dim):
        a = np.abs( points_rfft[:,k] ) # a is a vector
        power = power + a**2           # element-wise accumulation
    # the first element of power should be <x>^2 + <y>^2 + ..., can set to zero
    # the last element of power should probably be doubled
    # logger.info('power: %s ... %s', str(power[0:10]), str(power[-5:]))
    return(power)

def normalize_points_by_power(xy_hat):
    nk = xy_hat.shape[0]
    dim = xy_hat.shape[1]
    n = 2 * (nk - 1)
    xy_hat_norm = xy_hat.copy()
    for k in range(dim):
        xy_hat_norm[0,k] = 0.0
    #    points_centered[:,k] = np.fft.irfft(rfft_centered[:,k])

    xy_norm = np.zeros( (n, dim) )

    # this normalization sets the first fourier component to that of a unit circle
    # for a circle with radius r, power = n^2 r^2 / 2 where n = number of points
    # so multiply by 1/r = n / sqrt(2 x power)
    p1 = 0
    for k in range(dim):
        p1 = p1 + np.abs(xy_hat[1,k])**2
    facpower = float(n) / math.sqrt(2.0 * p1)
    xy_hat_norm = facpower * xy_hat_norm

    for k in range(dim):
        xy_norm[:,k] = np.fft.irfft(xy_hat_norm[:,k])

    return(xy_norm)
    
def normalize_points(points_rfft, area):
    nk = points_rfft.shape[0]
    dim = points_rfft.shape[1]
    n = 2 * (nk - 1)
    points_centered = np.zeros( (n, dim) )
    points_normarea = np.zeros( (n, dim) )
    points_normpower = np.zeros( (n, dim) )
    rfft_centered = points_rfft.copy()
    for k in range(dim):
        rfft_centered[0,k] = 0.0
        points_centered[:,k] = np.fft.irfft(rfft_centered[:,k])

    # for unit radius, area = pi r^2
    # so r = sqrt(area / pi)
    # divide by sqrt(area/pi) to get unit radius
    facarea = 1.0 / math.sqrt(area / math.pi )
    rfft_normarea = facarea * rfft_centered

    # this normalization sets the first fourier component to that of a unit circle
    # for a circle with radius r, power = n^2 r^2 / 2 where n = number of points
    # so multiply by 1/r = n / sqrt(2 x power)
    power = 0
    for k in range(dim):
        power = power + np.abs(points_rfft[1,k])**2
    facpower = float(n) / math.sqrt(2.0 * power)
    rfft_normpower = facpower * rfft_centered

    for k in range(dim):
        points_normarea[:,k] = np.fft.irfft(rfft_normarea[:,k])
        points_normpower[:,k] = np.fft.irfft(rfft_normpower[:,k])

    power_normarea = np.zeros(nk)
    power_normpower = np.zeros(nk)
    # normalize so that a unit circle has power = 1 for first component
    # multiply by 2/n^2
    prefac = 2.0/(n*n)
    for k in range(dim):
        ar = np.abs(rfft_normarea[:,k])
        power_normarea = power_normarea + prefac * ar**2
        ap = np.abs(rfft_normpower[:,k])
        power_normpower = power_normpower + prefac * ap**2
    return(points_centered, points_normarea, points_normpower,
           power_normarea, power_normpower)

def get_power_moments(power_rfft, n = -1):
    mysum = 0.0
    mymean = 0.0
    mylen = len(power_rfft)
    if (n == -1):
        n = len(power_rfft)
    assert(n <= mylen), 'bad length: ' + str(n)
    # logger.info('n = %s', str(n))
    mysum = sum(power_rfft[2:n])
    myfreq = range(n)
    mysum1 = sum(power_rfft[2:n] * myfreq[2:n])
    mysum2 = sum( (power_rfft[2:n] * myfreq[2:n]) * myfreq[2:n] )
    return(mysum, mysum1, mysum2)

def mad(a, scale = 1.482602218505602):
    ret = np.median(np.absolute(a - np.median(a)))
    ret = ret * scale
    return(ret)

def close_polygon(pts):
    x = list(pts[:,0])
    x.append(x[0])
    y = list(pts[:,1])
    y.append(y[0])
    return(x, y)

def write_boundary(pdf, coord_file, points_grid, points_irfft,
                   points_centered, points_normarea, points_normpower,
                   power_normarea, power_normpower):

    logger.info('%s', coord_file)
    (path, file) = os.path.split(coord_file)
    plt.figure(figsize=(16, 4))

    # points and irfft
    plt.subplot(131)
    plt.scatter(points_grid[:,0], points_grid[:,1], facecolors='none', edgecolors='b',
                label='grid interpolation')
    (x, y) = close_polygon(points_irfft)
    plt.plot(x, y, 'k', label='irfft')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.gca().invert_yaxis()
    #plt.legend()
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.title('%s' % file)
    
    plt.subplot(132)
    plt.title('Boundary, Centered & Scaled')    
    points = [ points_normarea, points_normpower ]
    colors = ['k', 'r']
    labels = ['Norm Area', 'Norm Power']
    for (p, c, l) in zip(points, colors, labels):
        (x,y) = close_polygon(p)
        plt.plot(x,y,c,label=l)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.legend(loc='best', fancybox=True, framealpha=0.5)

    plt.subplot(133)
    kk = range(11)
    baseline = np.zeros(len(power_normarea))
    baseline[1] = 1.0
    powers = [ power_normarea, power_normpower ]
    for (p, c, l) in zip(powers, colors, labels):
        plt.plot(kk, p[kk] - baseline[kk], c, label=l)
    plt.xlabel('Fourier Component')
    plt.ylabel('Power')
    plt.title('Spectral Power')
    plt.legend(loc='best', fancybox=True, framealpha=0.5)    
    
    pdf.savefig()
    plt.close()
    return None

def xyfile_to_spectrum(xyfile, nfft):
    xy_raw = file_to_points(xyfile)
    contour_length = get_contour_length(xy_raw)
    # make a regular grid
    tot_length = contour_length[-1]
    ds = tot_length / float(nfft)
    contour_grid = np.linspace(ds, tot_length, num=nfft)
    xy_interp = get_interpolate(xy_raw, contour_length, contour_grid)
    area = get_area(xy_interp)
    form_factor = (tot_length * tot_length) / (4.0 * math.pi * area)
    xy_hat = get_rfft(xy_interp)
    power_hat = get_power(xy_hat)
    # points_irfft = get_irfft(points_rfft)
    # try some normalizations
    #(points_centered, points_normarea, points_normpower,
    #power_normarea, power_normpower) = normalize_points(points_rfft, area)
    #write_boundary(boundarypdf, coord_file,
    #                   points_grid, points_irfft,
    #                   points_centered, points_normarea, points_normpower,
    #                   power_normarea, power_normpower)
    power_norm = np.copy(power_hat)
    power_norm[0] = 0.0
    fac = 1.0 / power_norm[1]
    power_norm = fac * power_norm
    return(xy_raw, xy_interp, xy_hat, tot_length, area, form_factor, power_norm)

def rgb2gray(img_rgb):
    if (img_rgb.shape[2] == 4):
        img_rgb = img_rgb[:,:,:3]
    assert(len(img_rgb.shape) == 3), 'bad shape'
    assert(img_rgb.shape[2] == 3), 'bad rgb dimension'
    img_gray = img_rgb[:,:,0] + img_rgb[:,:,1] + img_rgb[:,:,2]
    img_gray = img_gray / 3.0
    #logger.info('rgb shape %s converted to gray shape %s',
    #           str(img_rgb.shape), str(img_gray.shape))
    #for i in range(10):
    #    for j in range(10):
    #        logger.info('%d %d %f %f %f %f',
    #                    i, j, img_rgb[i,j,0], img_rgb[i,j,1], img_rgb[i,j,2], img_gray[i,j])
    return(img_gray)

def get_tags(imgfile):
    fp = open(imgfile, 'rb')
    tags = exifread.process_file(fp)
    fp.close()
    #logger.info('%s exif tags')
    #for t in sorted(tags.keys()):
    #    logger.info('%s -> %s', t, tags[t] )
    return(tags)

def create_boundary(img_orig, points, scale, fillstr):
    # change (x,y) to (row,col) and do the scaling
    points_scaled = [ ]
    for (x,y) in points:
        (row, col) = (scale * y, scale * x)
        row = min(row, img_orig.shape[0] - 1)
        col = min(col, img_orig.shape[1] - 1)
        points_scaled.append(col)
        points_scaled.append(row)
    boundary_pil = Image.new('1', (img_orig.shape[1], img_orig.shape[0]), 0)
    fillarg = None
    if (fillstr == 'fill'):
        fillarg = 1
    elif (fillstr == 'nofill'):
        fillarg = None
    else:
        logger.warn('unknown fill: %s', fillstr)
    ImageDraw.Draw(boundary_pil).polygon(points_scaled, outline=1, fill=fillarg)
    boundary_np = np.array(boundary_pil)
    # maxval = np.amax(img_orig)
    # img_boundary = np.copy(img_orig)
    # img_boundary[ boundary_np ] = maxval
    img_boundary = np.copy(img_orig)
    img_boundary[:,:,...] = 0
    img_boundary[boundary_np] = 1
    return(img_boundary, boundary_np)

def get_edge_pixles(img, xy, scale, distance=10):
    img = rgb2gray(img)
    (mask, array) = create_boundary(img, xy, scale, 'fill')
    #kernel
    k = disk(distance)
    #convolve mask and normalize output
    mask_c=ndimage.convolve(mask, k)
    mask_c = mask_c/(np.amax(mask_c))
    #create new mask for edge pixels
    edge_mask = np.zeros(mask.shape)
    #remask and threshold to get edge pixels
    edge_mask[(mask == 1) & (mask_c < 1)] = 1
    #create new mask for center pixels
    center_mask = np.zeros(mask.shape)
    center_mask[mask_c == 1] = 1
    return (edge_mask, center_mask)

def get_pixelsummary(arr):
    ret = dict()
    ret['cnt'] = arr.shape[0]
    ret['mean'] = np.mean(arr)
    ret['median'] = np.median(arr)
    ret['sd'] = np.std(arr)
    ret['mad'] =  mad(arr)
    return(ret)

def read_image_orig(fullpath):
    img_orig = mpimg.imread(fullpath)
    img_rgb = None
    tag_color = None
    if (img_orig.ndim == 3):
        tag_color = 'rgb'
        img_rgb = img_orig
    elif (img_orig.ndim == 2):
        tag_color = 'gray'
        myshape = ( img_orig.shape[0], img_orig.shape[1], 3 )
        img_rgb = np.zeros(myshape)
        for k in range(3):
            img_rgb[:,:,k] = img_orig[:,:]
    else:
        logger.info('bad shape: %s', str(img_orig.shape))
        img_rgb = img_orig

    tags = get_tags(fullpath)
    #for t in sorted(tags.keys()):
    #    logger.info('%s tag %s value %s', t, str(tags[t]))
    xtag = tags.get('Image XResolution', "1")
    ytag = tags.get('Image YResolution', "1")
    samplesperpixel = str(tags['Image SamplesPerPixel'])
    xresolution = str(xtag)
    yresolution = str(ytag)
    if (xresolution != yresolution):
        logger.warn('%s bad resolution %s %s', fullpath, xresolution, yresolution)
    #scale = float(xresolution)
    scale = eval("1.0*" + xresolution) # sometimes scalestr is 122093/62500
    #Scale is determined by looking at the images
    scale = 2.0
    #logger.info('resolution x %s y %s x %s y %s scale %s scale %s', xtag, ytag, xresolution, yresolution, scalestr, str(scale))
    
    phototag = tags.get('Image PhotometricInterpretation', 'NA')
    pixelmin = np.amin(img_rgb)
    pixelmax = np.amax(img_rgb)
    #logger.info('%s phototag %s before k14 min %d max %d dic min %d max %d', fullpath, phototag, pixelmink14, pixelmaxk14, pixelmindic, pixelmaxdic)
    if (str(phototag) == '1'):
        # logger.info('inverting scale')
        img_rgb = 255 - img_rgb
    pixelmin = np.amin(img_rgb)
    pixelmax = np.amax(img_rgb)
    #logger.info('%s phototag %s  after k14 min %d max %d dic min %d max %d', fullpath, phototag, pixelmink14, pixelmaxk14, pixelmindic, pixelmaxdic)
    
    return(img_rgb, scale, str(phototag), tag_color)

def test(data_folder='.', file_number='0'):
    dic_folder_name = 'DIC'
    k14_folder_name = 'K14'
    xy_coord_folder_name = 'XY'
    number_of_rows = 1

    plotfile = os.path.join(data_folder, 'organoids.pdf')
    pdf = PdfPages(plotfile)
    
    org_folder_names = [f for f in os.listdir(data_folder + '/' + dic_folder_name) if not f.startswith('.')]

    org_folder_names = sorted(org_folder_names)

    k_max = 128/2 # the k index goes from -nfft/2 to +nfft/2
    k_vector = np.arange(1 + k_max)
    k_sq = k_vector * k_vector # element multiply

    pkm = k_vector * math.pi / float(k_max)
    fac_smooth = (np.cos(pkm))**2
    fac_weight = ((float(k_max)/math.pi)*np.sin(pkm))**2
    
    xy_interp_dict = dict()
    xy_norm_dict = dict()
    xy_hat_dict = dict()
    weightspectrum_dict = dict()

    plt.figure(figsize=(30, number_of_rows*4))

    xy_file = os.path.join(data_folder, xy_coord_folder_name, org_folder_names[0], file_number+".txt")
    (xy_raw, xy_interp, xy_hat, tot_length, area, form_factor, power_norm) = xyfile_to_spectrum(xy_file, 128)

    (img_dic, scale_dic, tag_photometric_dic, tag_color_dic) = read_image_orig(os.path.join(data_folder, dic_folder_name, org_folder_names[0], file_number+".tif"))
    (img_k14, scale_k14, tag_photometric_k14, tag_color_k14) = read_image_orig(os.path.join(data_folder, k14_folder_name, org_folder_names[0], file_number+".tif"))


    power_norm = fac_smooth * power_norm
    power_ksq = fac_weight * power_norm
    sumpower = sum(power_norm[2:])
    sumpowerksq = sum(power_ksq[2:])
    
    xy_interp_dict[0] = xy_interp
    xy_norm_dict[0] = normalize_points_by_power(xy_hat)
    weightspectrum_dict[0] = power_ksq
    xy_hat_dict[0] = xy_hat


    i = 0 # plot sequence number starting point

    # K14 image
    i += 1
    plt.subplot(number_of_rows, 4, i)
    plt.imshow(img_k14)
    plt.title('K14')
    plt.ylabel('%s' % 0)

    # DIC image
    i += 1
    plt.subplot(number_of_rows, 4, i)
    plt.title('DIC')
    plt.imshow(img_dic)

    # boundary from raw points
    i += 1
    ax = plt.subplot(number_of_rows, 4, i)
    # make sure the path is closed
    x = list(xy_raw[:,0])
    y = list(xy_raw[:,1])
    x.append(x[0])
    y.append(y[0])
    xx = [ scale_dic * val for val in x ]
    yy = [ scale_dic * val for val in y]
    plt.plot(xx, yy, 'k', label='Boundary from file')
    plt.scatter(scale_dic * xy_interp[:,0], scale_dic * xy_interp[:,1],
            facecolors='none', edgecolors='b')
    # ax.axis('equal')
    #ax.set_xlim(0, img_dic.shape[1] - 1)
    #ax.set_ylim(0, img_dic.shape[0] - 1)
    plt.title('Form Factor %.3f' % (form_factor))
    plt.gca().invert_yaxis()
    
    # power spectrum
    i += 1
    plt.subplot(number_of_rows, 4, i)
    x = range(len(power_norm))
    # logger.info('len(x) %d len(pwr) %d', len(x), len(power_norm))
    # plt.plot(x[2:], power_norm[2:], 'k')
    plt.plot(x[2:], power_ksq[2:], 'b')
    # plt.xlabel('Frequency')
    plt.ylabel('Power k^2')
    plt.title('Spectral %.3f' % (sumpowerksq))

    #pdf.savefig()
    plt.show()
    #plt.close()
    #pdf.close()
    

def combine_images(data_folder, output_folder):
    print ("Info: combining images ...")
    if (not os.path.isdir(output_folder)):
        os.makedirs(output_folder)

    folders = [f for f in os.listdir(os.path.join(data_folder, 'DIC')) if not f.startswith('.')]

    for folder in folders:
        if (not os.path.isdir(os.path.join(output_folder, folder))):
            os.makedirs(os.path.join(output_folder, folder))
        files = os.listdir(os.path.join(data_folder, 'DIC', folder))
        files = sorted(files)
        for f in files:
            output_file_name = os.path.join(output_folder, folder, f)
            (img_DIC, scale_DIC, tag_photometric_DIC, tag_color_DIC) = read_image_orig(os.path.join(data_folder, 'DIC', folder, f))
            (img_K14, scale_K14, tag_photometric_K14, tag_color_K14) = read_image_orig(os.path.join(data_folder, 'K14', folder, f))

            #The images are mostly RGBA
            if(img_DIC.shape[2] == 3):
                combined_img = np.concatenate((img_DIC, img_K14), axis=1)
            elif (img_DIC.shape[2] > 3):
                combined_img = np.concatenate((img_DIC[:,:,:3], img_K14[:,:,:3]), axis=1)
            else:
                print ("Info: image dimentions are invalid. Skipping ...")
                continue

            new_img = Image.fromarray(combined_img)

            new_img.save(output_file_name)

def match_coord_image(all_coord, all_image):
    pairs = [ ]
    coord_dict = convert_fullpath_to_dict(all_coord)
    image_dict = convert_fullpath_to_dict(all_image)
    image_keys = sorted(image_dict.keys())
    coord_matched = dict()
    image_matched = dict()
    for image_base in image_keys:
        coord_base = string.replace(image_base, '.tif', '.txt')
        if coord_base in coord_dict:
            pairs.append( (coord_dict[coord_base], image_dict[image_base] ) )
            coord_matched[ coord_dict[coord_base] ] = True
            image_matched[ image_dict[image_base] ] = True
    return(pairs, coord_matched, image_matched)

def write_plots(pdf, coord_file, points, points_grid, power_scaled, power_sum, circ,
                img, img_boundary, img_fill, scalestr, insidemean, totmean,
                pixels_inside):
    logger.info('%s', coord_file)
    (root, base) = os.path.split(coord_file)
    
    plt.figure(figsize=(11,8.5))
        
    plt.subplot(231)
    # make sure the path is closed
    x = list(points[:,0])
    y = list(points[:,1])
    x.append(x[0])
    y.append(y[0])
    plt.plot(x, y, 'k', label='Boundary from file')
    plt.scatter(points_grid[:,0], points_grid[:,1],
                facecolors='none', edgecolors='b',
                label='Grid interpolation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.gca().invert_yaxis()
    #plt.legend()
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.title('%s' % base)           
                    
    plt.subplot(232)
    xx = range(2,MAXK)
    plt.plot(xx, power_scaled[xx])
    # plt.title('%s: Power Spectrum' % basename)
    plt.title('Power: sum %.3f, circ %.3f' % (power_sum, circ))
    (x1,x2,y1,y2) = plt.axis()
    y2 = max(y2, 0.1)
    plt.axis((x1,x2,y1,y2))
    plt.xlabel('Harmonic component')
    plt.ylabel('Scaled power')
    
    plt.subplot(233)
    plt.hist(pixels_inside.ravel(), normed=True, alpha=0.5, label='inside')
    plt.hist(img.ravel(), normed=True, alpha=0.5, label='entire image')
    plt.legend(loc='best', fancybox=True, framealpha=0.5)
    plt.title('Pixel Intensity')
    
    plt.subplot(234)
    plt.imshow(img, cmap=mpcm.gray)
    plt.title('Mean: inside %.1f, entire %.1f' % (insidemean, totmean) )
               
    plt.subplot(235)
    plt.imshow(img_boundary, cmap=mpcm.gray)
    plt.title('Boundary, Scale = ' + scalestr)
              
    plt.subplot(236)
    plt.imshow(img_fill, cmap=mpcm.gray)
    plt.title('Fill, Scale = ' + scalestr)

    #plt.subplot(326)
    #plt.imshow(img100, cmap=mpcm.gray)
    #plt.title('Scale = 100')

    pdf.savefig()
    plt.close()
    
    return(None)

def write_image_thumbnails(outdir, imagedir, results_dict, points_dict):

    # image shape is (1040, 1388)
    # image width : height is 1388 : 1040
    # the apect ratio is 1.335, call it 4/3 = standard screen aspect ratio
    # suppose with is w, then h is 0.75 w
    # 0.75 w^2 = # of images
    # w = sqrt(# of images * 4/3) = sqrt(2000 * 4 / 3) = 56
    # h = 42
    # 56 x 42 = 2352
    # for thumbnail, want 1388 / 56 = 24 pixels width
    # so 24 x 18 would be about right
    # or 32 x 24
    
    HEIGHT_OVER_WIDTH = 3.0 / 4.0
    (THUMB_HEIGHT, THUMB_WIDTH) = (192, 256)
    # (THUMB_HEIGHT, THUMB_WIDTH) = (96, 128)
    # (THUMB_HEIGHT, THUMB_WIDTH) = (384, 512)
    # (THUMB_HEIGHT, THUMB_WIDTH) = (24, 32)
    # (THUMB_HEIGHT, THUMB_WIDTH) = (18, 24)
    # (THUMB_HEIGHT, THUMB_WIDTH) = (15, 20)

    # get the coordinate filenames
    coordfiles = sorted( results_dict.keys() )
    # create the image filenames
    imagefiles = [ ]
    for c in coordfiles:
        (root, base) = os.path.split(c)
        toks = base.split('.')
        assert(toks[-1] == 'txt'), 'coordinate file does not end in .txt: %s' % c
        toks[-1] = '.tif'
        newbase = ''.join(toks)
        imagefile = os.path.join(imagedir, newbase)
        imagefiles.append(imagefile)
        if not os.path.isfile(imagefile):
            logger.warn('image file missing: %s', imagefile)
            continue
    pairs = zip(coordfiles, imagefiles)

    npair = len(pairs)
    nside = int(math.ceil(math.sqrt(npair)))
    tot_width = nside * THUMB_WIDTH
    tot_height = nside * THUMB_HEIGHT
    logger.info('npair %d nside %d spaces %d width %d height %d', npair, nside, nside*nside, tot_width, tot_height)
    thumbnails = np.zeros((tot_height, tot_width))
    #thumbnails_mean = np.zeros((tot_height, tot_width))
    #thumbnails_med = np.zeros((tot_height, tot_width))
    thumbnails_max = np.zeros((tot_height, tot_width))
    logger.info('thumbnails type %s shape %s', type(thumbnails), str(thumbnails.shape))
    
    cnt = 0
    for (coord_file, image_file) in pairs:
        img = mpimg.imread(image_file)
        assert( (len(img.shape) >= 2) and (len(img.shape) <= 3) ), 'bad shape'
        img_mode = 'GRAY'
        if (len(img.shape) == 3):
            img_mode = 'RGB'
            img_rgb = np.copy(img)
            img = rgb2gray(img_rgb)
        img_resize = sp.misc.imresize(img, (THUMB_HEIGHT, THUMB_WIDTH))
        mymean = np.mean(img_resize.ravel())
        mymed = np.median(img_resize.ravel())
        mymax = np.max(img_resize.ravel())
        logger.info('image %s mean %f med %f max %f', image_file, mymean, mymed, mymax)
        #if (mystd > 0.0):
        #    myz = (img_resize - mymean)/mystd
        #    img_resize = 10.0 * np.divide(1.0 , 1.0 + np.exp(-myz) )

        #if (mymean > 0.0):
        #    img_mean = (50.0 / mymean) * img_resize
        #else:
        #    img_mean = img_resize

        #if (mymed > 0.0):            
        #    img_med = (10.0 / mymed) * img_resize
        #else:
        #    img_med = img_resize

        img_max = img_resize.copy()
        img_max[ img_max <= mymed ] = 0.0
        # if (mymax > 20.0) or (mymax > 1.5 * mymed):
        if (mymax > 0.0):
            img_max = (100.0 / mymax) * img_max
            
        #plt.figure()
        #plt.imshow(img, cmap=mpcm.gray)
        #plt.savefig(os.path.join(outdir, '%02d_orig.pdf' % cnt))
        #plt.close()
        #plt.figure()
        #plt.imshow(img_resize, cmap=mpcm.gray)
        #plt.savefig(os.path.join(outdir, '%02d_resize.pdf' % cnt))
        #plt.close()
        
        column = cnt % nside
        row = cnt // nside
        (rowstart, rowend) = (row * THUMB_HEIGHT, (row+1) * THUMB_HEIGHT)
        (colstart, colend) = (column * THUMB_WIDTH, (column+1) * THUMB_WIDTH)
        thumbnails[rowstart:rowend,colstart:colend] = img_resize
        #thumbnails_mean[rowstart:rowend,colstart:colend] = img_mean
        #thumbnails_med[rowstart:rowend,colstart:colend] = img_med
        thumbnails_max[rowstart:rowend,colstart:colend] = img_max
        
        cnt = cnt + 1

#    for (myscale, myimg) in zip( ['original', 'mean', 'median', 'maximum'],
#        [thumbnails, thumbnails_mean, thumbnails_med, thumbnails_max]):
    for (myscale, myimg) in zip( ['original', 'maximum'],
        [thumbnails, thumbnails_max]):
        plt.figure(figsize=(64,48))
        # plt.figure(figsize=(32,24))
        plt.imshow(myimg)
        plt.title('Thumbnails, %s' % myscale)
        plt.axis('off')
        filename = 'thumbnails_images_%dx%d_%s.pdf' % (THUMB_HEIGHT, THUMB_WIDTH, myscale)
        plt.savefig(os.path.join(outdir, filename))
        plt.close()

    return(None)

def write_boundary_thumbnails(outdir, results_dict, points_dict):

    diam = 3.0 # typical diameter for an organoid scaled to unit circle
    HEIGHT_OVER_WIDTH = 3.0 / 4.0
    FIGSIZE = (12, 9)
    
    # order by increasing power
    byname = sorted(results_dict.keys())
    tups = [ (float(results_dict[k]['power']), k) for k in byname ]
    tups = sorted(tups)
    bypower = [ t[1] for t in tups ]
    
    ntot = len(byname)
    colors = mpcm.rainbow(np.linspace(0,1,ntot))
    name2color = dict(zip(bypower, colors))

    nside = int(math.ceil(math.sqrt(ntot)))
    dh = diam
    dw = diam / HEIGHT_OVER_WIDTH

    #orders = [ bypower ] 
    #titles = [ 'Organoids by Spectral Power' ] 
    #filenames = [ 'boundary_thumbnails_bypower.pdf' ]
    orders = [ byname ]
    titles = ['Boundaries, All']
    filenames = [ 'thumbnails_boundaries_all.pdf']
    edgecolors = [ 'none' ]
    
    for (myorder, mytitle, myfilename, myedgecolor) in zip(orders, titles, filenames, edgecolors):
        
        plt.figure(figsize=FIGSIZE)
        plt.title(mytitle)
        plt.gca().set_aspect('equal')
        axes = plt.gca()
        cnt = 0
        for (k) in myorder:
            pts = points_dict[k]
            row = cnt // nside
            col = cnt % nside
            dx = col * dw
            dy = row * dh
            #(x0, y0) = close_polygon(pts)
            #x1 = [ x + dx for x in x0 ]
            #y1 = [ y + dy for y in y0 ]
            # logger.info('r %f c %f clr %s', float(r), float(c), str(clr))
            # plt.plot(x1, y1, color=clr)
            newx = pts[:,0] + dx
            newy = pts[:,1] + dy
            xy = zip(newx, newy)
            axes.add_patch(Polygon(xy, closed=True, facecolor=name2color[k], edgecolor=myedgecolor) )
            cnt = cnt + 1
        
        axes.autoscale_view()    
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.savefig(os.path.join(outdir, myfilename))
        plt.close()
        
    #for annot in ANNOTS:
    #    for day in DAYNUMS:
    #        title = 'Boundaries, %s Day %s' % (annot, day)
    #        filename = 'thumbnails_boundaries_%s_%s.pdf' % (annot, day)
    #        plt.figure(figsize=FIGSIZE)
    #        plt.title(title)
    #        plt.gca().set_aspect('equal')
    #        axes = plt.gca()
    #        cnt = 0
    #        for (k) in byname:
    #            pts = points_dict[k]
    #            row = cnt // nside
    #            col = cnt % nside
    #            dx = col * dw
    #            dy = row * dh
    #            newx = pts[:,0] + dx
    #            newy = pts[:,1] + dy
    #            xy = zip(newx, newy)
    #            alpha = 0.2
    #            fc = 'gray'
    #            #if (results_dict[k]['annot'] == annot) and (results_dict[k]['daynum'] == day):
    #            #    alpha = 1.0
    #            #    fc = name2color[k]   
    #            axes.add_patch(Polygon(xy, closed=True, facecolor=fc, edgecolor='none', alpha=alpha) )
    #            cnt = cnt + 1
    #    
    #        axes.autoscale_view()    
    #        plt.gca().invert_yaxis()
    #        plt.axis('off')
    #        plt.savefig(os.path.join(outdir, filename))
    #        plt.close()            
    return None 

def get_gray(mystr):
    myval = 0.01 * float(mystr)
    myret = None
    if (myval <= 0.0):
        myret = '0.0'
    elif (myval <= 1.0):
        myret = str(myval)
    else:
        myret = '1.0'
    return(myret)

def write_thermometers_old(outdir, results_dict, points_dict):

    diam = 3.0 # typical diameter for an organoid scaled to unit circle
    
    # unit circle
    thetas = np.linspace(0.0, 2.0 * math.pi, num=256, endpoint=False)
    xcircle = [ math.cos(t) for t in thetas ]
    ycircle = [ math.sin(t) for t in thetas ]
    
    #for a in ANNOTS:
    #    for d in DAYNUMS:
    
    for a in ['Tumor']:
        for d in ['6']:
            # subset = [ k for k in results_dict.keys()
            #          if ((results_dict[k]['annot'] == a) and (results_dict[k]['daynum'] == d))]
            subset = [ k for k in results_dict.keys() if (k in points_dict) ]
            nsubset = len(subset)
            logger.info('annot %s day %s cnt %d', a, d, nsubset)
            if (nsubset == 0):
                continue
                                           
            # order by increasing power
            tups = [ (float(results_dict[k]['invasion_spectral']), k) for k in subset ]
            #logger.info('tups %s', str(tups[:5]))
            tups = sorted(tups)
            #logger.info('sorted %s', str(tups[:5]))
            bypower = [ t[1] for t in tups ]
            colors = mpcm.rainbow(np.linspace(0,1,nsubset))
            key2color = dict()
            for (k, c) in zip(bypower, colors):
                key2color[k] = c
                
            # group power by individual
            ctn2powers = dict()
            ctn2pxls = dict()
            ctn2num = dict()
            ctn2rgb = dict()
            ctn2gray = dict()
            for k in subset:
                ctn = results_dict[k]['ctn']
                ctn2powers[ctn] = ctn2powers.get(ctn,[ ]) + [ float(results_dict[k]['invasion_spectral'])]
                ctn2pxls[ctn] = ctn2pxls.get(ctn,[ ]) + [ float(results_dict[k]['k14_mean'])]
                ctn2num[ctn] = ctn2num.get(ctn, 0) + 1
                imgmode = results_dict[k]['tag_color']
                if (imgmode == 'rgb'):
                    ctn2rgb[ctn] = ctn2rgb.get(ctn, 0) + 1
                elif (imgmode == 'gray'):
                    ctn2gray[ctn] = ctn2gray.get(ctn, 0) + 1

            ctn2meanpower = dict()
            ctn2stdpower = dict()
            for (c,val) in ctn2powers.iteritems():
                ctn2meanpower[c] = np.mean(val)
                ctn2stdpower[c] = np.std(val, ddof=1)

            ctn2meanpxl = dict()
            ctn2stdpxl = dict()
            for (c, v) in ctn2pxls.iteritems():
                ctn2meanpxl[c] = np.mean(v)
                ctn2stdpxl[c] = np.std(v, ddof=1)

            tups = sorted([ (v, k) for (k, v) in ctn2meanpower.iteritems() ])
            # logger.info('ctns sorted: %s', str(tups))
            ctns = [ x[1] for x in tups ]


            # fill based on color from power
            title = 'Organoids from %s Day %s' % (a, d)
            filename = 'thermometer_%s_%s.pdf' % (a, d)
            plt.figure()
            plt.title(title)
            plt.gca().set_aspect('equal')
            axes = plt.gca()
            mycolumn = 0
            for ctn in ctns:
                myrow = 0
                for k in bypower:
                    if (results_dict[k]['ctn'] != ctn):
                        continue
                    if k not in points_dict:
                        continue
                    clr = key2color[k]
                    dx = mycolumn * diam
                    dy = myrow * diam
                    pts = points_dict[k]
                    newx = pts[:,0] + dx
                    newy = pts[:,1] + dy
                    xy = zip(newx, newy)
                    axes.add_patch(Polygon(xy, closed=True, facecolor=clr, edgecolor='none') )
                    myrow = myrow + 1
                mycolumn = mycolumn + 1
            axes.autoscale_view()    
            # plt.gca().invert_yaxis()
            plt.axis('off')
            plt.savefig(os.path.join(outdir, filename))
            plt.close()
            
            continue
            
            # fill based on mean pixel intensity
            title = 'Organoids from %s Day %s' % (a, d)
            filename = 'graymometer_%s_%s.pdf' % (a, d)
            plt.figure(figsize=(40,30))
            plt.title(title)
            plt.gca().set_aspect('equal')
            axes = plt.gca()
            mycolumn = 0
            for ctn in ctns:

                myrow = 0
                eclist = [ ]
                for k in bypower:
                    if (results_dict[k]['ctn'] != ctn):
                        continue
                    if k not in points_dict:
                        continue
                    edgeclr = key2color[k]
                    faceclr = get_gray(results_dict[k]['k14_sum'])
                    dx = mycolumn * diam
                    dy = myrow * diam
                    pts = points_dict[k]
                    # this plots a boundary
                    #(x0, y0) = close_polygon(pts)
                    #x1 = [ x + dx for x in x0 ]
                    #y1 = [ y + dy for y in y0 ]
                    # plt.plot(x1, y1, color=clr)
                    newx = pts[:,0] + dx
                    newy = pts[:,1] + dy
                    xy = zip(newx, newy)
                    axes.add_patch(Polygon(xy, closed=True, facecolor=faceclr, edgecolor=edgeclr) )
                    eclist.append(edgeclr)
                    myrow = myrow + 1
                    
                # bottom circle with overall intensity
                ctngray = get_gray(ctn2meanpxl[ctn])
                dx = mycolumn * diam
                dy = -1.0 * diam
                newx = [ x + dx for x in xcircle ]
                newy = [ y + dy for y in ycircle ]
                xy = zip(newx, newy)
                i = len(eclist) // 2
                if (i < len(eclist)):
                    axes.add_patch(Polygon(xy, closed=True, facecolor=ctngray, edgecolor=eclist[i]) )
                
                plt.text(mycolumn * diam, -2*diam, ctn, size='smaller', ha='center', va='center')
                plt.text(mycolumn * diam, -3*diam, str(ctn2rgb.get(ctn, 0)),
                         size='smaller', ha='center', va='center')
                plt.text(mycolumn * diam, -4*diam, str(ctn2gray.get(ctn,0)),
                         size='smaller', ha='center', va='center')
                    
                    
                mycolumn = mycolumn + 1
                
            axes.autoscale_view()    
            # plt.gca().invert_yaxis()
            plt.axis('off')
            plt.savefig(os.path.join(outdir, filename))
            plt.close()
            
            # plot between and within
            plt.figure(figsize=(15,5))
            
            plt.subplot(131)
            plt.title('Between-Sample, %s Day %s' % (a, d) )
            #plt.xlabel(r'$\langle$' + 'Protein Expression' + r'$\rangle')
            plt.xlabel(r'$\langle$' + 'K14 Expression' + r'$\rangle$')
            plt.ylabel(r'$\langle$' + 'Spectral Power' + r'$\rangle$')
            # plt.xscale('log')
            outliers = [ c for c in ctns if ctn2meanpxl[c] > 100 ]
            inliers = [ c for c in ctns if c not in outliers ]
            logger.info('outliers: %s', str(outliers))
            xx = [ ctn2meanpxl[c] for c in inliers ]
            yy = [ ctn2meanpower[c] for c in inliers ]
            plt.scatter(xx, yy, c='k', marker='o')
            
            # between-ctn fit
            between_fit = sm.OLS(yy,sm.add_constant(xx)).fit()
            logger.info('\n*** between fit %s %s:***\n%s', a, d, between_fit.summary())
            delta = max(xx) - min(xx)
            xx_fit = np.linspace(min(xx) - 0.1 * delta, max(xx) + 0.1*delta,100)
            logger.info('params: %s', str(between_fit.params))
            logger.info('pvalues: %s', str(between_fit.pvalues))
            plt.plot(xx_fit, xx_fit*between_fit.params[1] + between_fit.params[0], 'b')
            plt.text(max(xx), max(yy), 'p = %.3g' % between_fit.pvalues[1], horizontalalignment='right', verticalalignment='top')

            # within-ctn fit, delta
            plt.subplot(132)
            plt.title('Within-Sample, %s Day %s' % (a, d) )
            plt.xlabel(r'$\Delta$' + ' K14 Expression')
            plt.ylabel(r'$\Delta$' + ' Spectral Power')

            deltapower = [ ]
            deltapxl = [ ]
            zpower = [ ]
            zpxl = [ ]
            for k in bypower:
                c = results_dict[k]['ctn']
                if c in outliers:
                    continue
                dpower = results_dict[k]['invasion_spectral'] - ctn2meanpower[c]
                dpxl = results_dict[k]['k14_sum'] - ctn2meanpxl[c]
                deltapower.append(dpower)
                deltapxl.append(dpxl)
                if (ctn2num[c] < 5) or (ctn2stdpxl[c] == 0.0):
                    continue
                zpwr = dpower / ctn2stdpower[c]
                zpx = dpxl / ctn2stdpxl[c]
                zpower.append(zpwr)
                zpxl.append(zpx)
                
            plt.scatter(deltapxl, deltapower, c='k', marker='o')
            
            within_fit = sm.OLS(deltapower, deltapxl).fit()
            logger.info('\n*** within fit delta %s %s:***\n%s', a, d, within_fit.summary())
            (xmin, xmax) = (min(deltapxl), max(deltapxl))
            dx = 0.1 * (xmax - xmin)
            xx_fit = np.linspace(xmin - dx, xmax + dx, 100)
            logger.info('params: %s', str(within_fit.params))
            logger.info('pvalues: %s', str(within_fit.pvalues))
            plt.plot(xx_fit, xx_fit*within_fit.params[0], 'b')
            plt.text(xmax, max(deltapower), 'p = %.3g' % within_fit.pvalues[0], horizontalalignment='right', verticalalignment='top')
            
            # within-ctn fit, zscore
            plt.subplot(133)
            plt.title('Within-Sample, %s Day %s' % (a, d) )
            plt.xlabel('z-score, K14 Expression')
            plt.ylabel('z-score, Spectral Power')

            plt.scatter(zpxl, zpower, c='k', marker='o')
            
            within_fit = sm.OLS(zpower, zpxl).fit()
            logger.info('\n*** within fit zscore %s %s:***\n%s', a, d, within_fit.summary())
            (xmin, xmax) = (min(zpxl), max(zpxl))
            dx = 0.1 * (xmax - xmin)
            xx_fit = np.linspace(xmin - dx, xmax + dx, 100)
            logger.info('params: %s', str(within_fit.params))
            logger.info('pvalues: %s', str(within_fit.pvalues))
            plt.plot(xx_fit, xx_fit*within_fit.params[0], 'b')
            plt.text(xmax, max(zpower), 'p = %.3g' % within_fit.pvalues[0], horizontalalignment='right', verticalalignment='top')
      
      
            filename = 'heterogeneity_%s_%s.pdf' % (a, d)
            plt.savefig(os.path.join(outdir, 'FIGURES', filename))
            plt.close()

def write_thermometers(filename, organoid_list, group_list, invasion_list, points_dict):

    diam = 3.0 # typical diameter for an organoid scaled to unit circle
    
    # unit circle
    #thetas = np.linspace(0.0, 2.0 * math.pi, num=256, endpoint=False)
    #xcircle = [ math.cos(t) for t in thetas ]
    #ycircle = [ math.sin(t) for t in thetas ]
    
    ntot = len(organoid_list)
    
    k2g = dict()
    for (k, g) in zip(organoid_list, group_list):
        k2g[k] = g
                                           
    # order by increasing power
    tups = zip(invasion_list, organoid_list)
    #logger.info('tups %s', str(tups[:5]))
    tups = sorted(tups)
    #logger.info('sorted %s', str(tups[:5]))
    org_order = [ t[1] for t in tups ]
    colors = mpcm.rainbow(np.linspace(0,1,ntot))
    key2color = dict()
    for (k, c) in zip(org_order, colors):
        key2color[k] = c
                
    # group power by individual
    group2values = dict()
    for (g,v) in zip(group_list, invasion_list):
        group2values[g] = group2values.get(g,[ ]) + [ v ]

    group2mean = dict()
    for g in group2values.keys():
        group2mean[g] = np.mean(group2values[g])

    group_tups = [ (group2mean[g], g) for g in group2mean.keys() ]
    group_tups = sorted(group_tups)
    # logger.info('ctns sorted: %s', str(tups))
    groups = [ g for (m, g) in group_tups ]

    # fill based on color from power
    plt.figure(figsize=(10,10))
    # plt.title(title)
    plt.gca().set_aspect('equal')
    axes = plt.gca()
    mycolumn = 0
    for g in groups:
        shortname = str(int(g[3:]))
        axes.text(mycolumn*diam, -2*diam, shortname, horizontalalignment='center', fontsize=5)
        myrow = 0
        for k in org_order:
            if (k2g[k] != g):
                continue
            if k not in points_dict:
                continue
            clr = key2color[k]
            dx = mycolumn * diam
            dy = myrow * diam
            pts = points_dict[k]
            newx = pts[:,0] + dx
            newy = pts[:,1] + dy
            xy = zip(newx, newy)
            axes.add_patch(Polygon(xy, closed=True, facecolor=clr, edgecolor='none') )
            myrow = myrow + 1
        mycolumn = mycolumn + 1
    axes.autoscale_view()    
    # plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

def write_results(outdir, results_dict):
    samples = sorted(results_dict.keys())
    fields = dict()
    for s in samples:
        for f in results_dict[s].keys():
            fields[f] = fields.get(f, 0) + 1
    fields_list = sorted(fields.keys())
    logger.info('table fields: %s', str(fields_list))
    fp = open(os.path.join(outdir, 'results_table.txt'), 'w')
    fp.write('\t'.join(['filename'] + fields_list) + '\n')
    for s in samples:
        toks = [s] + [ str(results_dict[s][v]) for v in fields_list ]
        fp.write('\t'.join(toks) + '\n')
    fp.close()

def calculate(args):

    all_coords = get_files_recursive(args.coords, '*.txt')
    PIXEL_FIELDS = ['cnt', 'mean', 'median', 'sd', 'mad']
    
    doImages = args.images is not None
    logger.info('doImages = %s', str(doImages))
    all_images = [ ]
    if doImages:
        all_images = get_files_recursive(args.images, '*.tif')

    #logger.info('coordinate files: %s', str(all_coords))
    #logger.info('image files: %s', str(all_images))
    
    tagfp = open(os.path.join(args.outdir, 'exiftags.txt'),'w')

    coord_image_pairs = [ (c, None) for c in all_coords ]
    if doImages:
        (coord_image_pairs, coord_matched, image_matched) = match_coord_image(all_coords, all_images)
        fp = open(os.path.join(args.outdir, 'matches.txt'), 'w')
        for (c, i) in coord_image_pairs:
            fp.write('PAIR\t%s\t%s\n' % (c, i))
        for c in sorted(all_coords):
            if c not in coord_matched:
                fp.write('UNMATCHED\t%s\n' % c)
        for i in sorted(all_images):
            if i not in image_matched:
                fp.write('UNMATCHED\t%s\n' % i)
        fp.close()
    
    #logger.info('coord image pairs: %s', str(coord_image_pairs))
    logger.info('%d coordinate file, %d image files, %d pairs',
                len(all_coords), len(all_images), len(coord_image_pairs))

    results_dict = dict()
    points_dict = dict()
    power_dict = dict()
    rfft_dict = dict()
    
    if doImages:
        plotfile = os.path.join(args.outdir, 'plots.pdf')
        pdf = PdfPages(plotfile)
    
    boundaryfile = os.path.join(args.outdir, 'boundaries.pdf')
    boundarypdf = PdfPages(boundaryfile)
    
    for (coord_file, image_file) in coord_image_pairs:
        
        assert(coord_file not in results_dict), 'repeated file: ' + coord_file
        mykey = coord_file
        results_dict[mykey] = dict()
        
        # read the xy pairs from the input file
        # format is an array of tuples
        points = file_to_points(coord_file, args.dim)
        contour_length = get_contour_length(points)
        # make a regular grid
        tot_length = contour_length[-1]
        ds = tot_length / float(args.nfft)
        contour_grid = np.linspace(ds, tot_length, num=args.nfft)
        
        points_grid = get_interpolate(points, contour_length, contour_grid)
        
        area = get_area(points_grid)
        circularity = 4.0 * math.pi * area / (tot_length * tot_length)

        logger.info('ds %f tot %f %f area %f circ %f', ds, tot_length, contour_grid[-1],
                    area, circularity)
        
        points_rfft = get_rfft(points_grid)
        logger.info('hat(x) %s', str(points_rfft[:5,0]) )
        logger.info('hat(y) %s', str(points_rfft[:5,1]) )
        power_rfft = get_power(points_rfft)
        points_irfft = get_irfft(points_rfft)
        # try some normalizations
        (points_centered, points_normarea, points_normpower,
         power_normarea, power_normpower) = normalize_points(points_rfft, area)
        

        write_boundary(boundarypdf, coord_file,
                       points_grid, points_irfft,
                       points_centered, points_normarea, points_normpower,
                       power_normarea, power_normpower)

        (mysum, mysum1, mysum2) = get_power_moments(power_normpower, MAXK)
        power_sum = mysum

        results_dict[mykey]['power'] = power_sum
        results_dict[mykey]['circularity'] = circularity
        results_dict[mykey]['area'] = area
        results_dict[mykey]['circumference'] = tot_length

        points_newrfft = get_rfft(points_normpower)
        points_dict[mykey] = points_normpower
        rfft_dict[mykey] = points_newrfft
        power_dict[mykey] = power_normpower

        if doImages:        

            logger.info('image %s', image_file)
            img_base = os.path.basename(image_file)
            (img_root, img_ext) = os.path.splitext(img_base)

            tags = get_tags(image_file)
            for t in sorted(tags.keys()):
                tagfp.write('%s\t%s\t%s\n' % (image_file, t, tags[t]))

            img = mpimg.imread(image_file)
            logger.info('image shape: %s', str(img.shape))
            assert( (len(img.shape) >= 2) and (len(img.shape) <= 3) ), 'bad shape'
            #img_mode = 'GRAY'
            #if (len(img.shape) == 3):
            #    img_mode = 'RGB'
            #    img_rgb = np.copy(img)
            #    img = rgb2gray(img_rgb)
            img_mode = 'RGB'
            img = img[:,:,:3]
            
            xtag = tags.get('Image XResolution', "1")
            ytag = tags.get('Image YResolution', "1")
            xresolution = str(xtag)
            yresolution = str(ytag)
            logger.info('resolution %s %s %s %s', xtag, ytag, xresolution, yresolution)
            if (xresolution != yresolution):
                logger.warn('%s bad resolution %s %s', image_file, xresolution, yresolution)
            scalestr = str(xresolution)
            scale = eval("1.0*" + scalestr)
            #Scale is determined experimentally
            scale = 2.0
            (img_boundary, boundary_np) = create_boundary(rgb2gray(img), points_grid, scale, 'nofill')
            (img_fill, fill_np) = create_boundary(rgb2gray(img), points_grid, scale, 'fill')

            pixels = img[ fill_np > 0 ]
            pixelsummary_inside = get_pixelsummary(pixels.ravel())
            pixelsummary_tot = get_pixelsummary(img.ravel())
            logger.info('pixels inside %s %s', str(PIXEL_FIELDS),
                        str([pixelsummary_inside[pf] for pf in PIXEL_FIELDS ]))
            logger.info('pixels total %s %s', str(PIXEL_FIELDS),
                        str([pixelsummary_tot[pf] for pf in PIXEL_FIELDS ]))

            
            results_dict[mykey]['imgmode'] = img_mode
            for pf in PIXEL_FIELDS:
                results_dict[mykey]['pxlin' + pf] = pixelsummary_inside[pf]
                results_dict[mykey]['pxltot' + pf] = pixelsummary_tot[pf]
            results_dict[mykey]['scalex'] = xresolution
            results_dict[mykey]['scaley'] = yresolution
            results_dict[mykey]['scaleval'] = scale
        
            write_plots(pdf, coord_file, points, points_grid, power_normpower, power_sum, circularity,
                        img, img_boundary, img_fill, scalestr, pixelsummary_inside['mean'], pixelsummary_tot['mean'],
                        pixels)
        
    if doImages:
        pdf.close()
    boundarypdf.close()
    tagfp.close()
    logger.info('writing results')
    
    write_results(args.outdir, results_dict)
    
    return(results_dict, points_dict, power_dict, rfft_dict)


def read_results(outdir):
    results_file = os.path.join(outdir, 'results_table.txt')
    logger.info('reading results from %s', results_file)
    fp = open(results_file, 'r')
    ret = dict()
    header = fp.readline()
    cols = header.strip().split()
    fields = cols[1:]
    ntok = len(cols)
    for line in fp:
        toks = line.strip().split()
        k = toks[0]
        assert(k not in ret), 'repeated row: %s' % k
        ret[k] = dict()
        for (k1, v1) in zip( fields, toks[1:]):
            ret[k][k1] = v1
    return(ret)

def print_stats(results):

    # cnt by annot, day
    orgdict = dict()
    for k in results:
        ptr = results[k]
        (annot, daynum, ctnnum) = (ptr['annot'], ptr['daynum'], ptr['ctnnum'])
        orgdict[(annot, daynum, ctnnum)] = orgdict.get( (annot,daynum,ctnnum), 0) + 1
    orgcnt = dict()
    ctncnt = dict()
    for k in orgdict:
        (a,d,c) = k
        orgcnt[(a,d)] = orgcnt.get((a,d),0) + orgdict[k]
        ctncnt[(a,d)] = ctncnt.get((a,d), 0) + 1
    print "\nAnnot\tDay\tCTNs\tOrganoids"
    for k in sorted(orgcnt.keys()):
        (a,d) = k
        print "%s\t%s\t%d\t%d" % (a, d, ctncnt[k], orgcnt[k] )
    print "\n"
    return None

def read_image_pair(fullpath):
    img_orig = mpimg.imread(fullpath)
    img_rgb = None
    tag_color = None
    if (img_orig.ndim == 3):
        tag_color = 'rgb'
        img_rgb = img_orig
    elif (img_orig.ndim == 2):
        tag_color = 'gray'
        myshape = ( img_orig.shape[0], img_orig.shape[1], 3 )
        img_rgb = np.zeros(myshape)
        for k in range(3):
            img_rgb[:,:,k] = img_orig[:,:]
    else:
        logger.info('bad shape: %s', str(img_orig.shape))
        img_rgb = img_orig
    ny = img_rgb.shape[1]/2
    orig_left = img_rgb[:, 0:ny, ...]
    orig_right = img_rgb[:, ny:(2*ny), ...]
    (img_orig_dic, img_orig_k14) = (orig_left, orig_right)
    (mydir, filename) = os.path.split(fullpath)

    tags = get_tags(fullpath)
    #for t in sorted(tags.keys()):
    #    logger.info('%s tag %s value %s', t, str(tags[t]))
    xtag = tags.get('Image XResolution', "1")
    ytag = tags.get('Image YResolution', "1")
    samplesperpixel = str(tags['Image SamplesPerPixel'])
    xresolution = str(xtag)
    yresolution = str(ytag)
    if (xresolution != yresolution):
        logger.warn('%s bad resolution %s %s', fullpath, xresolution, yresolution)
    #scale = float(xresolution)
    #scale = eval("1.0*" + xresolution) # sometimes scalestr is 122093/62500
    #Scale is determined by looking at the images
    #scale = 2.0
    scale = 1.95
    #logger.info('resolution x %s y %s x %s y %s scale %s scale %s', xtag, ytag, xresolution, yresolution, scalestr, str(scale))
    
    phototag = tags.get('Image PhotometricInterpretation', 'NA')
    pixelmink14 = np.amin(img_orig_k14)
    pixelmaxk14 = np.amax(img_orig_k14)
    pixelmindic = np.amin(img_orig_dic)
    pixelmaxdic = np.amax(img_orig_dic)
    #logger.info('%s phototag %s before k14 min %d max %d dic min %d max %d', fullpath, phototag, pixelmink14, pixelmaxk14, pixelmindic, pixelmaxdic)
    if (str(phototag) == '1'):
        # logger.info('inverting scale')
        img_orig_k14 = 255 - img_orig_k14
        img_orig_dic = 255 - img_orig_dic
    pixelmink14 = np.amin(img_orig_k14)
    pixelmaxk14 = np.amax(img_orig_k14)
    pixelmindic = np.amin(img_orig_dic)
    pixelmaxdic = np.amax(img_orig_dic)
    #logger.info('%s phototag %s  after k14 min %d max %d dic min %d max %d', fullpath, phototag, pixelmink14, pixelmaxk14, pixelmindic, pixelmaxdic)
    
    return(img_orig_dic, img_orig_k14, scale, str(phototag), tag_color)

def calc_inv_vs_k14(args):

    orig_dir = args.images
    xy_dir = args.coords
    nfft = args.nfft
    outdir = args.outdir
    
    filenames = os.listdir(xy_dir)
    new_table = dict()
    org_id = [f.split('.')[0] for f in filenames if not f.startswith('.')]
    for i in org_id:
        new_table[i] = dict()
    new_fields = ['invasion_spectral', 'invasion_ff',
                  'size_area', 'size_perimeter', 'size_npixel', 'size_frac',
                  'k14_mean', 'k14_mean_edge', 'k14_mean_center', 'k14_sum', 'k14_sum_edge', 'k14_sum_center',
                  'tag_photometric', 'tag_color', 'tag_scale']
    for k in new_table:
        for f in new_fields:
            new_table[k][f] = '-1'
            
    plotfile = os.path.join(outdir, outdir + '.pdf')
    pdf = PdfPages(plotfile)

    ctn_list = new_table.keys()
    ctn_list.sort(key=float)
    #ctn_list.sort()

    k_vector = np.arange(1 + (nfft/2)) # since rfft uses only half the range
    k_sq = k_vector * k_vector # element multiply
    # a reasonable smoothing function is exp(- omega^2 t^2) with t the smoothing width
    # t = 1 is nearest neighbor
    # omega = (2 pi / T) k
    # so smooth the power with exp( - (2 pi / T)^2 k^2 )
    fac = 2.0 * math.pi / float(nfft)
    facsq = fac * fac
    smooth = np.exp(-facsq * k_sq)
    
    organoids = ctn_list
    #nrow = len(ctn_list)
    ncol = 3

    ff_list = [ ]
    sumpower_list = [ ]
    sumpowerksq_list = [ ]
    k14mean_list = [ ]
    k14_edge_mean_list = [ ]
    k14_center_mean_list = [ ]
    k14sum_list = [ ]
    k14sum_edge_list = [ ]
    k14sum_center_list = [ ]
    k14veena_list = [ ]
    area_list = [ ]
    sizefrac_list = [ ]

    chuncks = [ctn_list[i:i + 10] for i in range(0, len(ctn_list), 10)]

    for c in range(0, len(chuncks)): 

        nrow = len(chuncks[c])

        plt.figure(figsize=(25, nrow * ncol))
        rownum = 0       
        #    for c in ctn_list:
        # for c in ['CTN005']:
        #for c in ctn_list[:3]:
        for k in chuncks[c]:
            # k14 begin
            logger.info('... %s', k)
            
            xy_file = os.path.join(xy_dir, k + '.txt')
            (xy_raw, xy_interp, xy_hat, tot_length, area, form_factor, power_norm) = xyfile_to_spectrum(xy_file, nfft)
            power_norm = power_norm * smooth
            power_ksq = power_norm * k_sq
            sumpower = sum(power_norm[2:])
            sumpowerksq = sum(power_ksq[2:])

            fullpath = os.path.join(orig_dir, k + '.tif')
            (img_dic, img_k14, scale, tag_photometric, tag_color) = read_image_pair(fullpath)
            
            size_area = scale * scale * area
            size_perimeter = scale * tot_length

            # extract k14 intensity from image
            (img_fill, fill_np) = create_boundary(img_k14, xy_interp, scale, 'fill')
            pixels_inside = img_k14[ fill_np > 0 ]
            pixels_outside = img_k14[ fill_np == 0 ]

            # extract k14 intensity from image for peripheral and central pixels
            (edge_mask, center_mask) = get_edge_pixles(img_k14, xy_interp, scale)
            pixels_inside_peripheral_mask = img_k14[edge_mask > 0]
            pixels_inside_central_mask = img_k14[center_mask > 0]
            k14sum_edge = np.sum(pixels_inside_peripheral_mask.ravel())
            k14sum_center = np.sum(pixels_inside_central_mask.ravel())
            
            k14sum = np.sum(pixels_inside.ravel())
            k14mean = np.mean(pixels_inside.ravel())
            k14mean_edge = np.mean(pixels_inside_peripheral_mask.ravel())
            k14mean_center = np.mean(pixels_inside_central_mask.ravel())
            ninside = len(pixels_inside.ravel())
            ntotal = len(img_k14.ravel())
            # logger.info('%s %d pixels, sum = %f, mean = %f', k, ninside, k14sum, k14mean)
            if (tag_photometric == '1'):
                k14mean = 255.0 - k14mean
                k14mean_edge = 255.0 - k14mean_edge
                k14mean_center = 255.0 - k14mean_center
                k14sum = (255.0 * ninside) - k14sum
                k14sum_edge = (255.0 * ninside) - k14sum_edge
                k14sum_center = (255.0 * ninside) - k14sum_center
                # logger.info('-> %d pixels, sum = %f, mean = %f', ninside, k14sum, k14mean)
            k14sum = k14sum / float(ntotal * 255)
            k14sum_edge = k14sum_edge / float(ntotal * 255)
            k14sum_center = k14sum_center / float(ntotal * 255)
            k14mean = k14mean / 255.0
            k14mean_edge = k14mean_edge / 255.0
            k14mean_center = k14mean_center / 255.0
            # logger.info('k14 mean %f, k14sum normalized %f for %d total pixels', k14mean, k14sum, ntotal)            

            size_npixel = ninside
            if (tag_color == 'rgb'):
                size_npixel = size_npixel / 3.0
            size_frac = float(ninside)/float(ntotal)
            
            new_table[k]['invasion_spectral'] = str(sumpowerksq)
            new_table[k]['invasion_ff'] = str(form_factor)
            new_table[k]['size_area'] = str(size_area)
            new_table[k]['size_perimeter'] = str(size_perimeter)
            new_table[k]['size_npixel'] = str(size_npixel)
            new_table[k]['size_frac'] = str(size_frac)
            new_table[k]['k14_sum'] = str(k14sum)
            new_table[k]['k14_sum_edge'] = str(k14sum_edge)
            new_table[k]['k14_sum_center'] = str(k14sum_center)
            new_table[k]['k14_mean'] = str(k14mean)
            new_table[k]['k14_mean_edge'] = str(k14mean_edge)
            new_table[k]['k14_mean_center'] = str(k14mean_center)
            new_table[k]['tag_photometric'] = tag_photometric
            new_table[k]['tag_color'] = tag_color
            new_table[k]['tag_scale'] = str(scale)

            ff_list.append(form_factor)
            sumpower_list.append(sumpower)
            sumpowerksq_list.append(sumpowerksq)
            k14mean_list.append(k14mean)
            k14_edge_mean_list.append(k14mean_edge)
            k14_center_mean_list.append(k14mean_center)
            k14sum_list.append(k14sum)
            k14sum_edge_list.append(k14sum_edge)
            k14sum_center_list.append(k14sum_center)
            area_list.append(size_area)
            sizefrac_list.append(size_frac)
            
            # all the plots
            rownum += 1
            i = (rownum - 1) * ncol # plot sequence number starting point

            # K14 image
            i += 1
            plt.subplot(nrow, ncol, i)
            plt.imshow(img_k14)
            plt.title('K14 ')
            plt.ylabel('%s' % k)

            # DIC image
            #i += 1
            #plt.subplot(nrow, ncol, i)
            #plt.title('DIC')
            #plt.imshow(img_dic)

            # boundary from raw points
            i += 1
            ax = plt.subplot(nrow, ncol, i)
            # make sure the path is closed
            x = list(xy_raw[:,0])
            y = list(xy_raw[:,1])
            x.append(x[0])
            y.append(y[0])
            xx = [ scale * val for val in x ]
            yy = [ scale * val for val in y]
            plt.plot(xx, yy, 'k', label='Boundary from file')
            #plt.scatter(scale * xy_interp[:,0], scale * xy_interp[:,1], facecolors='none', edgecolors='b')
            # ax.axis('equal')
            plt.imshow(img_dic, alpha=0.5)
            ax.set_xlim(0, img_dic.shape[1] - 1)
            ax.set_ylim(0, img_dic.shape[0] - 1)
            #ax.set_aspect('auto')
            plt.title('Form Factor %.3f' % (form_factor))
            plt.gca().invert_yaxis()
            
            # power spectrum
            i += 1
            plt.subplot(nrow, ncol, i)
            x = range(len(power_norm))
            # logger.info('len(x) %d len(pwr) %d', len(x), len(power_norm))
            # plt.plot(x[2:], power_norm[2:], 'k')
            plt.plot(x[2:], power_ksq[2:], 'b')
            # plt.xlabel('Frequency')
            plt.ylabel('Power k^2')
            plt.title('Spectral %.3f' % (sumpowerksq))
            #plt.subplots_adjust(top=0.7, hspace=.5)

        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
    plt.figure(figsize=(25,8))

    for (sp, xlist, xname) in zip( (151, 152, 153, 154),
        (sizefrac_list, k14sum_list, k14sum_edge_list, k14sum_center_list),
        ('Fractional Area', 'K14 Sum', 'K14 Sum Peripheral Pixels', 'K14 Sum Centeral Pixels') ):
    
        plt.subplot(sp)
        plt.xlim(0,max(xlist))
        plt.ylim(0,max(sumpowerksq_list))
        for (x1, y1, a1) in zip(xlist, sumpowerksq_list, ctn_list):
            plt.text(x1, y1, a1, va='center', ha='center')
        plt.xlabel(xname)
        plt.ylabel('Invasion')
        (r, pval) = pearsonr(xlist, sumpowerksq_list)
        rsq = r*r
        plt.title('Rsq = %.3f, pval = %.3g' % (rsq, pval))

    plt.tight_layout()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(25,8))

    for (sp, xlist, xname) in zip( (151, 152, 153),
        (k14mean_list, k14_edge_mean_list, k14_center_mean_list),
        ('K14 Mean', 'K14 Peripheral Mean', 'K14 Central Mean') ):
    
        plt.subplot(sp)
        plt.xlim(0,max(xlist))
        plt.ylim(0,max(sumpowerksq_list))
        for (x1, y1, a1) in zip(xlist, sumpowerksq_list, ctn_list):
            plt.text(x1, y1, a1, va='center', ha='center')
        plt.xlabel(xname)
        plt.ylabel('Invasion')
        (r, pval) = pearsonr(xlist, sumpowerksq_list)
        rsq = r*r
        plt.title('Rsq = %.3f, pval = %.3g' % (rsq, pval))

    plt.tight_layout()
    pdf.savefig()
    plt.close()
    pdf.close()             

    textFile = open(os.path.join(outdir, outdir + '.txt'), 'w')

    textFile.write('ID\tInvasion\tFractional Area\tK14 Sum Peripheral Pixels\tK14 Sum Central Pixels\tK14 Total Sum\tK14 Total Mean\tK14 Peripheral Mean\tK14 Central Mean\n')

    for k in ctn_list:
        textFile.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" %(k, new_table[k]['invasion_spectral'], new_table[k]['size_frac'], new_table[k]['k14_sum_edge'], new_table[k]['k14_sum_center'], new_table[k]['k14_sum'], new_table[k]['k14_mean'], new_table[k]['k14_mean_edge'], new_table[k]['k14_mean_center']))

    textFile.close()
    
    return(new_table)    


def cluster(args):
    #K Means initializer
    num_clusters = 6
    kms = KMeans(n_clusters=num_clusters, random_state=0)
    #boundary scale
    scale = 150

    boundary = dict()
    spectral_power = dict()

    orig_dir = args.images
    xy_dir = args.coords
    nfft = args.nfft
    outdir = args.outdir
    
    filenames = os.listdir(xy_dir)
    org_id = [f.split('.')[0] for f in filenames if not f.startswith('.')]

    k_vector = np.arange(1 + (nfft/2)) # since rfft uses only half the range
    k_sq = k_vector * k_vector # element multiply

    fac = 2.0 * math.pi / float(nfft)
    facsq = fac * fac
    smooth = np.exp(-facsq * k_sq)
    
    for k in org_id:
        xy_file = os.path.join(xy_dir, k + '.txt')
        (xy_raw, xy_interp, xy_hat, tot_length, area, form_factor, power_norm) = xyfile_to_spectrum(xy_file, nfft)
        power_norm = power_norm * smooth
        power_ksq = power_norm * k_sq
        sumpower = sum(power_norm[2:])
        sumpowerksq = sum(power_ksq[2:])

        boundary[k] = xy_interp
        spectral_power[k] = np.array(power_ksq[2:]).astype(np.float)

    cls = kms.fit(spectral_power.values())

    diam = 3.0 # typical diameter for an organoid scaled to unit circle
    HEIGHT_OVER_WIDTH = 3.0 / 4.0
    FIGSIZE = (12, 9)
    
    # order by class
    classes = dict()
    number_of_features = spectral_power[k].size
    for c in range(0, num_clusters):
        classes[str(c)] = [ k for k in boundary.keys() if (cls.predict(spectral_power[k].reshape(1,number_of_features)) == c)]

    byname = sorted(boundary.keys())
    tups = [ (cls.predict(spectral_power[k].reshape(1,number_of_features)), k) for k in byname ]
    tups = sorted(tups)
    byclass = [ t[1] for t in tups ]

    class_tot = len(classes.keys())
    colors = mpcm.rainbow(np.linspace(0,1,class_tot))
    class2color = dict(zip(classes.keys(), colors))

    nside = int(math.ceil(math.sqrt(class_tot)))
    dh = diam
    dw = diam / HEIGHT_OVER_WIDTH

    plt.figure(figsize=FIGSIZE)
    plt.title('Clusters')
    plt.gca().set_aspect('equal')
    axes = plt.gca()
    myclass = 0
    for c in range(0, len(classes.keys())):
        cnt = 0
        cluster_length = len(classes[str(c)])
        nside = int(math.ceil(math.sqrt(cluster_length)))
        for (k) in classes[str(c)]:
            pts = boundary[k]
            row = cnt // nside
            col = (cnt % nside) + myclass
            dx = col * dw
            dy = row * dh
            newx = pts[:,0]/scale + dx
            newy = pts[:,1]/scale + dy
            xy = zip(newx, newy)
            axes.add_patch(Polygon(xy, closed=True, facecolor=class2color[str(c)], edgecolor='none') )
            cnt = cnt + 1
        myclass = myclass + nside
    axes.autoscale_view()    
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.savefig(os.path.join(outdir, 'cluster.pdf'))
    plt.close()
    return None

def main():
    parser = argparse.ArgumentParser(description='Stack 2D images make a 3D rendering', epilog='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datafolder', help='input directory with all the data', required=True)
    parser.add_argument('--images', help='input directory with image files as .tif', required=False)
    parser.add_argument('--coords', help='input directory with coordinates as .txt files with 2 columns, no header', required=False)
    parser.add_argument('--nfft', help='number of points of FFT', type=int, default=128, required=False)
    parser.add_argument('--outdir', help='output directory', default='output', required=False)
    parser.add_argument('--calculate', help='calculate everything', choices=['y','n'], required=False, default='n')
    parser.add_argument('--invasionvsk14', help='calculate invasion vs. K14', choices=['y','n'], required=False, default='n')
    parser.add_argument('--cluster', help='cluster organoids based on their spectral power', choices=['y','n'], required=False, default='n')
    parser.add_argument('--thumbnails', help='make thumbnails', choices=['y','n'], required=False, default='n')
    parser.add_argument('--thermometers', help='make thumbnails', choices=['y','n'], required=False, default='n')
    parser.add_argument('--dim', help='dimension of data', choices=[2], type=int, required=False, default=2)
    parser.add_argument('--test', help='test to see if images and boundaries match', choices=['y','n'], required=False, default='n')
    parser.add_argument('--filenumber', help='filenumber to test', type=str, required=False, default=1)
    parser.add_argument('--combineimgs', help='combine DIC and K14 images', choices=['y','n'], required=False, default='n')
    args = parser.parse_args()


    if(args.test == 'y'):
        test(args.datafolder, args.filenumber)
        return None

    if(args.combineimgs == 'y'):
        combine_images(args.datafolder, args.outdir)
        return None


    if (not os.path.isdir(args.outdir)):
        logger.info('creating output directory %s', args.outdir)
        os.makedirs(args.outdir)
    #for subdir in (['FIGURES','IMAGES','HISTOGRAMS']):
    #    subdirpath = os.path.join(args.outdir, subdir)
    #    if (not os.path.isdir(subdirpath)):
    #        os.makedirs(subdirpath)

    if (args.cluster=='y'):
        logger.info('Clustering organoids ...')
        result_dic = cluster(args)
        return None

    if (args.invasionvsk14=='y'):
        logger.info('calculating invasion vs. K14 ...')
        result_dic = calc_inv_vs_k14(args)
        return None
    
    picklefile = os.path.join(args.outdir, 'calculate.pkl')
    prot = cPickle.HIGHEST_PROTOCOL
    if (args.calculate=='y'):
        logger.info('calculating everything ...')
        (results_dict, points_dict, power_dict, rfft_dict) = calculate(args)
        with open(picklefile, 'wb') as fp:
            logger.info('pickling to %s protocol %s', picklefile, str(prot))
            cPickle.dump(results_dict, fp, protocol=prot)
            cPickle.dump(points_dict, fp, protocol=prot)
            cPickle.dump(power_dict, fp, protocol=prot)
            cPickle.dump(rfft_dict, fp, protocol=prot)

    results = read_results(args.outdir)
    with open(picklefile, 'rb') as fp:
        logger.info('unpickling from %s', picklefile)
        results_dict = cPickle.load(fp)
        points_dict = cPickle.load(fp)
        power_dict = cPickle.load(fp)
        rfft_dict = cPickle.load(fp)
        

    if (args.thumbnails=='y'):
        logger.info('writing thumbnails')
        write_image_thumbnails(args.outdir, args.images, results_dict, points_dict)
    
    write_boundary_thumbnails(args.outdir, results_dict, points_dict)

    if (args.thermometers == 'y'):
        write_thermometers_old(args.outdir, results_dict, points_dict)
    
    # analyze_results(args.outdir, results_dict)
        
    #print_stats(results)

    return None

if __name__ == "__main__":
    main()



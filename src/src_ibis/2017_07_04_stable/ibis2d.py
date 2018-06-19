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

import numpy as np
import scipy as sp
from scipy import interpolate
import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as mpcm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as mlines
from matplotlib.patches import Polygon

import statsmodels.api as sm

from PIL import Image, ImageDraw

import logging
logging.basicConfig(format='%(levelname)s %(name)s.%(funcName)s: %(message)s')
logger = logging.getLogger('ibis2d')
logger.setLevel(logging.INFO)

MAXK = 21

ANNOTS = ['Normal', 'NAT', 'Tumor']
DAYNUMS = ['0','6']

def get_basenames(indir):
    basenames = [ ]
    for file in sorted(os.listdir(indir)):
        if fnmatch.fnmatch(file, '*.txt'):
            basename = os.path.splitext(file)[0]
            basenames.append(basename)
    logger.info('files: %s', ' '.join(basenames))
    return(basenames)

def get_files_recursive(indir, pattern_str):
    ret = [ ]
    for root, dirnames, filenames in os.walk(indir):
        for filename in fnmatch.filter(filenames, pattern_str):
            ret.append( os.path.join(root, filename) )
    return(ret)

def convert_fullpath_to_dict(fullpath_list, check_tokens=True):
    base_dict = dict()
    for my_full in fullpath_list:
        (my_root, my_base) = os.path.split(my_full)
        # check that days match
        toks = my_base.split('_')
        
        if (check_tokens):
            errors = 0
            for t in toks:
                if ('Day' in t) and (t not in my_root):
                    #logger.warn('Day does not match: %s %s', my_root, my_base)
                    errors += 1
                elif ('NAT' in t) and (t not in my_root):
                    #logger.warn('NAT does not match: %s %s', my_root, my_base)
                    errors += 1
                elif ('Normal' in t) and (t not in my_root):
                    #logger.warn('Normal does not match: %s %s', my_root, my_base)
                    errors +=1
            if (errors > 0):
                logger.warn('skipping inconsistent file %s %s', my_root, my_base)
                continue
        if my_base in base_dict:
            logger.warn('repeated basename %s fullpath %s %s', my_base, base_dict[my_base], my_full)
            continue
        base_dict[my_base] = my_full
    return(base_dict)
    
def match_coord_image(all_coord, all_image):
    pairs = [ ]
    coord_dict = convert_fullpath_to_dict(all_coord, check_tokens=False)
    image_dict = convert_fullpath_to_dict(all_image, check_tokens=False)
    image_keys = sorted(image_dict.keys())
    coord_matched = dict()
    image_matched = dict()
    for image_base in image_keys:
        coord_base = string.replace(image_base, '_K14.tif', '_xy.txt')
        if coord_base in coord_dict:
            pairs.append( (coord_dict[coord_base], image_dict[image_base] ) )
            coord_matched[ coord_dict[coord_base] ] = True
            image_matched[ image_dict[image_base] ] = True
    return(pairs, coord_matched, image_matched)

def file_to_points(infile, dim):
    logger.info('reading %dD tuples from %s', dim, infile)
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
    logger.info('shape of points: %s', str(points.shape))
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
    logger.info('contour length %f', contour_length[m-1])
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
    logger.info('power: %s ... %s', str(power[0:10]), str(power[-5:]))
    return(power)
    
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
    
def print_organoid_table(filename, big_results):
    fp = open(filename, 'w')
    fp.write('\t'.join(['fullpath','dirname','filename','sum', 'sum1', 'sum2', 'circ']) + '\n')
    for (fullname, points, points_grid, power_scaled, mysum, mysum1, mysum2, circ) in big_results:
        (dirname, filename) = os.path.split(fullname)
        fp.write('%s\t%s\t%s\t%f\t%f\t%f\t%f\n' % (fullname, dirname, filename, mysum, mysum1, mysum2, circ))
    fp.close()

def parse_filename(filepath):
    # the abbreviation is the last part of the file name
    (root, filename) = os.path.split(filepath)
    (basename, extstr) = os.path.splitext(filename)
    toks = basename.split('_')
    (ctnnum, annot, daynum, orgnum) = (toks[0][3:], toks[1], toks[2][3:], toks[3])
    assert((int(ctnnum) > 0) and (int(ctnnum) < 1000)), 'bad ctnnum: ' + filepath
    assert(annot in ['Normal','NAT','Tumor']), 'bad annot: ' + filepath
    assert(daynum in ['0','6']), 'bad daynum: ' + filepath
    assert((int(orgnum) >= 0) and (int(orgnum) < 100)), 'bad orgnum: ' + filepath
    return(ctnnum, annot, daynum, orgnum)

# scale is 1.0 / scipy.stats.norm.ppf(0.75)    
def mad(a, scale = 1.482602218505602):
    ret = np.median(np.absolute(a - np.median(a)))
    ret = ret * scale
    return(ret)

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

def plot_annot_day(outdir, results, field='power', logscale=False):
    samples = sorted(results.keys())
    for a in ANNOTS:
        for d in DAYNUMS:
            logger.info('annot %s day %s', a, d)
            ctndata = dict()
            allvals = [ ]
            for s in samples:
                if (results[s]['annot'] != a) or (results[s]['daynum'] != d):
                    continue
                ctnnum = results[s]['ctnnum']
                if ctnnum not in ctndata:
                    ctndata[ctnnum] = dict()
                val = float(results[s][field])
                if logscale:
                    val = math.log10(val)
                ctndata[ctnnum][s] = val
                allvals.append(val)
            if len(allvals) == 0:
                logger.info('... no sample, skipping')
                continue
            minval = np.min(allvals)
            #meanval = np.mean(allvals)
            #maxval = np.max(allvals)
            plt.figure()
            mystr = field
            if logscale:
                mystr = 'log10[' + field + ']'
            plt.title(' '.join([mystr, 'for', a, 'Day', d]))
            for c in ctndata:
                vals = [ ctndata[c][s] for s in ctndata[c] ]
                x = np.mean(vals)
                mymin = np.min(vals)
                xx = [x,] * len(vals)
                plt.scatter(xx, vals)
                plt.text(x,mymin,c,horizontalalignment='center',verticalalignment='top',size='xx-small')
            plt.xlabel('CTN ordered by mean ' + mystr)
            plt.ylabel(mystr + ' for each organoid')
            toks = [a,d]
            if logscale:
                toks.append('log10' + field)
            else:
                toks.append(field)
            plt.savefig( os.path.join(outdir, 'FIGURES', '_'.join(toks) + '.pdf' ) )
            plt.close()
    return None

def plot_results(outdir, results):
    samples = sorted(results.keys())
    fields = ['circularity','power','pxlincnt','pxlinmean', 'pxlinsd', 'pxlinmedian', 'pxlinmad' ]
    logger.info('plot fields: %s', str(fields))
    nfields = len(fields)
    annot2color = dict()
    myhandles = dict()
    annots = ['Normal', 'NAT', 'Tumor']
    annot_color = dict()
    colors = ['g', 'y', 'r']
    for (a, c) in zip(annots, colors):
        annot2color[a] = c
        myhandles[a] = mlines.Line2D([ ], [ ], color=c, linestyle='none', marker='o', mec='none', markersize=5, label=a)
    daynums = ['0','6']
    
    # histograms of marginal values, all data
    for f in fields:
        plt.figure()
        x = [ float(results[s][f]) for s in samples ]
        plt.hist(x, bins=100)
        plt.title(f + ' all organoids')
        plt.xlabel(f)
        plt.ylabel('Count of organoids')
        plt.savefig( os.path.join(outdir, 'FIGURES', 'hist_' + f + '_all_hist.pdf'))
        plt.close()

    # histograms of marginal values, by annot and day
    for f in fields:
        for a in annots:
            for d in daynums:
                subset = [ s for s in samples if
                          (results[s]['annot'] == a) and
                          (results[s]['daynum'] == d) ]
                x = [ float(results[s][f]) for s in subset ]
                plt.figure()
                plt.hist(x, bins=100)
                plt.title(' '.join([f, a, 'Day' + d]))
                plt.xlabel(f)
                plt.ylabel('Count of organoids')
                plt.savefig( os.path.join(outdir, 'FIGURES', '_'.join(['hist',f,a,'Day'+d])))
                plt.close()
    
    for myday in ['0', '6']:
        sample_subset = [ s for s in samples if results[s]['daynum'] == myday ]
        sample_colors = [ annot2color.get(results[s]['annot'],'c') for s in sample_subset ]
        daystr = 'Day' + myday
        for i in range(nfields):
            for j in range(nfields):
                if (i == j):
                    continue
                f1 = fields[i]
                f2 = fields[j]
                x = [ float(results[s].get(f1, 0)) for s in sample_subset ]
                y = [ float(results[s].get(f2, 0)) for s in sample_subset ]
                plt.figure()
                plt.scatter(x, y, c=sample_colors, edgecolors='none', marker='o')
                # plt.axis((min(x),max(x),min(y),max(y)))
                #for (x1, y1, a1) in zip(x, y, abbrevs):
                #    plt.text(x1, y1, a1, va='center', ha='center')
                plt.xlabel(f1)
                plt.ylabel(f2)
                plt.title(daystr)
                plt.legend(handles = [myhandles[a] for a in annots])
                plotfile = '_'.join(['fig', daystr, f1, f2]) + '.pdf'
                plotfile = os.path.join(outdir, 'FIGURES', plotfile)
                plt.savefig(plotfile)
                plt.close()

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

def write_image_data(filename, img):
    fp = open(filename, 'w')
    logger.info('writing rgb to %s', filename)
    assert(len(img.shape) == 3), 'bad shape'
    assert(img.shape[2] == 3), 'bad shape'
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            vals = [ img[i,j,k] for k in range(3) ]
            tot = sum(vals)
            toks = [ str(v) for v in [i,j] + vals + [tot] ]
            fp.write('\t'.join(toks) + '\n')
    fp.close()
    return None

def rgb2gray(img_rgb):
    assert(len(img_rgb.shape) == 3), 'bad shape'
    assert(img_rgb.shape[2] == 3), 'bad rgb dimension'
    img_gray = img_rgb[:,:,0] + img_rgb[:,:,1] + img_rgb[:,:,2]
    img_gray = img_gray / 3.0
    logger.info('rgb shape %s converted to gray shape %s',
                str(img_rgb.shape), str(img_gray.shape))
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
    maxval = np.amax(img_orig)
    img_boundary = np.copy(img_orig)
    img_boundary[ boundary_np ] = maxval
    return(img_boundary, boundary_np)

def get_pixelsummary(arr):
    ret = dict()
    ret['cnt'] = arr.shape[0]
    ret['mean'] = np.mean(arr)
    ret['median'] = np.median(arr)
    ret['sd'] = np.std(arr)
    ret['mad'] =  mad(arr)
    return(ret)

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
        toks = base.split('_')
        assert(toks[-1] == 'xy.txt'), 'coordinate file does not end in _xy.txt: %s' % c
        toks[-1] = 'K14.tif'
        newbase = '_'.join(toks)
        imagefile = os.path.join(imagedir, newbase)
        imagefiles.append(imagefile)
        if not os.path.isfile(imagefile):
            logger.warn('image file missing: %s', imagefile)
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
        
    for annot in ANNOTS:
        for day in DAYNUMS:
            title = 'Boundaries, %s Day %s' % (annot, day)
            filename = 'thumbnails_boundaries_%s_%s.pdf' % (annot, day)
            plt.figure(figsize=FIGSIZE)
            plt.title(title)
            plt.gca().set_aspect('equal')
            axes = plt.gca()
            cnt = 0
            for (k) in byname:
                pts = points_dict[k]
                row = cnt // nside
                col = cnt % nside
                dx = col * dw
                dy = row * dh
                newx = pts[:,0] + dx
                newy = pts[:,1] + dy
                xy = zip(newx, newy)
                alpha = 0.2
                fc = 'gray'
                if (results_dict[k]['annot'] == annot) and (results_dict[k]['daynum'] == day):
                    alpha = 1.0
                    fc = name2color[k]   
                axes.add_patch(Polygon(xy, closed=True, facecolor=fc, edgecolor='none', alpha=alpha) )
                cnt = cnt + 1
        
            axes.autoscale_view()    
            plt.gca().invert_yaxis()
            plt.axis('off')
            plt.savefig(os.path.join(outdir, filename))
            plt.close()            

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

def write_thermometers(outdir, results_dict, points_dict):

    diam = 3.0 # typical diameter for an organoid scaled to unit circle
    
    # unit circle
    thetas = np.linspace(0.0, 2.0 * math.pi, num=256, endpoint=False)
    xcircle = [ math.cos(t) for t in thetas ]
    ycircle = [ math.sin(t) for t in thetas ]
    
    for a in ANNOTS:
        for d in DAYNUMS:
            
            subset = [ k for k in results_dict.keys()
                      if ((results_dict[k]['annot'] == a) and (results_dict[k]['daynum'] == d))]
            nsubset = len(subset)
            logger.info('annot %s day %s cnt %d', a, d, nsubset)
            if (nsubset == 0):
                continue
                                           
            # order by increasing power
            tups = [ (float(results_dict[k]['power']), k) for k in subset ]
            #logger.info('tups %s', str(tups[:5]))
            tups = sorted(tups)
            #logger.info('sorted %s', str(tups[:5]))
            bypower = [ t[1] for t in tups ]
            colors = mpcm.rainbow(np.linspace(0,1,nsubset))
            key2color = dict()
            for (k, c) in zip(bypower, colors):
                key2color[k] = c
                
            # group power by individual
            ctn2power = dict()
            ctn2pxls = dict()
            for k in subset:
                ctn = results_dict[k]['ctnnum']
                ctn2power[ctn] = ctn2power.get(ctn,[ ]) + [ float(results_dict[k]['power'])]
                ctn2pxls[ctn] = ctn2pxls.get(ctn,[ ]) + [ float(results_dict[k]['pxlinmean'])]
            # logger.info('\nctn2power %s', str(ctn2power))
            ctn2mean = dict()
            for (c,val) in ctn2power.iteritems():
                ctn2mean[c] = np.mean(val)
            tups = sorted([ (v, k) for (k, v) in ctn2mean.iteritems() ])
            # logger.info('ctns sorted: %s', str(tups))
            ctns = [ x[1] for x in tups ]

            ctn2pxl = dict()
            for (c, v) in ctn2pxls.iteritems():
                ctn2pxl[c] = np.mean(v)

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
                    if (results_dict[k]['ctnnum'] != ctn):
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
            
            # fill based on mean pixel intensity
            title = 'Organoids from %s Day %s' % (a, d)
            filename = 'graymometer_%s_%s.pdf' % (a, d)
            plt.figure()
            plt.title(title)
            plt.gca().set_aspect('equal')
            axes = plt.gca()
            mycolumn = 0
            for ctn in ctns:

                myrow = 0
                eclist = [ ]
                for k in bypower:
                    if (results_dict[k]['ctnnum'] != ctn):
                        continue
                    edgeclr = key2color[k]
                    faceclr = get_gray(results_dict[k]['pxlinmean'])
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
                ctngray = get_gray(ctn2pxl[ctn])
                # logger.info('ctn %s pxlmean %.2f gray %s', ctn, ctn2pxl[ctn], ctngray)
                dx = mycolumn * diam
                dy = -1.0 * diam
                newx = [ x + dx for x in xcircle ]
                newy = [ y + dy for y in ycircle ]
                xy = zip(newx, newy)
                i = len(eclist) // 2
                axes.add_patch(Polygon(xy, closed=True, facecolor=ctngray, edgecolor=eclist[i]) )
                    
                    
                mycolumn = mycolumn + 1
            axes.autoscale_view()    
            # plt.gca().invert_yaxis()
            plt.axis('off')
            plt.savefig(os.path.join(outdir, filename))
            plt.close()
            
            # plot between and within
            plt.figure(figsize=(12,5))
            
            plt.subplot(121)
            plt.title('Between-Sample, %s Day %s' % (a, d) )
            #plt.xlabel(r'$\langle$' + 'Protein Expression' + r'$\rangle')
            plt.xlabel(r'$\langle$' + 'K14 Expression' + r'$\rangle$')
            plt.ylabel(r'$\langle$' + 'Spectral Power' + r'$\rangle$')
            # plt.xscale('log')
            outliers = [ c for c in ctns if ctn2pxl[c] > 100 ]
            inliers = [ c for c in ctns if c not in outliers ]
            logger.info('outliers: %s', str(outliers))
            #for c in outliers:
            #    plt.text(ctn2mean[c],ctn2pxl[c],c,horizontalalignment='center',verticalalignment='top')
            xx = [ ctn2pxl[c] for c in inliers ]
            yy = [ ctn2mean[c] for c in inliers ]
            plt.scatter(xx, yy, c='k', marker='o')
            
            # now for the fit
            between_fit = sm.OLS(yy,sm.add_constant(xx)).fit()
            logger.info('\n*** between fit %s %s:***\n%s', a, d, between_fit.summary())
            delta = max(xx) - min(xx)
            xx_fit = np.linspace(min(xx) - 0.1 * delta,max(xx) + 0.1*delta,100)
            logger.info('params: %s', str(between_fit.params))
            logger.info('pvalues: %s', str(between_fit.pvalues))
            plt.plot(xx_fit, xx_fit*between_fit.params[1] + between_fit.params[0], 'b')
            plt.text(max(xx), max(yy), 'p = %.3g' % between_fit.pvalues[1], horizontalalignment='right', verticalalignment='top')

            plt.subplot(122)
            plt.title('Within-Sample, %s Day %s' % (a, d) )
            plt.xlabel(r'$\Delta$' + ' K14 Expression')
            plt.ylabel(r'$\Delta$' + ' Spectral Power')

            deltapower = [ ]
            deltapxl = [ ]
            for k in bypower:
                c = results_dict[k]['ctnnum']
                if c in outliers:
                    continue
                deltapower.append(results_dict[k]['power'] - ctn2mean[c])
                deltapxl.append(results_dict[k]['pxlinmean'] - ctn2pxl[c])
                
            # collect by ctn and calculate z-scores

            plt.scatter(deltapxl, deltapower, c='k', marker='o')
            
            # now for the fit
            within_fit = sm.OLS(deltapower, deltapxl).fit()
            logger.info('\n*** within fit %s %s:***\n%s', a, d, within_fit.summary())
            (xmin, xmax) = (min(deltapxl), max(deltapxl))
            dx = 0.1 * (xmax - xmin)
            xx_fit = np.linspace(xmin - dx, xmax + dx, 100)
            logger.info('params: %s', str(within_fit.params))
            logger.info('pvalues: %s', str(within_fit.pvalues))
            plt.plot(xx_fit, xx_fit*within_fit.params[0], 'b')
            plt.text(xmax, max(deltapower), 'p = %.3g' % within_fit.pvalues[0], horizontalalignment='right', verticalalignment='top')

            filename = 'heterogeneity_%s_%s.pdf' % (a, d)
            plt.savefig(os.path.join(outdir, 'FIGURES', filename))
            plt.close()


def recalculate(args):

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
        (ctnnum, annot, daynum, orgnum) = parse_filename(coord_file)
        for (k, v) in [ ('annot', annot), ('ctnnum', ctnnum), ('daynum', daynum),
            ('orgnum', orgnum)]:
            results_dict[mykey][k] = v
        
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
            img_mode = 'GRAY'
            if (len(img.shape) == 3):
                img_mode = 'RGB'
                img_rgb = np.copy(img)
                img = rgb2gray(img_rgb)
            
            xtag = tags.get('Image XResolution', "1")
            ytag = tags.get('Image YResolution', "1")
            xresolution = str(xtag)
            yresolution = str(ytag)
            logger.info('resolution %s %s %s %s', xtag, ytag, xresolution, yresolution)
            if (xresolution != yresolution):
                logger.warn('%s bad resolution %s %s', image_file, xresolution, yresolution)
            scalestr = str(xresolution)
            scale = eval("1.0*" + scalestr)
            (img_boundary, boundary_np) = create_boundary(img, points_grid, scale, 'nofill')
            (img_fill, fill_np) = create_boundary(img, points_grid, scale, 'fill')

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

def main():
    
    parser = argparse.ArgumentParser(description='Invasive boundary image score, 2D',
                                     epilog='Sample call: see run.sh',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--coords', help='input directory with coordinates as .txt files with 2 columns, no header', required=True)
    parser.add_argument('--images', help='input directory with image files as .tif', required=False)
    parser.add_argument('--outdir', help='output directory', required=True)
    parser.add_argument('--nfft', help='number of points for FFT', type=int, default=128, required=False)
    # parser.add_argument('--plotby', nargs='*', help='order to plot by', choices=['name','sum','circularity'], required=False)
    parser.add_argument('--recalculate', help='recalculate everything', choices=['y','n'], required=False, default='y')
    parser.add_argument('--thumbnails', help='make thumbnails', choices=['y','n'], required=False, default='n')
    parser.add_argument('--thermometers', help='make thumbnails', choices=['y','n'], required=False, default='n')
    parser.add_argument('--dim', help='dimension of data', choices=[2], type=int, required=False, default=2)
    args = parser.parse_args()
    
    logger.info('coords %s images %s outdir %s nfft %d recalculate %s', args.coords, args.images, args.outdir,
                args.nfft, args.recalculate)

    # check that the output directory exists; if not, create it
    if (not os.path.isdir(args.outdir)):
        logger.info('creating output directory %s', args.outdir)
        os.makedirs(args.outdir)
    for subdir in (['FIGURES','IMAGES','HISTOGRAMS']):
        subdirpath = os.path.join(args.outdir, subdir)
        if (not os.path.isdir(subdirpath)):
            os.makedirs(subdirpath)
    
    picklefile = os.path.join(args.outdir, 'recalculate.pkl')
    prot = cPickle.HIGHEST_PROTOCOL
    if (args.recalculate=='y'):
        logger.info('recalculating everything ...')
        (results_dict, points_dict, power_dict, rfft_dict) = recalculate(args)
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
        write_thermometers(args.outdir, results_dict, points_dict)
    
    # analyze_results(args.outdir, results_dict)
        
    print_stats(results)
    # plot_annot_day(args.outdir, results, 'power', True)
    # plot_annot_day(args.outdir, results, 'power', False)
    # plot_annot_day(args.outdir, results, 'circularity')
    # plot_results(args.outdir, results) 


if __name__ == "__main__":
    main()

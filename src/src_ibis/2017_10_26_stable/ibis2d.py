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
#import pandas as pd

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
    # maxval = np.amax(img_orig)
    # img_boundary = np.copy(img_orig)
    # img_boundary[ boundary_np ] = maxval
    img_boundary = np.copy(img_orig)
    img_boundary[:,:,...] = 0
    img_boundary[boundary_np] = 1
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
            ctn2powers = dict()
            ctn2pxls = dict()
            ctn2num = dict()
            ctn2rgb = dict()
            ctn2gray = dict()
            for k in subset:
                ctn = results_dict[k]['ctnnum']
                ctn2powers[ctn] = ctn2powers.get(ctn,[ ]) + [ float(results_dict[k]['power'])]
                ctn2pxls[ctn] = ctn2pxls.get(ctn,[ ]) + [ float(results_dict[k]['pxlinmean'])]
                ctn2num[ctn] = ctn2num.get(ctn, 0) + 1
                imgmode = results_dict[k]['imgmode']
                if (imgmode == 'RGB'):
                    ctn2rgb[ctn] = ctn2rgb.get(ctn, 0) + 1
                elif (imgmode == 'GRAY'):
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
            plt.figure(figsize=(40,30))
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
                ctngray = get_gray(ctn2meanpxl[ctn])
                dx = mycolumn * diam
                dy = -1.0 * diam
                newx = [ x + dx for x in xcircle ]
                newy = [ y + dy for y in ycircle ]
                xy = zip(newx, newy)
                i = len(eclist) // 2
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
                c = results_dict[k]['ctnnum']
                if c in outliers:
                    continue
                dpower = results_dict[k]['power'] - ctn2meanpower[c]
                dpxl = results_dict[k]['pxlinmean'] - ctn2meanpxl[c]
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
    return(xy_raw, xy_interp, tot_length, area, form_factor, power_norm)

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
    return None

def read_table(filename, delim='\t'):
    logger.info('reading table from %s', filename)
    fp = open(filename, 'r')
    ret = dict()
    orig_order = [ ]
    header = fp.readline()
    cols = header.strip().split(delim)
    fields = cols[1:]
    ntok = len(cols)
    for line in fp:
        toks = line.strip().split(delim)
        if toks[0][0] == '#':
            logger.info('skipping comment line: %s' % line)
            continue
        assert(len(toks) == ntok), 'bad token count: %s' % line
        k = toks[0]
        assert(k not in ret), 'repeated row: %s' % k
        ret[k] = dict()
        for (k1, v1) in zip( fields, toks[1:]):
            ret[k][k1] = v1
        orig_order.append(k)
    logger.info('%s: %d rows, %d columns', filename, len(ret), ntok)
    return(ret)
    #return(ret, orig_order)
    
def write_table(my_table, fullpath):
    fp = open(fullpath, 'w')
    keys = sorted(my_table.keys())
    assert(len(keys) > 0), 'empty table'
    fields = sorted(my_table[keys[0]].keys())
    fp.write('\t'.join(['key'] + fields) + '\n')
    for k in keys:
        toks = [k] + [ str(my_table[k][f]) for f in fields ]
        fp.write('\t'.join(toks) + '\n')
    fp.close()

def plot_group_mean(mytable, byname, xname, yname, fullpath):
    # group rows according to column byname
    group_cnt = dict()
    x_sum = dict()
    y_sum = dict()
    for k in mytable.keys():
        g = mytable[k][byname]
        group_cnt[g] = group_cnt.get(g, 0) + 1
        x = float(mytable[k][xname])
        x_sum[g] = x_sum.get(g, 0) + x
        y = float(mytable[k][yname])        
        y_sum[g] = y_sum.get(g, 0) + y
    groups = sorted(group_cnt.keys())
    logger.info('found %d groups: %s', len(groups), str(groups))
    largegroups = [ g for g in groups if group_cnt[g] >= 5 ]
    logger.info('found %d large groups: %s', len(largegroups), str(largegroups))
    groups = largegroups
    
    x_avg = dict()
    y_avg = dict()
    for g in groups:
        n = float(group_cnt[g])
        x_avg[g] = x_sum[g] / n
        y_avg[g] = y_sum[g] / n
    #outfile = os.path.join(outdir, '_'.join([byname, 'between', xname, yname]) + '.txt')
    #fp = open(outfile, 'w')
    #fp.write('%s\t%s\t%s\t%s\n' % (byname, 'cnt', xname, yname))
    #for g in groups:
    #    fp.write('%s\t%d\t%f\t%f\n' % (g, group_cnt[g], x_avg[g], y_avg[g]))
    #fp.close()
    
    xx_between = [x_avg[g] for g in groups]
    yy_between = [y_avg[g] for g in groups]
    xx_delta = [ ]
    yy_delta = [ ]
    #fp = open(deltafile, 'w')
    #fp.write('%s\t%s\t%s\t%s\n' % (byname, 'organoid', xname, yname) )
    for k in sorted(mytable.keys()):
        g = mytable[k][byname]
        if g not in groups:
            continue
        x = float(mytable[k][xname]) - x_avg[g]
        y = float(mytable[k][yname]) - y_avg[g]
        xx_delta.append(x)
        yy_delta.append(y)
    #   fp.write('%s\t%s\t%f\t%f\n' % (g, k, x, y) )
    #fp.close()
    
    plt.figure(figsize=(12,6))
    i = 0
    for (methstr, xx, yy) in [ ('between', xx_between, yy_between),
        ('within', xx_delta, yy_delta) ]:
        
        logger.info('working on %s', methstr)
    
        i += 1
        plt.subplot(1,2,i)
        plt.xlabel(xname)
        plt.ylabel(yname)
        plt.scatter(xx, yy, c='k', marker='o')
                
        # linear model
        fit = sm.OLS(yy,sm.add_constant(xx)).fit()
        # logger.info('\n*** %s %s %s %s:***\n%s', byname, methstr, xname, yname, fit.summary())
        delta = max(xx) - min(xx)
        xx_fit = np.linspace(min(xx) - 0.1 * delta, max(xx) + 0.1*delta,100)
        logger.info('params: %s', str(fit.params))
        logger.info('pvalues: %s', str(fit.pvalues))
        plt.plot(xx_fit, xx_fit*fit.params[1] + fit.params[0], 'b')
        plt.title('%s-test, Rsq = %.2f, p = %.2g' % (methstr, fit.rsquared_adj, fit.pvalues[1]) )
        #plt.text(max(xx), max(yy), 'p = %.3g' % fit.pvalues[1],
        #         horizontalalignment='right', verticalalignment='top')
    
    plt.savefig(fullpath)

    return None

def plot_side_by_side_by_side(match_table, renameddir, origdir, outdir):
    renamedfiles = sorted(match_table.keys())
    origfiles = [ match_table[k]['origfilename'] for k in renamedfiles ]
    cntgood = 0
    cntbad = 0
    for (renamed, orig) in zip(renamedfiles, origfiles):
        renamedpath = os.path.join(renameddir, renamed)
        origpath = os.path.join(origdir, orig)
        errcnt = 0
        if not os.path.isfile(renamedpath):
            logger.info('%s not found', renamedpath)
            errcnt += 1
        if not os.path.isfile(origpath):
            logger.info('%s not found', origpath)
            errcnt += 1
        if errcnt > 0:
            cntbad += 1
            continue
        cntgood += 1
    logger.info('paired images: %d found, %d missing', cntgood, cntbad)
    
    group_dict = dict()
    group_origfile = dict()
    group_renamedfile = dict()
    for f in renamedfiles:
        g = match_table[f]['ctnnum']
        group_dict[g] = group_dict.get(g, 0) + 1
        group_renamedfile[g] = group_renamedfile.get(g, [ ]) + [ f ]
        origfile = match_table[f]['origfilename']
        group_origfile[g] = group_origfile.get(g, [ ]) + [ origfile ]
    groups = sorted(group_dict.keys())
    logger.info('%d groups', len(groups))
    
    plotfile = os.path.join(outdir, 'side_by_side_by_side.pdf')
    pdf = PdfPages(plotfile)
    for g in groups:
        if g not in ['024', '039', '086']:
            continue
        nrow = group_dict[g]
        ncol = 4
        plt.figure(figsize=(12, 3 * nrow))
        plt.suptitle('CTN%s Tumor Day6, %d organoids' % (g, nrow))
        row = 0
        for (renamed, orig) in zip( group_renamedfile[g], group_origfile[g] ):
           row = row + 1
           i = (row - 1) * ncol
           orgnum = match_table[renamed]['orgnum']
           img_renamed = mpimg.imread(os.path.join(renameddir, renamed))
           img_orig = mpimg.imread(os.path.join(origdir, orig))
           logger.info('%s\t%s\t%s', renamed, str(img_renamed.shape), str(img_orig.shape))
           ny = img_renamed.shape[1]
           assert(img_orig.shape[1] == 2 * ny), '%f has bad shape' % renamed
           orig_left = img_orig[:, 0:ny, ...]
           orig_right = img_orig[:, ny:(2*ny), ...]
           (img_orig_dic, img_orig_k14) = (orig_left, orig_right)
           if (g == '024'):
                (img_orig_dic, img_orig_k14) = (orig_right, orig_left)
           img_diff = img_renamed - img_orig_dic
           plt.subplot(nrow, ncol, i+1)
           plt.imshow(img_renamed)
           plt.ylabel('CTN%s org %s' % (g,orgnum))
           plt.subplot(nrow, ncol, i+2)
           plt.imshow(img_orig_dic)
           plt.subplot(nrow, ncol, i+3)
           plt.imshow(img_orig_k14)
           plt.subplot(nrow, ncol, i+4)
           plt.imshow(img_diff)
           logger.info('done with %s', renamed)
        pdf.savefig()
        plt.close()
        logger.info('done with %s', g)
    pdf.close()

def veena_find_orig(veena_table_raw, orig_dir, orig_order):
    ret = dict()
    found_order = [ ]
    valid_score = ['0','1','2','3']
    assert(os.path.isdir(orig_dir)), 'original directory missing: %s' % orig_dir
    for f in orig_order:
        annot = veena_table_raw[f]['Type']
        if (annot != 'Tumor'):
            continue
        day = veena_table_raw[f]['Day']
        if (day != '6'):
            continue
        invasion = veena_table_raw[f]['Invasion Score']
        k14 = veena_table_raw[f]['K14 score']
        # if ( (invasion not in valid_score) or (k14 not in valid_score)):
            # logger.info('skipping %s %s', f, str(veena_table_raw[f]))
            # continue
        # if it looks like a directory structure, get the filename
        (path, filename) = os.path.split(f)
        fullpath = os.path.join(orig_dir, filename)
        if not os.path.isfile(fullpath):
            # logger.info('%s missing', fullpath)
            continue
        assert(filename not in ret), 'repeated file: %s' % f
        ret[filename] = dict()
        for (key, val) in veena_table_raw[f].items():
            ret[filename][key] = val
        found_order.append(filename)
    logger.info('%d of %d files found', len(ret), len(orig_order))
    return(ret, found_order)

def read_image_orig(fullpath):
    img_orig = mpimg.imread(fullpath)
    img_rgb = None
    if (img_orig.ndim == 3):
        img_rgb = img_orig
    elif (img_orig.ndim == 2):
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
    if ('CTN024' == filename[:6]):
        (img_orig_dic, img_orig_k14) = (orig_right, orig_left)
        
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
    
    return(img_orig_dic, img_orig_k14, scale, str(phototag))

def veena_matchmaker(veena_orig, orig_dir, renamed_dir, found_order, outdir):

    origfiles = found_order
    newfiles = sorted(os.listdir(renamed_dir))
    ctn_to_newfiles = dict()
    orig_to_new = dict()
    
    for nf in newfiles:
        if not os.path.isfile(os.path.join(renamed_dir, nf)):
            logger.info('skipping %s, not a file', nf)
            continue
        toks = nf.split('_')
        assert(len(toks) == 3), 'bad filename: %s' % nf
        if (toks[1] != 'Day6'):
            continue
        ctn = toks[0]
        ctn_to_newfiles[ctn] = ctn_to_newfiles.get(ctn, [ ]) + [ nf ]
    ctn_to_origfiles = dict()
    for f in found_order:
        ctn = veena_orig[f]['CTN']
        ctn_to_origfiles[ctn] = ctn_to_origfiles.get(ctn, [ ]) + [ f ]
    ctns = sorted(ctn_to_origfiles.keys())
    
    plotfile = os.path.join(outdir, 'propose_match.pdf')
    pdf = PdfPages(plotfile)
    
    for ctn in ctns:
        orig_files = ctn_to_origfiles[ctn]
        new_files = ctn_to_newfiles[ctn]
        (norig, nnew) = (len(orig_files), len(new_files))
        if (norig != nnew):
            logger.info('%s: %d orig, %d renamed', ctn, len(ctn_to_origfiles[ctn]), len(ctn_to_newfiles[ctn]))
        nbigger = max(norig, nnew)
        nsmaller = min(norig, nnew)
        
        orig_to_img = dict()
        orig_to_k14 = dict()
        orig_to_sum = dict()
        orig_matched = dict()
        new_to_img = dict()
        new_to_sum = dict()
        new_matched = dict()
        
        for o in orig_files:
            orig_matched[o] = False
            img_orig = mpimg.imread(os.path.join(orig_dir, o))
            ny = img_orig.shape[1]/2
            orig_left = img_orig[:, 0:ny, ...]
            orig_right = img_orig[:, ny:(2*ny), ...]
            (img_orig_dic, img_orig_k14) = (orig_left, orig_right)            
            if (ctn == 'CTN024'):
                (img_orig_dic, img_orig_k14) = (orig_right, orig_left)
            orig_to_img[o] = img_orig_dic
            orig_to_k14[o] = img_orig_k14
            orig_to_sum[o] = np.sum(img_orig_dic)
            
        for n in new_files:
            new_matched[n] = False
            img_renamed = mpimg.imread(os.path.join(renamed_dir, n))
            new_to_img[n] = img_renamed
            new_to_sum[n] = np.sum(img_renamed)
            
        for o in orig_files:
            found = False
            newname = None
            osum = orig_to_sum[o]
            for n in new_files:
                if new_matched[n] == True:
                    continue
                if osum != new_to_sum[n]:
                    continue
                maxdiff = np.amax(np.abs(orig_to_img[o] - new_to_img[n]))
                if (maxdiff == 0):
                    found = True
                    newname = n
                    break
            if not found:
                continue
            orig_to_new[o] = dict()
            orig_to_new[o]['new'] = n
            orig_matched[o] = True
            new_matched[n] = True
        
        logger.info('%s\t%d orig\t%d new', ctn, len(orig_files), len(new_files))
        for o in orig_files:
            if (orig_matched[o]):
                logger.info('%s -> %s', o, orig_to_new[o]['new'])
        for o in orig_files:
            if (not orig_matched[o]):
                logger.info('orig not matched: %s', o)
        for n in new_files:
            if (not new_matched[n]):
                logger.info('new not matched: %s', n)
                
        nrow = len(orig_files)
        nmatched = 0
        for n in new_files:
            if (not new_matched[n]):
                nrow += 1
            else:
                nmatched += 1
        ncol = 4
        plt.figure(figsize=(12, 3 * nrow))
        plt.suptitle('%s Tumor Day6, %d old, %d new, %d matched' % (ctn, len(orig_files), len(new_files), nmatched))

        rownum = 0        
        for o in orig_files:
            if not orig_matched[o]:
                continue
            n = orig_to_new[o]['new']
            rownum += 1
            i = (rownum - 1) * ncol # plot sequence number starting point
            plt.subplot(nrow, ncol, i+1)
            plt.imshow(orig_to_k14[o])
            plt.title('%s' % o)
            plt.subplot(nrow, ncol, i+2)
            plt.imshow(orig_to_img[o])
            plt.subplot(nrow, ncol, i+3)
            plt.imshow(new_to_img[n])
            plt.title('%s' % n)
            img_diff = orig_to_img[o] - new_to_img[n]
            plt.subplot(nrow, ncol, i+4)
            plt.imshow(img_diff)
        for o in orig_files:
            if orig_matched[o]:
                continue
            rownum += 1
            i = (rownum - 1) * ncol
            plt.subplot(nrow, ncol, i+1)
            plt.imshow(orig_to_k14[o])
            plt.title('%s' % o)
            plt.subplot(nrow, ncol, i+2)
            plt.imshow(orig_to_img[o])
        for n in new_files:
            if new_matched[n]:
                continue
            rownum += 1
            i = (rownum - 1) * ncol
            plt.subplot(nrow, ncol, i+3)
            plt.imshow(new_to_img[n])
            plt.title('%s' % n)
            
        logger.info('done with %s', ctn)
        pdf.savefig()
        plt.close()
    pdf.close()            

    return(orig_to_new)

def get_merge_table(orig_to_new, veena_table_raw, xy_dir):
    merge_table = dict()
    SCORE_LIST = ['0','1','2','3']
    ctns = dict()
    for v in sorted(veena_table_raw.keys()):
        (path, o) = os.path.split(v)
        if o not in orig_to_new:
            continue
        n = orig_to_new[o]['new']
        (ctn_veena, day_veena, treatment_veena, duplicate_veena, invasion_veena, k14_veena, type_veena) = (
            veena_table_raw[v]['CTN'], veena_table_raw[v]['Day'], veena_table_raw[v]['Treatment'], veena_table_raw[v]['Is_Duplicate'],
            veena_table_raw[v]['Invasion Score'], veena_table_raw[v]['K14 score'],
            veena_table_raw[v]['Type'])
        if invasion_veena not in SCORE_LIST:
            continue
        if k14_veena not in SCORE_LIST:
            continue
        if day_veena != '6':
            continue
        if treatment_veena != 'Vehicle':
            logger.info('unexpected treatment: %s', treatment_veena)
        if duplicate_veena not in ['NA','No']:
            logger.info('unexpected duplicate: %s', duplicate_veena)
        if type_veena != 'Tumor':
            logger.info('unexpected duplicate: %s', type_veena)
        (b, extension) = os.path.splitext(n) # b stands for basename
        toks = b.split('_')
        assert(len(toks) == 3), 'bad token count: %s' % b
        (ctnstr, daystr, orgnum) = toks
        annotstr = 'Tumor'
        k = '_'.join([ctnstr, annotstr, daystr, orgnum])
        assert(k not in merge_table), 'bad key: %s' % k
        xy_file = os.path.join(xy_dir, k + '_xy.txt')
        if not os.path.isfile(xy_file):
            # logger.info('%s missing, skipping', xy_file)
            continue
        merge_table[k] = dict()
        merge_table[k]['file_orig'] = o
        merge_table[k]['file_new'] = n
        merge_table[k]['ctn'] = ctnstr
        merge_table[k]['orgnum'] = orgnum
        merge_table[k]['invasion_veena'] = invasion_veena
        merge_table[k]['k14_veena'] = k14_veena
        ctns[ctnstr] = True
    logger.info('%d organoids from %d ctns retained from %d original matches', len(merge_table), len(ctns), len(orig_to_new))
    return(merge_table)

def get_exif_tags(orig_table, orig_dir):
    new_table = copy.deepcopy(orig_table)
    new_fields = ['exif_photometric', 'exif_resolution', 'exif_samplesperpixel']
    for k in new_table:
        for f in new_fields:
            new_table[k][f] = '-1'

    for k in new_table:
        fullpath = os.path.join(orig_dir, new_table[k]['file_orig'])
        tags = get_tags(fullpath)
        xtag = tags.get('Image XResolution', "1")
        ytag = tags.get('Image YResolution', "1")
        samplesperpixel = str(tags['Image SamplesPerPixel'])
        xresolution = str(xtag)
        yresolution = str(ytag)
        if (xresolution != yresolution):
            logger.warn('%s bad resolution %s %s', fullpath, xresolution, yresolution)
            #scale = float(xresolution)
        resolution = eval("1.0*" + xresolution) # sometimes scalestr is 122093/62500
        photometric = tags.get('Image PhotometricInterpretation', '-1')
        samplesperpixel = tags.get('Image SamplesPerPixel', '1')
        new_table[k]['exif_resolution'] = str(resolution)
        new_table[k]['exif_photometric'] = str(photometric)
        new_table[k]['exif_samplesperpixel'] = str(samplesperpixel)
    
    return(new_table)
    

def get_dic_xy(merge_table, orig_dir, xy_dir, nfft, outdir):
    
    invasion_table = copy.deepcopy(merge_table)
    new_fields = ['invasion_spectral', 'invasion_ff']
    for k in invasion_table:
        for f in new_fields:
            invasion_table[k][f] = '-1'
            
    plotfile = os.path.join(outdir, 'dic_xy.pdf')
    pdf = PdfPages(plotfile)
    ctn_to_keys = dict()
    for k in merge_table:
        ctn = merge_table[k]['ctn']
        ctn_to_keys[ctn] = ctn_to_keys.get(ctn, [ ]) + [k]
    ctn_list = sorted(ctn_to_keys.keys())

    k_vector = np.arange(1 + (nfft/2)) # since rfft uses only half the range
    k_sq = k_vector * k_vector # element multiply
    # a reasonable smoothing function is exp(- omega^2 t^2) with t the smoothing width
    # t = 1 is nearest neighbor
    # omega = (2 pi / T) k
    # so smooth the power with exp( - (2 pi / T)^2 k^2 )
    fac = 2.0 * math.pi / float(nfft)
    facsq = fac * fac
    smooth = np.exp(-facsq * k_sq)
    
    #    for c in ctn_list:
    # for c in ['CTN005']:
    # for c in ctn_list[:5]:
    for c in ctn_list:
        organoids = sorted(ctn_to_keys[c])
        nrow = len(organoids)
        ncol = 4
        org_list = [ ]
        inv_list = [ ]
        ff_list = [ ]
        sumpower_list = [ ]
        sumpowerksq_list = [ ]
        plt.figure(figsize=(25, 4 * nrow))
        plt.suptitle('%s Tumor Day6, %d organoids' % (c, nrow))
        rownum = 0
        for k in organoids:
            
            xy_file = os.path.join(xy_dir, k + '_xy.txt')
            (xy_raw, xy_interp, tot_length, area, form_factor, power_norm) = xyfile_to_spectrum(xy_file, nfft)
            power_norm = power_norm * smooth
            power_ksq = power_norm * k_sq
            sumpower = sum(power_norm[2:])
            sumpowerksq = sum(power_ksq[2:])

            rownum += 1
            fullpath = os.path.join(orig_dir, merge_table[k]['file_orig'])
            (img_dic, img_k14, scale, phototag) = read_image_orig(fullpath)
            i = (rownum - 1) * ncol # plot sequence number starting point

            inv = int(merge_table[k]['invasion_veena'])
            k14 = int(merge_table[k]['k14_veena'])

            # K14 image
            i += 1
            plt.subplot(nrow, ncol, i)
            plt.imshow(img_k14)
            plt.title('k14 %d' % k14)
            plt.ylabel('%s' % k)

            # DIC image
            i += 1
            plt.subplot(nrow, ncol, i)
            plt.imshow(img_dic)

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
            plt.scatter(scale * xy_interp[:,0], scale * xy_interp[:,1],
                    facecolors='none', edgecolors='b')
            plt.title('Veena %d' % inv)
            # ax.axis('equal')
            ax.set_xlim(0, img_dic.shape[1] - 1)
            ax.set_ylim(0, img_dic.shape[0] - 1)
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
            plt.title('sumksq %.3f ff %.3f' % (sumpowerksq, form_factor))

            org_list.append(merge_table[k]['orgnum'])
            inv_list.append(inv)
            ff_list.append(form_factor)
            sumpower_list.append(sumpower)
            sumpowerksq_list.append(sumpowerksq)
            
            invasion_table[k]['invasion_spectral'] = str(sumpowerksq)
            invasion_table[k]['invasion_ff'] = str(form_factor)
            
        pdf.savefig()
        plt.close()
        
        xlo = -0.5
        xhi = 3.5
        plt.figure(figsize=(25,5))
        plt.suptitle('%s Tumor Day6, %d organoids' % (c, nrow))

        plt.subplot(131)
        for (x1, y1, a1) in zip(inv_list, ff_list, org_list):
            plt.text(x1, y1, a1, va='center', ha='center')
        plt.xlim(xlo,xhi)
        plt.ylim(0,max(ff_list))
        plt.xlabel('Invasion Score')
        plt.ylabel('Form Factor')

        #plt.subplot(152)
        #for (x1, y1, a1) in zip(inv_list, sumpower_list, org_list):
        #    plt.text(x1, y1, a1, va='center', ha='center')
        #plt.xlim(xlo,xhi)
        #plt.ylim(0,max(sumpower_list))
        #plt.xlabel('Invasion Score')
        #plt.ylabel('Sum(Power)')

        plt.subplot(132)
        # plt.scatter(inv_list, sum3_list)
        for (x1, y1, a1) in zip(inv_list, sumpowerksq_list, org_list):
            plt.text(x1, y1, a1, va='center', ha='center')
        plt.xlim(xlo,xhi)
        plt.ylim(0,max(sumpowerksq_list))
        plt.xlabel('Invasion Score')
        plt.ylabel('Sum(Power k^2)')
        
        plt.subplot(133)
        plt.xlim(0,max(sumpowerksq_list))
        plt.ylim(0,max(ff_list))
        for (x1, y1, a1) in zip(sumpowerksq_list, ff_list, org_list):
            plt.text(x1, y1, a1, va='center', ha='center')
        plt.xlabel('Sum(Power k^2)')
        plt.ylabel('Form Factor')

        pdf.savefig()
        plt.close()
        
        logger.info('done with %s', c)
        
    pdf.close()             
    return(invasion_table)

def get_k14(orig_table, orig_dir, xy_dir, nfft, outdir):
    
    new_table = copy.deepcopy(orig_table)
    new_fields = ['k14_mean', 'k14_sum', 'photometric_interpretation']
    for k in new_table:
        for f in new_fields:
            new_table[k][f] = '-1'
            
    plotfile = os.path.join(outdir, 'k14.pdf')
    pdf = PdfPages(plotfile)

    ctn_to_keys = dict()
    for k in new_table:
        ctn = new_table[k]['ctn']
        ctn_to_keys[ctn] = ctn_to_keys.get(ctn, [ ]) + [k]
    ctn_list = sorted(ctn_to_keys.keys())

    for c in ctn_list:
        logger.info('starting %s ...', c)
        organoids = sorted(ctn_to_keys[c])
        nrow = len(organoids)
        ncol = 4
        org_list = [ ]
        inv_list = [ ]
        k14mean_list = [ ]
        k14sum_list = [ ]
        k14veena_list = [ ]
        plt.figure(figsize=(30, 4 * nrow))
        plt.suptitle('%s Tumor Day6, %d organoids' % (c, nrow))
        rownum = 0
        for k in organoids:
            
            logger.info('... %s', k)
            
            rownum += 1
            fullpath = os.path.join(orig_dir, new_table[k]['file_orig'])
            (img_dic, img_k14, scale, photometric_interpretation) = read_image_orig(fullpath)
            
            xy_file = os.path.join(xy_dir, k + '_xy.txt')
            (xy_raw, xy_interp, tot_length, area, form_factor, power_norm) = xyfile_to_spectrum(xy_file, nfft)
            (img_fill, fill_np) = create_boundary(img_k14, xy_interp, scale, 'fill')
            pixels_inside = img_k14[ fill_np > 0 ]
            pixels_outside = img_k14[ fill_np == 0 ]
            
            i = (rownum - 1) * ncol # plot sequence number starting point

            k14veena = int(new_table[k]['k14_veena'])
            k14sum = np.sum(pixels_inside.ravel())
            k14mean = np.mean(pixels_inside.ravel())
            ninside = len(pixels_inside.ravel())
            ntotal = len(img_k14.ravel())
            # logger.info('%s %d pixels, sum = %f, mean = %f', k, ninside, k14sum, k14mean)
            if (photometric_interpretation == '1'):
                k14mean = 255.0 - k14mean
                k14sum = (255.0 * ninside) - k14sum
                # logger.info('-> %d pixels, sum = %f, mean = %f', ninside, k14sum, k14mean)
            k14sum = k14sum / float(ntotal * 255)
            k14mean = k14mean / 255.0
            # logger.info('k14 mean %f, k14sum normalized %f for %d total pixels', k14mean, k14sum, ntotal)

            # K14 image
            i += 1
            plt.subplot(nrow, ncol, i)
            plt.imshow(img_k14)
            # plt.colorbar()
            (a, b) = (np.amin(img_k14), np.amax(img_k14))
            (na, nb) = ( np.sum(img_k14.ravel() == a), np.sum(img_k14.ravel() == b) )
            plt.title('k14 %d %d->%d %d->%d' % (k14veena, a, na, b, nb))
            plt.ylabel('%s' % k)

            # DIC image
            i += 1
            plt.subplot(nrow, ncol, i)
            (a, b) = (np.amin(img_dic), np.amax(img_dic))
            (na, nb) = ( np.sum(img_dic.ravel() == a), np.sum(img_dic.ravel() == b) )
            plt.title("phototag %s %d->%d %d->%d" % (photometric_interpretation, a, na, b, nb))
            plt.imshow(img_dic)
            # plt.colorbar()

            # fill region
            i += 1
            plt.subplot(nrow, ncol, i)
            plt.title('Mask')

            img_mask = np.zeros_like(img_k14)
            if (photometric_interpretation == '1'):
                img_mask[ fill_np > 0 ] = 1.0
            else:
                img_mask[ fill_np > 0] = 255
            # img_mask = 255 * img_fill
            # if (photometric_interpretation == '1'):
            #    img_mask = 255 - img_mask
            plt.imshow(img_mask, cmap='gray')

            # pixel intensity histograms
            i += 1
            plt.subplot(nrow, ncol, i)
            plt.hist(pixels_inside.ravel(), normed=True, alpha=0.5, label='K14 inside')
            plt.hist(pixels_outside.ravel(), normed=True, alpha=0.5, label='K14 outside')
            plt.hist(img_dic.ravel(), normed=True, alpha=0.5, label='DIC')
            # plt.hist(img_mask.ravel(), normed=True, alpha=0.5, label='Mask')
            plt.legend(loc='best', fancybox=True, framealpha=0.5)
            plt.title('Pixel Intensity')


            org_list.append(new_table[k]['orgnum'])
            k14mean_list.append(k14mean)
            k14sum_list.append(k14sum)
            k14veena_list.append(k14veena)
            inv_list.append(float(new_table[k]['invasion_spectral']))
            new_table[k]['k14_mean'] = str(k14mean)
            new_table[k]['k14_sum'] = str(k14sum)
            new_table[k]['photometric_interpretation'] = photometric_interpretation
            
        pdf.savefig()
        plt.close()
        
        plt.figure(figsize=(30,5))
        plt.suptitle('%s Tumor Day6, %d organoids' % (c, nrow))

        i = 150
        
        i += 1
        plt.subplot(i)
        #plt.scatter(k14veena_list, k14mean_list)
        plt.xlim(min(k14veena_list), max(k14veena_list))
        plt.ylim(min(k14mean_list), max(k14mean_list))
        for (x1, y1, a1) in zip(k14veena_list, k14mean_list, org_list):
            plt.text(x1, y1, a1, va='center', ha='center')
        fit1 = sm.OLS(k14veena_list,sm.add_constant(k14mean_list)).fit()
        plt.title('Rsq = %.2g' % fit1.rsquared_adj)
        plt.xlabel('K14, Veena')
        plt.ylabel('K14, Mean')

        i += 1
        plt.subplot(i)
        #plt.scatter(k14veena_list, k14sum_list)
        plt.xlim(min(k14veena_list), max(k14veena_list))
        plt.ylim(min(k14sum_list), max(k14sum_list))
        for (x1, y1, a1) in zip(k14veena_list, k14sum_list, org_list):
            plt.text(x1, y1, a1, va='center', ha='center')
        fit2 = sm.OLS(k14veena_list,sm.add_constant(k14sum_list)).fit()
        plt.title('Rsq = %.2g' % fit2.rsquared_adj)
        plt.xlabel('K14, Veena')
        plt.ylabel('K14, Sum')

        i += 1
        plt.subplot(i)
        plt.xlim(min(inv_list), max(inv_list))
        plt.ylim(min(k14veena_list), max(k14veena_list))
        for (x1, y1, a1) in zip(inv_list, k14veena_list, org_list):
            plt.text(x1, y1, a1, va='center', ha='center')
        veena_fit = sm.OLS(inv_list,sm.add_constant(k14veena_list)).fit()
        plt.title('Rsq = %.2g' % veena_fit.rsquared_adj)
        plt.xlabel('Invasion')
        plt.ylabel('K14, Veena')

        i += 1
        plt.subplot(i)
        # plt.scatter(inv_list, k14mean_list)
        plt.xlim(min(inv_list), max(inv_list))
        plt.ylim(min(k14mean_list), max(k14mean_list))
        for (x1, y1, a1) in zip(inv_list, k14mean_list, org_list):
            plt.text(x1, y1, a1, va='center', ha='center')
        mean_fit = sm.OLS(inv_list,sm.add_constant(k14mean_list)).fit()
        plt.title('Rsq = %.2g' % mean_fit.rsquared_adj)
        plt.xlabel('Invasion')
        plt.ylabel('K14, Mean')

        i += 1
        plt.subplot(i)
        # plt.scatter(inv_list, k14sum_list)
        plt.xlim(min(inv_list), max(inv_list))
        plt.ylim(min(k14sum_list), max(k14sum_list))
        for (x1, y1, a1) in zip(inv_list, k14sum_list, org_list):
            plt.text(x1, y1, a1, va='center', ha='center')
        sum_fit = sm.OLS(inv_list,sm.add_constant(k14sum_list)).fit()
        plt.title('Rsq = %.2g' % sum_fit.rsquared_adj)
        plt.xlabel('Invasion')
        plt.ylabel('K14, Sum')

        pdf.savefig()
        plt.close()
        
        logger.info('... done with %s', c)
    
    logger.info('closing %s', plotfile)    
    pdf.close()
    logger.info('returning new table')
    return(new_table)

def get_group_mean_delta(group_list, x_list):
    group_cnt = dict()
    group_sum = dict()
    group_mean = dict()
    sample_delta = [ ]
    for (g, x) in zip(group_list, x_list):
        group_cnt[g] = group_cnt.get(g, 0) + 1
        group_sum[g] = group_sum.get(g, 0) + float(x)
    groups = sorted(group_cnt.keys())
    for g in groups:
        group_mean[g] = float(group_sum[g]) / float(group_cnt[g])
    for (g, x) in zip(group_list, x_list):
        delta = x - group_mean[g]
        sample_delta.append(delta)
    return(group_mean, sample_delta)

def compare_invasion(invasion_table, outdir):
    samples = sorted(invasion_table.keys())
    ctn_cnt = dict()
    for s in samples:
        c = invasion_table[s]['ctn']
        ctn_cnt[c] = ctn_cnt.get(c, 0) + 1
    ctns = sorted(ctn_cnt.keys())
    logger.info('%d ctns', len(ctns))
    ctn_list = [ invasion_table[s]['ctn'] for s in samples ]
    iv_list = [ float(invasion_table[s]['invasion_veena']) for s in samples ]
    is_list = [ float(invasion_table[s]['invasion_spectral']) for s in samples ]
    (iv_mean, iv_delta) = get_group_mean_delta(ctn_list, iv_list)
    (is_mean, is_delta) = get_group_mean_delta(ctn_list, is_list)

    plt.figure(figsize=(12,5))

    plt.subplot(121)
    xx = [ iv_mean[c] for c in ctns ]
    yy = [ is_mean[c] for c in ctns ]
    aa = [ c[4:] for c in ctns ]
    for (x1, y1, a1) in zip(xx, yy, aa):
        plt.text(x1, y1, a1, va='center', ha='center')
    #plt.xlim(min(xx), max(xx))
    #plt.ylim(min(yy), max(yy))
    plt.xlabel('Invasion, Veena')
    plt.ylabel('Invasion, Spectral')
    
    mean_fit = sm.OLS(yy,sm.add_constant(xx)).fit()
    #logger.info('\n*** mean fit:***\n%s', mean_fit.summary())
    xx_fit = np.linspace(min(xx), max(xx),100)
    plt.plot(xx_fit, xx_fit*mean_fit.params[1] + mean_fit.params[0], 'b')
    #plt.text(min(xx), max(yy), 'Rsq = %.2g' % mean_fit.rsquared_adj, horizontalalignment='left', verticalalignment='top')
    plt.title('Mean Invasion by CTN, Rsq = %.2g' % mean_fit.rsquared_adj)
    
    plt.subplot(122)
    xx = iv_delta
    yy = is_delta
    aa = [ invasion_table[s]['ctn'][4:] + '_' + invasion_table[s]['orgnum'] for s in samples ]
    plt.scatter(xx, yy, c='k')
    plt.xlabel('Delta Invasion, Veena')
    plt.ylabel('Delta Invasion, Spectral')

    delta_fit = sm.OLS(yy,xx).fit()
    #logger.info('\n*** delta fit:***\n%s', delta_fit.summary())
    xx_fit = np.linspace(min(xx), max(xx),100)
    plt.plot(xx_fit, xx_fit*delta_fit.params[0], 'b')
    #plt.text(min(xx), max(yy), 'Rsq = %.2g' % delta_fit.rsquared_adj, horizontalalignment='left', verticalalignment='top')
    plt.title('Delta Invasion by Organoid, Rsq = %.2g' % delta_fit.rsquared_adj)
    
    plt.savefig(os.path.join(outdir, 'invasion_comparison.pdf'))
    plt.close()
    
def compare_k14(results_table, outdir):
    
    plotfile = os.path.join(outdir, 'k14_comparison.pdf')
    pdf = PdfPages(plotfile)
    
    samples_all = sorted(results_table.keys())
    
    for photometric in ['3']:
        
        samples = [ ]
        for s in samples_all:
            if (results_table[s]['exif_photometric'] == photometric) or (photometric == '3'):
                samples.append(s)
        
    
        ctn_cnt = dict()
        for s in samples:
            c = results_table[s]['ctn']
            ctn_cnt[c] = ctn_cnt.get(c, 0) + 1
        ctns = sorted(ctn_cnt.keys())
        logger.info('%d ctns', len(ctns))
        ctn_list = [ results_table[s]['ctn'] for s in samples ]
    
        for yfield in ['invasion_veena', 'invasion_spectral']:
            
            plt.figure(figsize=(15,20))
            nrow = 3
            ncol = 2
            i = 0

            inv_list = [ float(results_table[s][yfield]) for s in samples ]
            inv_list = rankdata(inv_list)
            (inv_mean, inv_delta) = get_group_mean_delta(ctn_list, inv_list)
    
            for xfield in ['k14_veena', 'k14_sum', 'k14_mean']:
                
                logger.info('testing PMI %s, %s, %s', photometric, yfield, xfield)
    
                x_list = [ float(results_table[s][xfield]) for s in samples ]
                x_list = rankdata(x_list)
                (x_mean, x_delta) = get_group_mean_delta(ctn_list, x_list)
                
                i += 1
                plt.subplot(nrow, ncol, i)
                xx = [ x_mean[c] for c in ctns ]
                yy = [ inv_mean[c] for c in ctns ]
                aa = [ c[4:] for c in ctns ]
                plt.xlim(min(xx), max(xx))
                plt.ylim(min(yy), max(yy))
                for (x1, y1, a1) in zip(xx, yy, aa):
                    plt.text(x1, y1, a1, va='center', ha='center')
                plt.xlabel(xfield)
                plt.ylabel(yfield)
            
                mean_fit = sm.OLS(yy,sm.add_constant(xx)).fit()
                logger.info('\n*** mean fit:***\n%s', mean_fit.summary())
                xx_fit = np.linspace(min(xx), max(xx),100)
                plt.plot(xx_fit, xx_fit*mean_fit.params[1] + mean_fit.params[0], 'b')
                #plt.text(min(xx), max(yy), 'Rsq = %.2g' % mean_fit.rsquared_adj, horizontalalignment='left', verticalalignment='top')
                plt.title('Between-CTN test, Rsq = %.2g, p = %.2g' % (mean_fit.rsquared_adj, mean_fit.pvalues[1]))
            
                i += 1
                plt.subplot(nrow, ncol, i)
                xx = x_delta
                yy = inv_delta
                # aa = [ results_table[s]['ctn'][4:] + '_' + results_table[s]['orgnum'] for s in samples ]
                plt.xlabel('Delta ' + xfield)
                plt.ylabel('Delta ' + yfield)
        
                delta_fit = sm.OLS(yy,sm.add_constant(xx)).fit()
                logger.info('\n*** delta fit:***\n%s', delta_fit.summary())
                xx_fit = np.linspace(min(xx), max(xx),100)
                plt.hist2d(xx, yy,bins=20)
                #plt.scatter(xx, yy, c='k')
                #plt.plot(xx_fit, xx_fit*delta_fit.params[1] + delta_fit.params[0], 'b')
                #plt.text(min(xx), max(yy), 'Rsq = %.2g' % delta_fit.rsquared_adj, horizontalalignment='left', verticalalignment='top')
                plt.title('Within-CTN test, Rsq = %.2g, p = %.2g' % (delta_fit.rsquared_adj, delta_fit.pvalues[1]))
    
            pdf.savefig()
            plt.close()
    pdf.close()
    return(None)
    
def plot_k14_survival(survival_table, fullpath):
    plt.figure()
    years_str = survival_table.keys()
    years_int = [ int(y) for y in years_str ]
    years_int = sorted(years_int)
    death_k14pos = [ float(survival_table[str(y)]['deathK14pos']) for y in years_int ]
    death_k14neg = [ float(survival_table[str(y)]['deathK14neg']) for y in years_int ]
    plt.scatter(years_int, death_k14pos, c='r', label='K14+')
    plt.plot(years_int, death_k14pos, c='r')
    plt.scatter(years_int, death_k14neg, c='k', label='K14-')
    plt.plot(years_int, death_k14neg, c='k')
    plt.xlabel('Years')
    plt.ylabel('Mortality')
    plt.xlim(0,20)
    plt.ylim(0,1.0)
    plt.legend(loc='upper left', frameon=False)
    plt.title('de Silva Rudland et al 2011 Am J Path, p < 0.0001')
    plt.savefig(fullpath)
    plt.close()
    
    
def main():
    
    parser = argparse.ArgumentParser(description='Invasive boundary image score, 2D',
                                     epilog='Sample call: see run.sh',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--veenafile', help='Veena image names', required=False)
    parser.add_argument('--invasionfile', help='Spectral invasion score', required=False)
    parser.add_argument('--k14file', help='K14 score', required=False)
    parser.add_argument('--dic_orig', help='directory with DIC/K14 images, original names')
    parser.add_argument('--dic_renamed', help='directory with DIC images, renamed', required=False)
    parser.add_argument('--matchfile', help='match original to new names', required=False)
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

    logger.info('veenafile %s dic_orig %s dic_renamed %s', args.veenafile, args.dic_orig, args.dic_renamed)
    
    # check that the output directory exists; if not, create it
    if (not os.path.isdir(args.outdir)):
        logger.info('creating output directory %s', args.outdir)
        os.makedirs(args.outdir)

    k14_survival = read_table('k14_survival.txt')
    plot_k14_survival(k14_survival, 'k14_survival.pdf')
    
    if (args.invasionfile is None):
        
        veena_table_raw = read_table(args.veenafile)
        veena_order = sorted(veena_table_raw.keys())
        orig_to_new = None
        if (args.matchfile is None):
            logger.info('No match file, recalculating orig_to_new ...')
            (veena_table_origfound, found_order) = veena_find_orig(veena_table_raw, args.dic_orig, veena_order)
            orig_to_new = veena_matchmaker(veena_table_origfound, args.dic_orig, args.dic_renamed, found_order, args.outdir)
            matchfile = os.path.join(args.outdir,'orig_to_new.txt')
            logger.info('writing matches to %s', matchfile)
            write_table(orig_to_new, os.path.join(args.outdir, matchfile))
        else:
            logger.info('reading matches from %s', args.matchfile)
            orig_to_new = read_table(os.path.join(args.outdir, args.matchfile))
            logger.info('%d files matched', len(orig_to_new))
    
        merge_table = get_merge_table(orig_to_new, veena_table_raw, args.coords)
        write_table(merge_table, os.path.join(args.outdir, 'merge_table.txt'))
    
        invasion_table = get_dic_xy(merge_table, args.dic_orig, args.coords, args.nfft, args.outdir)
        write_table(invasion_table, os.path.join(args.outdir, 'invasion_table.txt'))
        
    invasion_table = read_table(os.path.join(args.outdir, args.invasionfile))
    compare_invasion(invasion_table, args.outdir)
    
    if (args.k14file is None):
        k14_table = get_k14(invasion_table, args.dic_orig, args.coords, args.nfft, args.outdir)
        write_table(k14_table, os.path.join(args.outdir, 'k14_table.txt'))
        
    k14_table = read_table(os.path.join(args.outdir, args.k14file))
    results_table = get_exif_tags(k14_table, args.dic_orig)
    # area_table = get_areainfo(exif_table, args.coords, args.nfft, args.outdir)
    write_table(results_table, os.path.join(args.outdir, 'results_table.txt'))
    
    compare_k14(results_table, args.outdir)
    plot_group_mean(results_table, 'ctn', 'k14_veena', 'invasion_veena', os.path.join(args.outdir, 'veena_mean.pdf'))
    

    return None
    
    
    # plot_side_by_side_by_side(match_table, args.dic_renamed, args.dic_orig, args.outdir)

    return None

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

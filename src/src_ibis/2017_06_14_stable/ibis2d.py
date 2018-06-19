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

import numpy as np
from scipy import interpolate

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as mpcm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.lines as mlines

from PIL import Image, ImageDraw

import logging
logging.basicConfig(format='%(levelname)s %(name)s.%(funcName)s: %(message)s')
logger = logging.getLogger('ibis2d')
logger.setLevel(logging.INFO)

MAXK = 21

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
        debug = (image_base == 'CTN106_Tumor_Day6_K14_01.tif')
        coord_base = string.replace(image_base, '_K14_', '_')
        if (coord_base[-4:] == '.tif'):
            coord_base = coord_base[:-4] + '.txt'
        if debug:
            logger.info('working on %s %s', image_base, coord_base)
        if coord_base in coord_dict:
            pairs.append( (coord_dict[coord_base], image_dict[image_base] ) )
            coord_matched[ coord_dict[coord_base] ] = True
            image_matched[ image_dict[image_base] ] = True
            if debug:
                logger.info('matched')
        else:
            if debug:
                logger.info('not matched')
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
    
def get_power_moments(power_rfft, n = -1):
    mysum = 0.0
    mymean = 0.0
    mylen = len(power_rfft)
    if (n == -1):
        n = len(power_rfft)
    assert(n <= mylen), 'bad length: ' + str(n)
    logger.info('n = %s', str(n))
    norm = power_rfft[1]
    mysum = sum(power_rfft[2:n]) / norm
    myfreq = range(n)
    mysum1 = sum(power_rfft[2:n] * myfreq[2:n]) / norm
    mysum2 = sum( (power_rfft[2:n] * myfreq[2:n]) * myfreq[2:n] ) / norm
    return(mysum, mysum1, mysum2)
    
def print_organoid_table(filename, big_results):
    fp = open(filename, 'w')
    fp.write('\t'.join(['fullpath','dirname','filename','sum', 'sum1', 'sum2', 'circ']) + '\n')
    for (fullname, points, points_grid, power_scaled, mysum, mysum1, mysum2, circ) in big_results:
        (dirname, filename) = os.path.split(fullname)
        fp.write('%s\t%s\t%s\t%f\t%f\t%f\t%f\n' % (fullname, dirname, filename, mysum, mysum1, mysum2, circ))
    fp.close()

def plot4(filename, shortname_to_points, mytitle, name_list):
    location = [221, 222, 223, 224]
    assert(len(name_list) == len(location))
    with PdfPages(filename) as pdf:

        plt.figure(figsize=(5.5,5))
        plt.suptitle(mytitle)
        for name in name_list:
            assert(name in shortname_to_points)
            points = shortname_to_points[name]
            myloc = location.pop(0)
            
            plt.subplot(myloc)
            x = list(points[:,0])
            y = list(points[:,1])
            x.append(x[0])
            y.append(y[0])
            plt.plot(x, y, 'k')
            plt.xlabel('x coordinate')
            plt.ylabel('y coordinate')
            plt.tick_params(axis='x',labelbottom='off')
            plt.tick_params(axis='y',labelleft='off')
            # plt.axes().set_aspect('equal')
            
        pdf.savefig()
        plt.close()  

def parse_filename(filepath):
    # the abbreviation is the last part of the file name
    (root, filename) = os.path.split(filepath)
    (basename, extstr) = os.path.splitext(filename)
    toks = basename.split('_')
    annot = 'Tumor'
    if 'NAT' in filename:
        annot = 'NAT'
    elif 'Normal' in filename:
        annot = 'Normal'
    ctnnum = toks[0][3:]
    daynum = toks[-2][3:]
    orgnum = toks[-1]
    return(annot, ctnnum, daynum, orgnum)

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

ANNOTS = ['Normal', 'NAT', 'Tumor']
DAYNUMS = ['0','6']
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

def write_plots(pdf, coord_file, points, points_grid, power_scaled, power_sum, circ,
                img, img_boundary, img_fill, scalestr,
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
    plt.title('Original')
               
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

def make_thumbnails(args):

    all_coords = get_files_recursive(args.coords, '*.txt')
    
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
    
    #plotfile = os.path.join(args.outdir, 'plots.pdf')
    #pdf = PdfPages(plotfile)
    
    npair = len(coord_image_pairs)
    nrow = math.ceil(math.sqrt(npair))
    plt.figure(figsize=(12,12))
    (ir, ic) = (0, 0)
    
    for (coord_file, image_file) in coord_image_pairs:
        
        assert(coord_file not in results_dict), 'repeated file: ' + coord_file
        mykey = coord_file
        results_dict[mykey] = dict()
        (annot, ctnnum, daynum, orgnum) = parse_filename(coord_file)
        for (k, v) in [ ('annot', annot), ('ctnnum', ctnnum), ('daynum', daynum),
            ('orgnum', orgnum)]:
            results_dict[mykey][k] = v
        
        if image_file is not None:        

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
            
    plt.close()
    return(None)

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
    
    plotfile = os.path.join(args.outdir, 'plots.pdf')
    pdf = PdfPages(plotfile)
    
    for (coord_file, image_file) in coord_image_pairs:
        
        assert(coord_file not in results_dict), 'repeated file: ' + coord_file
        mykey = coord_file
        results_dict[mykey] = dict()
        (annot, ctnnum, daynum, orgnum) = parse_filename(coord_file)
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
        logger.info('ds %f tot %f %f', ds, tot_length, contour_grid[-1])
        
        points_grid = get_interpolate(points, contour_length, contour_grid)
        
        area = get_area(points_grid)
        circularity = 4.0 * math.pi * area / (tot_length * tot_length)
        
        points_rfft = get_rfft(points_grid)
        power_rfft = get_power(points_rfft)
        # points_irfft = get_irfft(points_rfft)
        
        power_norm = power_rfft[1]
        power_scaled = (1.0 / power_norm) * power_rfft
        power_scaled[0] = 0.0
        power_scaled[1] = 0.0
        (mysum, mysum1, mysum2) = get_power_moments(power_rfft, MAXK)
        power_sum = mysum

        results_dict[mykey]['power'] = power_sum
        results_dict[mykey]['circularity'] = circularity
        results_dict[mykey]['area'] = area
        results_dict[mykey]['circumference'] = tot_length

        if image_file is not None:        

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
            
        write_plots(pdf, coord_file, points, points_grid, power_scaled, power_sum, circularity,
                    img, img_boundary, img_fill, scalestr,
                    pixels)
        
    pdf.close()
    tagfp.close()
    logger.info('writing results')
    write_results(args.outdir, results_dict)
    
    return(None)
    
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
    print"\n"

        
    
    

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
    
    if (args.thumbnails=='y'):
        logger.info('making thumbnails')
        make_thumbnails(args)
        
    if (args.recalculate=='y'):
        logger.info('recalculating everything ...')
        recalculate(args)
        
    results = read_results(args.outdir)
    print_stats(results)
    plot_annot_day(args.outdir, results, 'power', True)
    plot_annot_day(args.outdir, results, 'power', False)
    #plot_annot_day(args.outdir, results, 'circularity')
    # plot_results(args.outdir, results) 


if __name__ == "__main__":
    main()

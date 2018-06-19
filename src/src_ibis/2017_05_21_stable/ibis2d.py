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
    image_dict = convert_fullpath_to_dict(all_image, check_tokens=True)
    image_keys = sorted(image_dict.keys())
    for image_base in image_keys:
        coord_base = string.replace(image_base, '_K14_', '_')
        if (coord_base[-4:] == '.tif'):
            coord_base = coord_base[:-4] + '.txt'
        if coord_base in coord_dict:
            pairs.append( (coord_dict[coord_base], image_dict[image_base] ) )
    return(pairs)

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
    
def plot_results(outdir, results):
    samples = sorted(results.keys())
    fields = ['circularity','power','pixelcnt','pixelmean', 'pixelstdev' ]
    logger.info('plot fields: %s', str(fields))
    nfields = len(fields)
    sample_color = dict()
    myhandles = dict()
    annots = ['Normal', 'NAT', 'Tumor']
    colors = ['g', 'y', 'r']
    for (a, c) in zip(annots, colors):
        sample_color[a] = c
        myhandles[a] = mlines.Line2D([ ], [ ], color=c, linestyle='none', marker='o', mec='none', markersize=5, label=a)
        
    for myday in ['0', '6']:
        sample_subset = [ s for s in samples if results[s]['daynum'] == myday ]
        sample_colors = [ sample_color.get(results[s]['annot'],'c') for s in sample_subset ]
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
                plotfile = os.path.join(outdir, plotfile)
                plt.savefig(plotfile)
                plt.close()
    
def recalculate(args):
    
    all_coords = get_files_recursive(args.coords, '*.txt')
    
    doImages = args.images is not None
    logger.info('doImages = %s', str(doImages))
    all_images = [ ]
    if doImages:
        all_images = get_files_recursive(args.images, '*.tif')

    #logger.info('coordinate files: %s', str(all_coords))
    #logger.info('image files: %s', str(all_images))

    coord_image_pairs = [ (c, None) for c in all_coords ]
    if doImages:
        coord_image_pairs = match_coord_image(all_coords, all_images)
    
    #logger.info('coord image pairs: %s', str(coord_image_pairs))
    logger.info('%d coordinate file, %d image files, %d pairs',
                len(all_coords), len(all_images), len(coord_image_pairs))

    # check that the output directory exists; if not, create it
    if (not os.path.isdir(args.outdir)):
        logger.info('creating output directory %s', args.outdir)
        os.makedirs(args.outdir)
    
    big_results = [ ]
    shortname_to_points = dict()
    results_dict = dict()
    
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
        (root, coord_base) = os.path.split(coord_file)
        shortname_to_points[coord_base] = points

        if image_file is not None:        

            img = mpimg.imread(image_file)
            logger.info('image shape: %s', str(img.shape))
            width = img.shape[0]
            height = img.shape[1]
            
            # images are stored as (row, column) arrays
            # this means that the 1st coordinate is y, 2nd coordinate is x
            # and (0, 0) is at the top left
            points_boundary = [ ]
            points_xy = [ ]
            for rec in points_grid:
                (row, col) = (2 * rec[1], 2 * rec[0])
                row = min(row, img.shape[0] - 1)
                col = min(col, img.shape[1] - 1)
                points_boundary.append( [row, col] )
                points_xy.append(col)
                points_xy.append(row)
            
            img_base = os.path.basename(image_file)
            (img_root, img_ext) = os.path.splitext(img_base)

            #img_pdf = os.path.join(args.outdir, img_root + '.pdf')
            #logger.info('writing to %s', img_pdf)
            #mpimg.imsave(img_pdf, img, cmap=mpcm.gray)
            
            # histogram of pixel values
            plt.figure()
            plt.hist(img.flatten(), bins=100)
            plt.title(img_root + ' entire image')
            plt.savefig( os.path.join(args.outdir, img_root + '_hist.pdf'))
            plt.close()
            minval = np.amin(img)
            maxval = np.amax(img)
            logger.info('min %f max %f', minval, maxval)

            inside_pil = Image.new('1', (height, width), 0)
            ImageDraw.Draw(inside_pil).polygon(points_xy, outline=1, fill=1)
            inside_np = np.array(inside_pil)

            pixels = img[ inside_np > 0 ]
            pixelcnt = pixels.shape[0]
            pixelmean = np.mean(pixels)
            pixelmedian = np.median(pixels)
            pixelstdev = np.std(pixels)
            pixelmad = mad(pixels)
            logger.info('pixels shape %s cnt %d mean %f median %f sd %f', str(pixels.shape),
                        pixelcnt, pixelmean, pixelmedian, pixelstdev)
            plt.figure()
            plt.hist(pixels.flatten(), bins=100)
            plt.title(img_root + ' inside')
            plt.savefig( os.path.join(args.outdir, img_root + '_inside_hist.pdf'))
            plt.close()
            
            # add boundary to image
            for (r, c) in points_boundary:
                img[r, c] = maxval
            img_pdf = os.path.join(args.outdir, img_root + '_boundary.pdf')
            logger.info('writing to %s', img_pdf)
            mpimg.imsave(img_pdf, img, cmap=mpcm.gray)
            
            # add fill to image
            img[ inside_np > 0 ] = maxval
            img_fill = os.path.join(args.outdir, img_root + '_fill.pdf')
            logger.info('writing %s', img_fill)
            mpimg.imsave(img_fill, img, cmap=mpcm.gray)
            
            results_dict[mykey]['pixelcnt'] = pixelcnt
            results_dict[mykey]['pixelmean'] = pixelmean
            results_dict[mykey]['pixelmedian'] = pixelmedian
            results_dict[mykey]['pixelstdev'] = pixelstdev
            results_dict[mykey]['pixelmad'] = pixelmad
            
            
        big_results.append( (coord_file, points, points_grid,
                             power_scaled, mysum, mysum1, mysum2, circularity) )
        results_dict[mykey]['power'] = mysum
        results_dict[mykey]['circularity'] = circularity

        
    big_results_by_sum = sorted(big_results, key = lambda tup: (tup[4], tup))
    big_results_by_sum1 = sorted(big_results, key = lambda tup: (tup[5], tup))
    big_results_by_sum2 = sorted(big_results, key = lambda tup: (tup[6], tup))
    big_results_by_circ = sorted(big_results, key = lambda tup: (1.0 - tup[7], tup))
    
    organoid_table = os.path.join(args.outdir, 'organoid_table.txt')
    print_organoid_table(organoid_table, big_results)
    
    write_results(args.outdir, results_dict)
    
    if False:
        filename = os.path.join(args.outdir,"ctn053_day6.pdf")
        file_list = ['CTN053_Day6_1.txt', 'CTN053_Day6_19.txt', 'CTN053_Day6_18.txt', 'CTN053_Day6_2.txt']
        plot4(filename, shortname_to_points, '(B) CTN053, Tumor, Day6', file_list)

        filename = os.path.join(args.outdir,"ctn094_day6.pdf")
        file_list = ['CTN094_Day6_1.txt', 'CTN094_Day6_10.txt', 'CTN094_Day6_14.txt', 'CTN094_Day6_20.txt']
        plot4(filename, shortname_to_points, '(C) CTN094, Tumor, Day6', file_list)
    
    if (args.plotby is not None):
        for my_plotby in args.plotby:
            myfile = None
            mylist = None
            if (my_plotby == 'name'):
                myfile = 'byname.pdf'
                mylist = big_results
                # do nothing
            elif (my_plotby == 'sum'):
                myfile = 'bysum.pdf'
                mylist = big_results_by_sum
            elif (my_plotby == 'circularity'):
                myfile = 'bycirc.pdf'
                mylist = big_results_by_circ
            else:
                logger.warning('bad plotby: %s', args.plotby)
        
            logger.info('plotting %s', myfile)
    
            pltfile = os.path.join(args.outdir, myfile)
            with PdfPages(pltfile) as pdf:
            
                for (fullname, points, points_grid, power_scaled, power_sum, power_sum1, power_sum2, circ) in mylist:
                    logger.info('%s', fullname)
                    (root, shortname) = os.path.split(fullname)
    
                    plt.figure(figsize=(12,5))
        
                    plt.subplot(121)
                    # make sure the path is closed
                    x = list(points[:,0])
                    y = list(points[:,1])
                    x.append(x[0])
                    y.append(y[0])
                    plt.plot(x, y, 'k')
                    # plt.plot(points_grid[:,0], points_grid[:,1], 'ko')
                    plt.xlabel('column')
                    plt.ylabel('row')
                    plt.title('%s' % shortname)           
                    
                    plt.subplot(122)
                    xx = range(2,MAXK)
                    plt.plot(xx, power_scaled[xx])
                    # plt.title('%s: Power Spectrum' % basename)
                    plt.title('Power: sum %.3f, circ %.3f' %
                              (power_sum, circ))
                    (x1,x2,y1,y2) = plt.axis()
                    y2 = max(y2, 0.1)
                    plt.axis((x1,x2,y1,y2))
                    plt.xlabel('Harmonic component')
                    plt.ylabel('Scaled power')
                
                    pdf.savefig()
                    plt.close()
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

def main():
    
    parser = argparse.ArgumentParser(description='Invasive boundary image score, 2D',
                                     epilog='Sample call: see run.sh',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--coords', help='input directory with coordinates as .txt files with 2 columns, no header', required=True)
    parser.add_argument('--images', help='input directory with image files as .tif', required=False)
    parser.add_argument('--outdir', help='output directory', required=True)
    parser.add_argument('--nfft', help='number of points for FFT', type=int, default=128, required=False)
    parser.add_argument('--plotby', nargs='*', help='order to plot by', choices=['name','sum','circularity'], required=False)
    parser.add_argument('--recalculate', help='recalculate everything', choices=['y','n'], required=False, default='y')
    parser.add_argument('--dim', help='dimension of data', choices=[2], type=int, required=False, default=2)
    args = parser.parse_args()
    
    logger.info('coords %s images %s outdir %s nfft %d recalculate %s', args.coords, args.images, args.outdir,
                args.nfft, args.recalculate)
    

    if (args.recalculate=='y'):
        logger.info('recalculating everything ...')
        recalculate(args)
        
    results = read_results(args.outdir)
    plot_results(args.outdir, results) 


if __name__ == "__main__":
    main()

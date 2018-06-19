#!/usr/bin/env python
import argparse
import os
import fnmatch
import math

import numpy as np
from scipy import interpolate

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

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

def get_files_recursive(indir):
    ret = [ ]
    for root, dirnames, filenames in os.walk(indir):
        for filename in fnmatch.filter(filenames, '*.txt'):
            ret.append( os.path.join(root, filename) )
    return(ret)

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

def main():
    DIM = 2 # dimension of data
    parser = argparse.ArgumentParser(description='Invasive boundary image score, 2D',
                                     epilog='Sample call: see run.sh',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('indir', help='input directory, with .txt files with 2 columns, no header')
    parser.add_argument('outdir', help='output directory')
    parser.add_argument('n', help='number of points for FFT', type=int, default=4096)
    args = parser.parse_args()
    
    logger.info('indir %s outdir %s n %d', args.indir, args.outdir, args.n)
    all_files = get_files_recursive(args.indir)

    # check that the output directory exists; if not, create it
    if (not os.path.isdir(args.outdir)):
        logger.info('creating output directory %s', args.outdir)
        os.makedirs(args.outdir)
    
    big_results = [ ]
    shortname_to_points = dict()
    
    for fullname in all_files:
        # read the xy pairs from the input file
        # format is an array of tuples
        points = file_to_points(fullname, DIM)
        contour_length = get_contour_length(points)
        # make a regular grid
        tot_length = contour_length[-1]
        ds = tot_length / float(args.n)
        contour_grid = np.linspace(ds, tot_length, num=args.n)
        logger.info('ds %f tot %f %f', ds, tot_length, contour_grid[-1])
        
        points_grid = get_interpolate(points, contour_length, contour_grid)

        area = get_area(points_grid)
        circularity = 4.0 * math.pi * area / (tot_length * tot_length)
        
        points_rfft = get_rfft(points_grid)
        points_irfft = get_irfft(points_rfft)
        power_rfft = get_power(points_rfft)
        
        power_norm = power_rfft[1]
        power_scaled = (1.0 / power_norm) * power_rfft
        power_scaled[0] = 0.0
        power_scaled[1] = 0.0
        (mysum, mysum1, mysum2) = get_power_moments(power_rfft, MAXK)
        big_results.append( (fullname, points, points_grid, power_scaled, mysum, mysum1, mysum2, circularity) )
        (root, shortname) = os.path.split(fullname)
        shortname_to_points[shortname] = points
        
    # big_results_by_sum = sorted(big_results, key = lambda tup: (tup[4], tup))
    # big_results_by_sum1 = sorted(big_results, key = lambda tup: (tup[5], tup))
    # big_results_by_sum2 = sorted(big_results, key = lambda tup: (tup[6], tup))
    # big_results_by_circ = sorted(big_results, key = lambda tup: (1.0 - tup[7], tup))
    
    organoid_table = os.path.join(args.outdir, 'organoid_table.txt')
    print_organoid_table(organoid_table, big_results)
    
    filename = os.path.join(args.outdir,"ctn053_day6.pdf")
    file_list = ['CTN053_Day6_1.txt', 'CTN053_Day6_19.txt', 'CTN053_Day6_18.txt', 'CTN053_Day6_2.txt']
    plot4(filename, shortname_to_points, '(B) CTN053, Tumor, Day6', file_list)

    filename = os.path.join(args.outdir,"ctn094_day6.pdf")
    file_list = ['CTN094_Day6_1.txt', 'CTN094_Day6_10.txt', 'CTN094_Day6_14.txt', 'CTN094_Day6_20.txt']
    plot4(filename, shortname_to_points, '(C) CTN094, Tumor, Day6', file_list)
    
    for (myfile, mylist) in [
        #('byname.pdf', big_results)
        #('bysum.pdf', big_results_by_sum),
        #('bysum1.pdf', big_results_by_sum1),
        #('bysum2.pdf', big_results_by_sum2),
        #('bycirc.pdf', big_results_by_circ)
        ]:
        
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
                plt.xlabel('x coordinate')
                plt.ylabel('y coordinate')
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
                        


if __name__ == "__main__":
    main()

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

def get_basenames(indir):
    basenames = [ ]
    for file in sorted(os.listdir(indir)):
        if fnmatch.fnmatch(file, '*.txt'):
            basename = os.path.splitext(file)[0]
            basenames.append(basename)
    logger.info('files: %s', ' '.join(basenames))
    return(basenames)

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
    # final normalization so that 0-frequency component is <x>
    ret = (1.0 / float(n)) * ret
    # and set the zero-freq component (the mean) to zero
    ret[0,:] = 0.0
    #logger.info('start of transform: %s', str(ret[0:5,:]))
    #logger.info('end of transform: %s', str(ret[-5:,:]))
    return(ret)
    
def get_power(points_rfft):
    nfreq = points_rfft.shape[0]
    dim = points_rfft.shape[1]
    power = np.zeros(nfreq)
    for k in range(dim):
        a = np.abs( points_rfft[:,k] )
        power = power + a**2
    # the first element of power should be <x>^2 + <y>^2 + ..., can set to zero
    # the last element of power should probably be doubled
    logger.info('power: %s ... %s', str(power[0:10]), str(power[-5:]))
    return(power)
    
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
    basenames = get_basenames(args.indir)

    # check that the output directory exists; if not, create it
    if (not os.path.isdir(args.outdir)):
        logger.info('creating output directory %s', args.outdir)
        os.makedirs(args.outdir)
    
    big_results = [ ]
    for basename in basenames:
        infile = os.path.join(args.indir, basename + '.txt')
        # read the xy pairs from the input file
        # format is an array of tuples
        points = file_to_points(infile, DIM)
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
        power_rfft = get_power(points_rfft)
        power_norm = power_rfft[1]
        power_scaled = (1.0 / power_norm) * power_rfft
        power_scaled[1] = 0.0
        power_sum = sum(power_scaled)
        freq = range(len(power_scaled))
        power_moment = sum(freq * power_scaled)
        power_mean = (power_moment / power_sum) if (power_sum > 0) else 0.0
        big_results.append( (basename, points, power_scaled, power_sum, power_moment, power_mean, circularity) )
        
        continue # no diagnostic plots
    
        pltfile = os.path.join(args.outdir, basename + '.pdf')
        with PdfPages(pltfile) as pdf:

            plt.figure()
            plt.plot(points[:,0], points[:,1])
            plt.title('%s: Raw Image' % basename)
            pdf.savefig()
            plt.close()
            
            plt.figure()
            plt.plot(contour_length)
            plt.title('%s: Contour Length' % basename)
            pdf.savefig()
            plt.close()
            
            plt.figure()
            plt.plot(contour_length, points[:,0], 'k-')
            plt.plot(contour_length, points[:,1], 'k--')
            plt.plot(contour_grid, points_grid[:,0], 'g-')
            plt.plot(contour_grid, points_grid[:,1], 'g--')
            plt.title('%s: Parametric Interpolation' % basename)
            pdf.savefig()
            plt.close()
            
            plt.figure()
            # for power spectrum, only the top components
            xx = range(2,21)
            #plt.plot(xx, np.abs( points_rfft[xx,0] )**2, 'b-' )
            #plt.plot(xx, np.abs( points_rfft[xx,1] )**2, 'b--' )
            plt.plot(xx, power_scaled[xx])
            plt.title('%s: Power Spectrum' % basename)
            pdf.savefig()
            plt.close()
            
            plt.figure(figsize=(10,5))

            plt.subplot(121)
            plt.plot(points[:,0], points[:,1])
            plt.xlabel('x coordinate')
            plt.ylabel('y coordinate')
            plt.title('%s: Raw Image' % basename)           
            
            plt.subplot(122)
            xx = range(2,21)
            plt.plot(xx, power_scaled[xx])
            plt.xlabel('Harmonic component')
            plt.ylabel('Scaled power')
            plt.title('%s: Power Spectrum' % basename)
            
            pdf.savefig()
            
            plt.close()
    
    
    big_results_by_sum = sorted(big_results, key = lambda tup: (tup[3], tup))
    big_results_by_moment = sorted(big_results, key = lambda tup: (tup[4], tup))
    big_results_by_mean = sorted(big_results, key = lambda tup: (tup[5], tup))
    big_results_by_circ = sorted(big_results, key = lambda tup: (1.0 - tup[6], tup))
    
    for (myfile, mylist) in [ ('byname.pdf', big_results),
        ('bysum.pdf', big_results_by_sum),
        ('bymoment.pdf', big_results_by_moment),
        ('bymean.pdf', big_results_by_mean),
        ('bycirc.pdf', big_results_by_circ) ]:
        
        logger.info('plotting %s', myfile)
    
        pltfile = os.path.join(args.outdir, myfile)
        with PdfPages(pltfile) as pdf:
            
            for (basename, points, power_scaled, power_sum, power_moment, power_mean, circ) in mylist:
                logger.info('  %s', basename)
            
                plt.figure(figsize=(12,5))
    
                plt.subplot(121)
                plt.plot(points[:,0], points[:,1])
                plt.xlabel('x coordinate')
                plt.ylabel('y coordinate')
                plt.title('%s: Raw Image' % basename)           
                
                plt.subplot(122)
                xx = range(2,21)
                plt.plot(xx, power_scaled[xx])
                # plt.title('%s: Power Spectrum' % basename)
                plt.title('Power: sum %.3f, mom %.3f, mean %.1f, circ %.3f' %
                          (power_sum, power_moment, power_mean, circ))
                (x1,x2,y1,y2) = plt.axis()
                y2 = max(y2, 0.1)
                plt.axis((x1,x2,y1,y2))
                plt.xlabel('Harmonic component')
                plt.ylabel('Scaled power')
                
                pdf.savefig()
                plt.close()        
                        


if __name__ == "__main__":
    main()

#!/usr/bin/env python

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
from matplotlib.patches import Ellipse

from skimage.restoration import unwrap_phase

import statsmodels.api as sm

from PIL import Image, ImageDraw

from scipy import ndimage
from skimage.morphology import disk, dilation, watershed, closing, skeletonize, medial_axis

import logging

logging.basicConfig(format='%(levelname)s %(name)s.%(funcName)s: %(message)s')
logger = logging.getLogger('stack_images')
logger.setLevel(logging.INFO)


def get_files_recursive(indir, pattern_str):
    ret = [ ]
    for root, dirnames, filenames in os.walk(indir):
        for filename in fnmatch.filter(filenames, pattern_str):
            ret.append( os.path.join(root, filename) )
    return(ret)

def read_file(filename):
    f = open(filename, 'r')
    data = dict()
    keys = f.readline().split('\t')
    for k in range(0,len(keys)):
        keys[k] = keys[k].strip('\n')
        data[keys[k]] = []

    for l in f:
        for k in range(0,len(keys)):
            data[keys[k]].append(l.split('\t')[k])
    return data

def plot_ellipse(sp, x, y, xlabel, ylabel):
    ids = list(range(len(x)))
    ids = np.array(ids).astype(np.str)
    x = np.array(x).astype(np.float)
    y = np.array(y).astype(np.float)
    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    ax = plt.subplot(sp)
    plt.xlim(-max(x)/1.25, max(x)*1.25)
    plt.ylim(-max(y)/1.25, max(y)*1.25)
    for j in xrange(1, 6):
        ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=lambda_[0]*j*2, height=lambda_[1]*j*2,
                      angle=np.rad2deg(np.arccos(v[0, 0])), color='black')
        ell.set_facecolor('none')
        ax.add_artist(ell)
    for (x1,y1,id1) in zip (x, y, ids):
        plt.text(x1,y1,id1, va='center', ha='center')
    (r, pval) = pearsonr(x, y)
    rsq = r*r
    plt.title('Rsq = %.3f, pval = %.3g' % (rsq, pval))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return None

def main():
    parser = argparse.ArgumentParser(description='Plot From Data File', epilog='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--indir', help='input directory', default='.', required=True)
    parser.add_argument('--outdir', help='output directory', default='.', required=True)
    parser.add_argument('--filename', help='output file name', default='Summary_Plot', type=str, required=False)
    args = parser.parse_args()

    files = get_files_recursive(args.indir, "*.txt")

    outfile = os.path.join(args.outdir, args.filename)
    pdf = PdfPages(outfile)
    plt.figure()
    rows = len(files)

    for f in files:
        plt.figure(figsize=(25,10))
        plt.suptitle(f.split('/')[-1])
        data = read_file(f)
        sps = [241, 242, 243, 244, 245, 246, 247]
        for (sp, xname) in zip( sps, ('K14 Sum Peripheral Pixels', 'Fractional Area', 'K14 Sum Central Pixels', 'K14 Total Sum', 'K14 Total Mean', 'K14 Peripheral Mean', 'K14 Central Mean') ):
            plot_ellipse(sp, data[xname], data['Invasion'], xname, 'Invasion')
        pdf.savefig()
        plt.close()
    plt.tight_layout()
    pdf.close()             
    
    return None

if __name__ == "__main__":
    main()




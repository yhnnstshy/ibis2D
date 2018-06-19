#!/usr/bin/env python
import argparse
import os
import fnmatch
from shutil import copyfile

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

def main():
    parser = argparse.ArgumentParser(description='copy files and rename systematically',
                                     epilog='Sample call: see copyfiles.sh',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('indir', help='input directory, with .txt files with 2 columns, no header')
    parser.add_argument('outdir', help='output directory')
    args = parser.parse_args()
    
    logger.info('indir %s outdir %s', args.indir, args.outdir)
    basenames = get_basenames(args.indir)

    # check that the output directory exists; if not, create it
    if (not os.path.isdir(args.outdir)):
        logger.info('creating output directory %s', args.outdir)
        os.makedirs(args.outdir)
    
    for basename in basenames:
        infile = os.path.join(args.indir, basename + '.txt')
        toks = basename.split('_')
        assert(len(toks) == 3), 'bad tokens: ' + infile
        (sample, day, number) = toks
        newnum = '%02d' % int(number)
        newbase = '_'.join([sample, day, newnum])
        outfile = os.path.join(args.outdir, newbase + '.txt')
        logger.info('%s -> %s', infile, outfile)
        copyfile(infile, outfile)

if __name__ == "__main__":
    main()

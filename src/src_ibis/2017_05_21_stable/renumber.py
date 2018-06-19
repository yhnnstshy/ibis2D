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
import shutil

import numpy as np
from scipy import interpolate

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as mpcm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import logging
logging.basicConfig(format='%(levelname)s %(name)s.%(funcName)s: %(message)s')
logger = logging.getLogger('ibis2d')
logger.setLevel(logging.INFO)

MAXK = 21

#def get_files_recursive(indir, pattern_str):
def get_files_recursive(indir, exclude = [ ]):
    ret = [ ]
    for root, dirnames, filenames in os.walk(indir):
        #  for filename in fnmatch.filter(filenames, pattern_str):
        for f in filenames:
            fullname = os.path.join(root, f)
            keep = True
            for mystr in exclude:
                if mystr in fullname:
                    keep = False
                    break
            if keep:
                ret.append( fullname )
    return(ret)

def check_uniq(all_files):
    d = dict()
    f_to_root = dict()
    for a in all_files:
        (r, f) = os.path.split(a)
        d[f] = d.get(f,0) + 1
        f_to_root[f] = f_to_root.get(f, '') + ' ' + r
    ks = sorted(d.keys())
    n_error = 0
    for k in ks:
        if d[k] != 1:
            logger.info('bad file count: %s %s', k, f_to_root[k])
            n_error += 1
    logger.info('%d errors in file count', n_error)
    assert(n_error == 0)
    return(True)
    
def get_new_names(all_files, pad):
    new_names = [ ]
    for a in all_files:
        (r, f) = os.path.split(a)
        f = f.replace(' ','') # remove internal spaces
        (base_str, ext_str) = os.path.splitext(f)
        toks = base_str.split('_')
        num_str = toks[-1]
        logger.info('%s %s', a, num_str)
        new_num_str = num_str
        if num_str.isdigit():
            new_num_str = num_str.zfill(pad)
        toks[-1] = new_num_str
        new_name = '_'.join(toks) + ext_str
        new_names.append(new_name)
        logger.info('%s %s', f, new_name)
    return(new_names)
    
def main():
    
    parser = argparse.ArgumentParser(description='renumber files',
                                     epilog='Sample call: see renumber.sh',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--indir', help='input directory', required=True)
    parser.add_argument('--outdir', help='output directory', required=True)
    parser.add_argument('--pad', help='zero padding length', required=True, type=int)
    args = parser.parse_args()
    
    logger.info('indir %s outdir %s pad %d', args.indir, args.outdir, args.pad)
    assert(args.pad >= 0)
    all_files = get_files_recursive(args.indir, exclude = ['Special', 'zip', 'DS_Store'])
    logger.info('allfiles: %s', str(all_files))
    
    check_uniq(all_files)
    new_names = get_new_names(all_files, args.pad)

    # check that the output directory exists; if not, create it
    if (not os.path.isdir(args.outdir)):
        logger.info('creating output directory %s', args.outdir)
        os.makedirs(args.outdir)
    
    logger.info('copying to %s', args.outdir)
    new_paths = [ os.path.join(args.outdir, f) for f in new_names ]
    for (src_str, dest_str) in zip(all_files, new_paths):
        shutil.copy2(src_str, dest_str) # copies metadata also

    exit(1)
    
if __name__ == "__main__":
    main()

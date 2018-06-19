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

def read_table(filename, delim='\t'):
    logger.info('reading table from %s', filename)
    fp = open(filename, 'r')
    ret = dict()
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
    logger.info('%s: %d rows, %d columns', filename, len(ret), ntok)
    return(ret)

def main():
    
    veenafile = 'Veena_Tumor_Day6.txt'
    srcdir = '../IMAGES_DIC_Tumor_Day6_original_names/CTN024_K14_DIC'
    destdir = '../IMAGES_DIC_Tumor_Day6_original_names/CTN024_DIC_K14'
    ctn024 = 'CTN024'

    if not os.path.isdir(srcdir):
        logger.info('%s does not exist', srcdir)
        return None
    if not os.path.isdir(destdir):
        logger.info('%s does not exist', destdir)

    veenatable = read_table(veenafile)

    files = sorted(veenatable.keys())
    for f in files:
        if veenatable[f]['CTN'] != ctn024:
            continue
        srcpath = os.path.join(srcdir, f)
        destpath = os.path.join(destdir, f)
        if not os.path.isfile(srcpath):
            logger.info('%s does not exist', srcpath)
            continue
        img_orig = mpimg.imread(srcpath)
        img_new = np.copy(img_orig)
        width = img_orig.shape[1]
        ny = width / 2
        img_new[:,0:ny,...] = img_orig[:,ny:width,...]
        img_new[:,ny:width,...] = img_orig[:,0:ny,...]
        mpimg.imsave(destpath, img_new)
        logger.info('swapped %s to %s', srcpath, destpath)
    
    # copy_files_recursive(args.indir, args.outdir, exclude)
    
    #for (src_str, dest_str) in zip(all_files, new_paths):
    #   shutil.copy2(src_str, dest_str) # copies metadata also

    exit(1)
    
if __name__ == "__main__":
    main()

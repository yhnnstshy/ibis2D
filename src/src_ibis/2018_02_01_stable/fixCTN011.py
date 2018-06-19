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

def main():
    
    mydir = '../IMAGES_Tumor_Day6_DIC_renamed'
    mybase = 'CTN011_Day6'
    myoffset = 50
    pad = 2
    
    if not os.path.isdir(mydir):
        logger.info('%s does not exist', mydir)
        return None

    for num in range(1,19):
        origname = "%s_Set2_%s.tif" % (mybase, str(num).zfill(pad))
        newname = "%s_%s.tif" % (mybase, str(num + myoffset).zfill(pad))
        origfull = os.path.join(mydir, origname)
        if not os.path.isfile(origfull):
            logger.info('%s missing', origfull)
            continue
        newfull = os.path.join(mydir, newname)
        if os.path.isfile(newfull):
            logger.info('%s already exists', newfull)
            continue
        logger.info('copying %s to %s', origfull, newfull)
        shutil.copy2(origfull, newfull)
    
    # copy_files_recursive(args.indir, args.outdir, exclude)
    
    #for (src_str, dest_str) in zip(all_files, new_paths):
    #   shutil.copy2(src_str, dest_str) # copies metadata also

    exit(1)
    
if __name__ == "__main__":
    main()

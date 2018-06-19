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
import logging

logging.basicConfig(format='%(levelname)s %(name)s.%(funcName)s: %(message)s')
logger = logging.getLogger('circle')
logger.setLevel(logging.INFO)

def main():
    
    parser = argparse.ArgumentParser(description='make a circle',
                                     epilog='Sample call: see circle.sh',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n', help='number of points', type=int, required=False, default=128)
    parser.add_argument('--r', help='radius', required=False, type=float, default=1.0)
    args = parser.parse_args()
    
    logger.info('n %d r %f', args.n, args.r)
    
    r = args.r
    n = args.n
    for i in range(args.n):
        theta = 2.0 * math.pi * float(i)/float(n)
        print "%f\t%f" % (r * math.cos(theta), r * math.sin(theta))
    
    exit(1)
    
if __name__ == "__main__":
    main()

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

import logging
logging.basicConfig(format='%(levelname)s %(name)s.%(funcName)s: %(message)s')
logger = logging.getLogger('ibis2d')
logger.setLevel(logging.INFO)

def main():

    myroot = '../IMAGES_ALL/Normal Mammary/CTN096_Normal_Day6'
    mybase = 'CTN096_Normal_Day6'
    for i in range(1, 22):
        mysuffix = ('%02d' % i) + '.tif'
        oldname = os.path.join(myroot, mybase + mysuffix)
        newname = os.path.join(myroot, mybase + '_' + mysuffix)
        logger.info('copying %s to %s', oldname, newname)
        shutil.copy2(oldname, newname) # copies metadata also

    exit(1)
    
if __name__ == "__main__":
    main()

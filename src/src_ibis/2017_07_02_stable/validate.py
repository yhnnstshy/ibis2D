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

def validate_ctn(ctnstr):
    errcnt = 0
    if (len(ctnstr) != 6):
        errcnt += 1
    if (ctnstr[0:3] != 'CTN'):
        errcnt += 1
    if (not ctnstr[3:].isdigit()):
        errcnt += 1
    return(errcnt == 0)

def validate_annot(annotstr):
    errcnt = 0
    if (annotstr not in ['Tumor', 'NAT', 'Normal']):
        errcnt += 1
    return(errcnt == 0)
    
def validate_daystr(daystr):
    errcnt = 0
    if (daystr not in ['Day0', 'Day6']):
        errcnt += 1
    return(errcnt == 0)
    
def validate_numstr(numstr):
    errcnt = 0
    if (len(numstr) < 2) or (len(numstr) > 3):
        errcnt += 1
    if (not numstr.isdigit()):
        errcnt += 1
    return(errcnt == 0)

def fixctnstr(ctnstr):
    if (ctnstr == 'CTN0015'):
        ctnstr = 'CTN015'
    elif (ctnstr == 'CTN036 '):
        ctnstr = 'CTN036'
    elif (ctnstr == 'CTN05'):
        ctnstr = 'CTN005'
    return(ctnstr)
    
def fixdaystr(daystr):
    if (daystr == 'Day 0'):
        daystr = 'Day0'
    elif (daystr == 'Day 6'):
        daystr = 'Day6'
    return(daystr)

PAD = 2
def fixnumstr(numstr):
    if (len(numstr) != PAD) and (numstr.isdigit()):
        numstr = str(int(numstr)).zfill(PAD)
    return(numstr)

def stradd(numstr, intval):
    assert(numstr.isdigit()), 'trying to add to non-integer %s' % numstr
    numstr = str( int(numstr) + intval )
    return(numstr)

ANNOTS = ['Normal', 'NAT', 'Tumor']
SET2ADD = 50
def validate_file(root, basename, ext, outdir):
    # coordinate filename: CTN###_ANNOT_Day#_##.txt
    # K14 filename: CTN###_ANNOT_Day#_K14_##.tif
    # ANNOT is Normal or NAT, missing for Tumor
    fullname = os.path.join(root, basename + ext)
    toks = basename.split('_')
    toks = [ t.strip() for t in toks ]
    (ctnstr, annotstr, daystr, k14str, numstr) = ('', '', '', '', '')
    if ext == '.txt':
        # logger.info('validating %s %s %s toks %s len %d', filename, str(toks), len(toks))
        if len(toks) == 3:
            (ctnstr, daystr, numstr) = toks
            annotstr = 'Tumor'
        elif len(toks) == 4:
            if (toks[2] == 'Set2'): # some have ctn - day - set2 - num
                toks = [ toks[0], 'Tumor', toks[1], toks[3] ]
                toks[3] = stradd(toks[3], SET2ADD)
            (ctnstr, annotstr, daystr, numstr) = toks
        else:
            logger.warn('bad token count: %s', filename)
    elif ext == '.tif':
        # fix some errors
        if (toks[1] == 'Day6K14'):
            toks = [ toks[0], 'Day6', 'K14' ] + toks[2:] 
        if (toks[0][:3] == 'CTn'):
            toks[0] = toks[0].upper()
        if len(toks) == 2:
            if (toks[0] in ['CTN094', 'CTN102']) and (toks[1][:4] == 'Day0') and ('Tumor' in root):
                toks = [ toks[0], 'Tumor', toks[1][:4], 'K14', toks[1][4:] ]
        # check for merged K14 and number string
        if (toks[-1][:3] == 'K14') and (toks[-1][3:].isdigit()):
            toks = toks[:-1] + [ toks[-1][:3], toks[-1][3:] ]
        # check for merged day and number
        if (toks[-1][:3] == 'Day') and (toks[-1][3:].isdigit()):
            toks = toks[:-1] + [ toks[-1][:4], toks[-1][4:] ]
        if len(toks) == 3:
            if (toks[0] == 'CTN085') and (toks[1] == 'Day6') and ('Tumor' in root):
                toks = [ toks[0], 'Tumor', toks[1], 'K14', toks[2] ]
        # check for missing k14
        if ('K14' not in toks) and ('k14' not in toks):
            toks = toks[:-1] + ['K14',  toks[-1]]
        if len(toks) == 4:
            (ctnstr, daystr, k14str, numstr) = toks
            if ('Tumor' in root):
                annotstr = 'Tumor'
        elif len(toks) == 5:
            if (toks[2] == 'Set2') and ('Tumor' in root):
                toks = [ toks[0], 'Tumor', toks[1], toks[3], toks[4] ]
                toks[4] = stradd(toks[4], SET2ADD)
            (ctnstr, annotstr, daystr, k14str, numstr) = toks
            if ((ctnstr == 'CTN036') and (annotstr == 'Day6') and (daystr == 'NAT')):
                (annotstr, daystr) = (daystr, annotstr)
        if (k14str == 'k14'):
            k14str = 'K14'
        if (k14str != 'K14'):
            logger.info('bad k14 str: %s', fullname)
            return(False)
    else:
        logger.info('bad extention: %s', fullname)
        return(False)

    ctnstr = fixctnstr(ctnstr)
    daystr = fixdaystr(daystr)
    numstr = fixnumstr(numstr)
        
    errlist = [ ]
    if not (daystr in root):
        errlist.append('day mismatch: %s' % fullname)
    if not validate_ctn(ctnstr):
        errlist.append('badctn %s' % ctnstr)
    if not validate_annot(annotstr):
        errlist.append('badannot %s' % annotstr)
    if not validate_daystr(daystr):
        errlist.append('badday %s' % daystr)
    if not validate_numstr(numstr):
        errlist.append('badnum %s' % numstr)

    fields = [ctnstr, annotstr, daystr, numstr]
    if (ext == '.tif'):
        fields.append('K14')
    elif (ext == '.txt'):
        fields.append('xy')
    newname = '_'.join(fields) + ext
    newfullname = os.path.join(outdir, newname)
    if os.path.exists(newfullname):
        errlist.append('duplicate %s %s' % (fullname, newfullname))
        
    if (len(errlist) > 0):
        logger.warn('file %s has errors %s', fullname, str(errlist))
        return(False)

    shutil.copy2(fullname, newfullname)
    return(True)
    

#def get_files_recursive(indir, pattern_str):
def validate_files_recursive(indir, outdir, exclude):
    (npass, nfail, nexclude) = (0, 0, 0)
    for root, dirnames, filenames in os.walk(indir):
        #  for filename in fnmatch.filter(filenames, pattern_str):
        for f in filenames:
            fullname = os.path.join(root, f)
            (basename, ext) = os.path.splitext(f)
            if (ext not in ['.txt', '.tif']):
                nexclude += 1
                continue
            cnt = sum( [ x in fullname for x in exclude ])
            if (cnt > 0):
                logger.info('excluding %s', fullname)
                nexclude += 1
                continue
            isgood = validate_file(root, basename, ext, outdir)
            if (isgood):
                npass += 1
            else:
                nfail += 1
    logger.info('npass %d nfail %d nexclude %d', npass, nfail, nexclude)

def main():
    
    parser = argparse.ArgumentParser(description='validate coordinate and image files',
                                     epilog='Sample call: see validate.sh',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--indir', help='input directory', required=True)
    parser.add_argument('--outdir', help='output directory', required=True)
    args = parser.parse_args()
    
    exclude = ['Special', 'Day6_DIC']
    logger.info('indir %s outdir %s exclude %s',
                args.indir, args.outdir, str(exclude))
    
    if (not os.path.isdir(args.outdir)):
        logger.info('creating output directory %s', args.outdir)
        os.makedirs(args.outdir)

    # check that outdir is empty
    previous_contents = os.listdir(args.outdir)
    assert(len(previous_contents) == 0), '%s must be empty or absent' % args.outdir
    
    validate_files_recursive(args.indir, args.outdir, exclude)
    
    #for (src_str, dest_str) in zip(all_files, new_paths):
    #   shutil.copy2(src_str, dest_str) # copies metadata also

    exit(1)
    
if __name__ == "__main__":
    main()

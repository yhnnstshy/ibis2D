#!/bin/bash


source /Users/ytsehay/work/py_env/bin/activate

DATAFOLDER='/Users/ytsehay/work/Data_Final/Human'

time python ../../src/ibis2d.py --recalculate y --thermometers n --veenafile Veena_Orig.txt --matchfile Match_File.txt --dic_orig ${DATAFOLDER}/Images --outdir outdir --coords ${DATAFOLDER}/XY --nfft 256

deactivate

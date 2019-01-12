#!/bin/bash

DATASET="PyMT"
o="1"
RUN="convolution_demo"
DAYNUM=5

source /Users/ytsehay/work/py_env/bin/activate

DATAFOLDER='/Users/ytsehay/work/Data_Final/'${DATASET}

SUBDIRNAME=mouse_${o}_Day${DAYNUM}

python /Users/ytsehay/work/project/src/ibis2d_mine.py --datafolder ${DATAFOLDER} --images ${DATAFOLDER}/Images/${SUBDIRNAME} --coords ${DATAFOLDER}/XY/${SUBDIRNAME} --${RUN} y



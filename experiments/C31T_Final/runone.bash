#!/bin/bash


if [ $# -ne 2 ];
then
	echo "Usage: Organoid_Number Organoid_Day"
	exit 1
fi

ORGNUM=$1
DAYNUM=$2

source /Users/ytsehay/work/py_env/bin/activate

DATAFOLDER='/Users/ytsehay/work/Data_Final/C31T'
SUBDIRNAME=mouse_${ORGNUM}_Day${DAYNUM}

python /Users/ytsehay/work/project/src/ibis2d_mine.py --datafolder ${DATAFOLDER} --images ${DATAFOLDER}/Images/${SUBDIRNAME} --coords ${DATAFOLDER}/XY/${SUBDIRNAME} --outdir ${SUBDIRNAME} --invasionvsk14 y

deactivate

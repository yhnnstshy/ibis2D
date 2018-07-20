#!/bin/bash


if [ $# -ne 3 ];
then
	echo "Usage: Organoid_Number Organoid_Size[Small|Large] Organoid_Day"
	exit 1
fi

ORGNUM=$1
ORGSIZE=$2
DAYNUM=$3

source /Users/ytsehay/work/py_env/bin/activate

DATAFOLDER='/Users/ytsehay/work/data_2'
SUBDIRNAME=${ORGNUM}_${ORGSIZE}Orgs_Day${DAYNUM}

python /Users/ytsehay/work/project/src/ibis2d_mine.py --datafolder ${DATAFOLDER} --images ${DATAFOLDER}/K14/${SUBDIRNAME} --coords ${DATAFOLDER}/XY/${SUBDIRNAME} --outdir ${SUBDIRNAME} --calculate y

deactivate
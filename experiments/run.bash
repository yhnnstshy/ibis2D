#!/bin/bash


if [ $# -ne 3 ];
then
	echo "Usage: Organoid_Number Organoid_Size[small|large] Organoid_Day"
	exit 1
fi

ORGNUM=$1
ORGSIZE=$2
DAYNUM=$3

source /Users/ytsehay/work/py_env/bin/activate

DATAFOLDER='/Users/ytsehay/work/data_2'

python /Users/ytsehay/work/project/src/ibis2d_mine.py --datafolder ${DATAFOLDER} --images ${DATAFOLDER}/Image --coords ${DATAFOLDER}/XY --outdir output --calculate y --orgnum ${ORGNUM} --orgsize ${ORGSIZE} --daynum ${DAYNUM}

deactivate

#!/bin/bash

if [ $# -lt 4 ];
then
	echo "Usage: DATASET [PyMT|C31T] MOUSE [1|2|3|4] IMAGENAME [Image1|Image2...] RUN [plots|figures] ORGLIST [1 2 3 ...]"
	exit 1
fi

RUN="generate_${4}"
DATASET=$1
o=$2
IMAGENAME=$3
ORGS=($(echo "${@:5}"))
DAYNUM=5


rm -r tmp

source /Users/ytsehay/work/py_env/bin/activate

DATAFOLDER='/Users/ytsehay/work/Data_Final/'${DATASET}

mkdir -pv tmp

SUBDIRNAME=mouse_${o}_Day${DAYNUM}

for org in ${ORGS[@]};
do
	cp ${DATAFOLDER}/XY/${SUBDIRNAME}/"${org}.txt" tmp
done

python /Users/ytsehay/work/project/src/ibis2d_mine.py --datafolder ${DATAFOLDER} --images ${DATAFOLDER}/Images/${SUBDIRNAME} --coords tmp  --outdir ${SUBDIRNAME} --${RUN} y

rm -r tmp

mv ${SUBDIRNAME}.pdf ${IMAGENAME}.pdf

#if [ ${RUN} == 'invasionvsk14' ];
#then
#	python /Users/ytsehay/work/project/src/generate_summary_plots.py --indir "." --outdir "."
#fi

deactivate

#!/bin/bash


if [ $# -ne 1 ];
then
	echo "Usage: RUN [invasionvsk14|calculate]"
	exit 1
fi

RUN=$1
ORGSIZE=('Small Large')
DAYNUM=5


source /Users/ytsehay/work/py_env/bin/activate

DATAFOLDER='/Users/ytsehay/work/Data_Final/PyMT'

for o in `seq 1 4`;
do
	SUBDIRNAME=mouse_${o}_Day${DAYNUM}
	python /Users/ytsehay/work/project/src/ibis2d_mine.py --datafolder ${DATAFOLDER} --images ${DATAFOLDER}/Images/${SUBDIRNAME} --coords ${DATAFOLDER}/XY/${SUBDIRNAME} --outdir ${SUBDIRNAME} --${RUN} y
done

if [ ${RUN} == 'invasionvsk14' ];
then
	python /Users/ytsehay/work/project/src/generate_summary_plots.py --indir "." --outdir "."
fi

deactivate

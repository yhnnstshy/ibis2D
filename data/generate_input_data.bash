#!/bin/bash


if [ $# -ne 2 ];
then
	echo "Usage: RawDataFolder OutputDirectory[absolute path]"
	exit 1
fi


DATAFOLDER=$1
OUTDIR=$2
DATATYPE=('DIC K14 XY')
source ~/work/py_env/bin/activate


cd ${DATAFOLDER}


for d in ${DATATYPE[@]};
do
	./organize_data.bash ${d} tmp
done

python ~/work/project/src/ibis2d_mine.py --datafolder tmp --outdir ${OUTDIR}/Images --combineimgs y

for d in ${DATATYPE[@]};
do
	cp -r tmp/${d} ${OUTDIR}
done

rm -r tmp

echo "Done!"

#!/bin/bash


if [ $# -ne 2 ];
then
	echo "Usage: RawDataFolder OutDataFolder"
	exit 1
fi


RAWDATAFOLDE=$1
OUTDATAFOLDER=$2

echo "Info: Organizing input data ..."

cp organize_data.bash ${RAWDATAFOLDE}

./generate_input_data.bash ${RAWDATAFOLDE} ${OUTDATAFOLDER}

echo "Done!"

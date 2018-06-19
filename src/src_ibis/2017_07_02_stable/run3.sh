/bin/rm -rf ../analysis_3
python ./ibis2d.py --coords ../COORDS_3 --images ../IMAGES_3 --outdir ../analysis_3 --nfft 256 --recalculate y

#python ./ibis2d.py --coords ../COORDS_2017_05_16 --images ../IMAGES_2016_07_26 --outdir ../analysis_2017_05_16_256 --nfft 256 --plotby name --recalculate n

#python ./ibis2d.py --coords ../COORDS_tmp/ --images ../IMAGES_tmp --outdir ../analysis_2017_05_16 --nfft 1024 --plotby name
#python ./ibis2d.py --coords ../Coord_HuTu_2016_07_28/ --images ../IMAGES_2016_07_26 --outdir ../analysis_2017_02_15 --nfft 1024 --plotby name
# python ./organoid_stats.py ../analysis_2017_02_15

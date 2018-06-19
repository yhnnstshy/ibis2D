time python ./ibis2d.py --veenafile Veena_Orig.txt --matchfile Veena_Matches.txt --dic_orig ../IMAGES_Tumor_Day6_DIC_K14_original --outdir ../autoveena_2017_12_14 --coords ../COORDS_VALID --nfft 256

#time python ./ibis2d.py --veenafile Veena_Orig.txt --matchfile Veena_Matches.txt --invasionfile invasion_table.txt --k14file k14_table.txt --dic_orig ../IMAGES_Tumor_Day6_DIC_K14_original --outdir ../autoveena_2017_10_23 --coords ../COORDS_VALID --nfft 256

#time python ./ibis2d.py --invasionfile invasion_table.txt --dic_orig ../IMAGES_Tumor_Day6_DIC_K14_original --outdir ../autoveena_2017_10_23 --coords ../COORDS_VALID --nfft 256

#time python ./ibis2d.py --veenafile Veena_Orig.txt --invasionfile invasion_table.txt --dic_orig ../IMAGES_Tumor_Day6_DIC_K14_original --dic_renamed ../IMAGES_Tumor_Day6_DIC_renamed --outdir ../autoveena_2017_10_23 --coords ../COORDS_VALID --images ../IMAGES_VALID --nfft 256 --recalculate n --thumbnails n --thermometers n

#time python ./ibis2d.py --coords ../COORDS_VALID --images ../IMAGES_VALID --outdir ../analysis_2017_07_02_zscore --nfft 256 --recalculate n --thumbnails n --thermometers y


# time python ./ibis2d.py --coords ../COORDS_106 --images ../IMAGES_VALID --outdir ../analysis_106 --nfft 256 --recalculate n --thumbnails y --thermometers n


#time python ./ibis2d.py --coords ../COORDS_VALID --outdir ../analysis_2017_06_28 --nfft 256 --recalculate y

#time python ./ibis2d.py --coords ../COORDS_CIRCLE --outdir ../analysis_circle --nfft 256 --recalculate y --thumbnails n

#time python ./ibis2d.py --coords ../COORDS_106 --outdir ../analysis_106 --nfft 256 --recalculate y --thumbnails n

#time python ./ibis2d.py --coords ../COORDS_106 --images ../IMAGES_VALID --outdir ../analysis_106 --nfft 256 --recalculate y --thumbnails n

#time python ./ibis2d.py --coords ../COORDS_VALID --images ../IMAGES_VALID --outdir ../analysis_2017_05_22_figs --nfft 256 --plotby name --recalculate n

#python ./ibis2d.py --coords ../COORDS_2017_05_16 --images ../IMAGES_2016_07_26 --outdir ../analysis_2017_05_16_256 --nfft 256 --plotby name --recalculate n

#python ./ibis2d.py --coords ../COORDS_tmp/ --images ../IMAGES_tmp --outdir ../analysis_2017_05_16 --nfft 1024 --plotby name
#python ./ibis2d.py --coords ../Coord_HuTu_2016_07_28/ --images ../IMAGES_2016_07_26 --outdir ../analysis_2017_02_15 --nfft 1024 --plotby name
# python ./organoid_stats.py ../analysis_2017_02_15

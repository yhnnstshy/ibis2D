#!/usr/bin/env python
import argparse
import os
import fnmatch
import math

import numpy as np
from scipy import interpolate
from scipy import stats as spstats

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import logging
logging.basicConfig(format='%(levelname)s %(name)s.%(funcName)s: %(message)s')
logger = logging.getLogger('ibis2d')
logger.setLevel(logging.INFO)

def plot_organoids(myfile, key_organoid_data, persons, groups, days):
    logger.info('plotting %s', myfile)
    with PdfPages(myfile) as pdf:
        for p in persons:
            for g in groups:
                color = 'r'
                if (g == 'Normal'):
                    color = 'g'
                elif (g == 'NAT'):
                    color = 'orange'
                for d in days:
                    k = (p, g, d)
                    if (k not in key_organoid_data):
                        continue
                    logger.info('... %s', str(k))
                    
                    sumvec = [ ]
                    sum1vec = [ ]
                    sum2vec = [ ]
                    noncircvec = [ ]
                    for org in key_organoid_data[k].keys():
                        sumvec.append( key_organoid_data[k][org]['sum'])
                        sum1vec.append( key_organoid_data[k][org]['sum1'])
                        sum2vec.append( key_organoid_data[k][org]['sum2'])
                        noncircvec.append( key_organoid_data[k][org]['noncirc'])
            
                    plt.figure(figsize=(12,12))
    
                    plt.subplot(221)
                    plt.scatter(sumvec, sum1vec, c=color)
                    plt.xlabel('sum')
                    plt.ylabel('sum1')
                    plt.title('%s' % str(k))           
                
                    plt.subplot(222)
                    plt.scatter(sum2vec, noncircvec, c=color)
                    plt.xlabel('sum2')
                    plt.ylabel('noncirc')
                    plt.title('%s' % str(k))           

                    plt.subplot(223)
                    plt.scatter(sumvec, sum2vec, c=color)
                    plt.xlabel('sum')
                    plt.ylabel('sum2')
                    plt.title('%s' % str(k))           
                
                    plt.subplot(224)
                    plt.scatter(sumvec, noncircvec, c=color)
                    plt.xlabel('sum')
                    plt.ylabel('noncirc')
                    plt.title('%s' % str(k))           

                    
                    pdf.savefig()
                    plt.close() 

def plot_groupday_organoid(myfile, key_organoid_data, groups, days, stats):
    logger.info('plotting %s', myfile)
    groupdays = [ ]
    for g in groups:
        for d in days:
            groupdays.append((g, d))
    groupday_stats = dict()
    for k in groupdays:
        groupday_stats[k] = dict()
        for s in stats:
            groupday_stats[k][s] = [ ]
    for (p, g, d) in key_organoid_data:
        gd = (g, d)
        for org in key_organoid_data[(p,g,d)]:
            for s in stats:
                groupday_stats[gd][s].append(key_organoid_data[(p,g,d)][org][s])
                
    # for each day, for each stat, make a combined histogram of Normal, NAT, Tumor
    grouporder = ['Tumor','NAT', 'Normal']
    groupcolor = ['r', 'orange', 'g']
    with PdfPages(myfile) as pdf:
        for d in days:
            for s in stats:
                logger.info('%s %s', d, s)
                plt.figure(figsize=(5,5))
                plt.subplot(111)
                x = [ ]
                for g in grouporder:
                    x.append( groupday_stats[ (g, d) ][s] )
                plt.hist(x,bins=20,normed=True,color=groupcolor,label=grouporder)
                plt.legend()
                plt.xlabel(s)
                plt.ylabel('Probability')
                plt.title('%s %s' % (d,s))           
                pdf.savefig()
                plt.close() 

def plot_groupday_person(myfile, pgd_data, groups, days, stats, moments):
    logger.info('plotting %s', myfile)
    gdsm_list = [ ]
    for g in groups:
        for d in days:
            for s in stats:
                for m in moments:
                    gdsm_list.append( (g, d, s, m) )
    gdsm_data = dict()
    for k in gdsm_list:
        gdsm_data[k] = [ ]
    for (p, g, d) in pgd_data:
        for s in stats:
            for m in moments:
                gdsm_data[(g,d,s,m)].append( pgd_data[(p,g,d)][s][m])
                
    # for each day, for each stat, make a combined histogram of Normal, NAT, Tumor
    grouporder = ['Tumor','NAT','Normal']
    groupcolor = ['r', 'orange', 'g']
    with PdfPages(myfile) as pdf:
        for d in days:
            for s in stats:
                # for m in moments:
                for m in ['mean']:
                    logger.info('%s %s %s', d, s, m)
                    for g in grouporder:
                        mymean = sum(gdsm_data[(g,d,s,m)])/len(gdsm_data[(g,d,s,m)])
                        logger.info('%s %s %s %s mean %f', d, s, m, g, mymean)
                    for (g1, g2) in [ ('Tumor', 'NAT'), ('Tumor','Normal'), ('NAT','Normal') ]:
                        (tstat, pval) = spstats.ttest_ind(gdsm_data[(g1,d,s,m)], gdsm_data[(g2,d,s,m)])
                        logger.info('%s vs %s for %s %s %s: t = %f, p = %f',
                                    g1, g2, d, s, m, tstat, pval)
                    plt.figure(figsize=(5,5))
                    plt.subplot(111)
                    x = [ ]
                    for g in grouporder:
                        x.append( gdsm_data[(g, d, s, m)] )
                    plt.hist(x,bins=10,normed=False,color=groupcolor,label=grouporder)
                    plt.legend()
                    plt.xlabel('%s' % s)
                    plt.ylabel('Count')
                    plt.title('%s %s' % (d,s))           
                    pdf.savefig()
                    plt.close()
 
 
def plot_tumor_day6_histogram(myfile, pgd_data):
    logger.info('plotting %s', myfile)
    data_list = [ ]
    for (p, g, d) in pgd_data:
        if (g != 'Tumor'):
            continue
        if (d != 'Day6'):
            continue
        data_list.append( pgd_data[(p,g,d)]['sum']['mean'])
                
    with PdfPages(myfile) as pdf:
        plt.figure(figsize=(10,6))
        plt.subplot(111)
        plt.hist(data_list,bins=12,normed=False,color='k',facecolor='gray')
        # plt.legend()
        plt.xlabel('%s' % 'Spectral power, individual mean')
        plt.ylabel('Number of individuals')
        plt.title('(A) Heterogeneity of invasiveness')           
        pdf.savefig()
        plt.close()
                    
# scatter plot
# for each group, for each stat, for each moment, plot day 6 vs day 0 for each person
def plot_personscatter(myfile, pgd_data, groups, days, stats):
    logger.info('plotting %s', myfile)
    moments = ['mean']
    gsm_list = [ ]
    for g in groups:
        for s in stats:
            for m in moments:
                gsm_list.append( (g, s, m) )

    gsm_day0 = dict()
    gsm_day6 = dict()
    gsm_person = dict()
    for gsm in gsm_list:
        gsm_day0[gsm] = [ ]
        gsm_day6[gsm] = [ ]
        gsm_person[gsm] = [ ]
        
    for (p, g, d) in pgd_data:
        if (d != 'Day0'):
            continue
        if (p, g, 'Day6') not in pgd_data:
            continue
        for s in stats: 
            for m in ['mean']:  
                gsm = (g, s, m )
                gsm_day0[gsm].append(pgd_data[(p,g,'Day0')][s][m])
                gsm_day6[gsm].append(pgd_data[(p,g,'Day6')][s][m])
                gsm_person[gsm].append(p)
    
    with PdfPages(myfile) as pdf:
        for g in groups:
            plt.figure(figsize=[12,5])
            
            plt.subplot(121)
            gsm = (g, 'sum', 'mean')
            if gsm not in gsm_list:
                continue
            plt.scatter(gsm_day0[gsm], gsm_day6[gsm])
            for i in range(len(gsm_day0[gsm])):
                plt.text(gsm_day0[gsm][i], gsm_day6[gsm][i], gsm_person[gsm][i])
            plt.xlabel('Day 0')
            plt.ylabel('Day 6')
            (r, p) = spstats.pearsonr(gsm_day0[gsm],gsm_day6[gsm])
            plt.title('%s %s r=%.2f p=%.2f' % (g,'Power', r, p))
            
            
            plt.subplot(122)
            gsm = (g, 'noncirc', 'mean')
            if gsm not in gsm_list:
                continue
            plt.scatter(gsm_day0[gsm], gsm_day6[gsm])
            for i in range(len(gsm_day0[gsm])):
                plt.text(gsm_day0[gsm][i], gsm_day6[gsm][i], gsm_person[gsm][i])            
            plt.xlabel('Day 0')
            plt.ylabel('Day 6')
            (r, p) = spstats.pearsonr(gsm_day0[gsm],gsm_day6[gsm])
            plt.title('%s %s r=%.2f p=%.2f' % (g,'Non-Circ', r, p))
            
            pdf.savefig()
            plt.close()
            
# scatter plot
# for day 0 and day 6, for sum power and non-circ, plot tumor vs nat
def plot_tumornatscatter(myfile, pgd_data):
    logger.info('plotting %s', myfile)
    days = ['Day0', 'Day6']
    stats = ['sum', 'noncirc']
    ds_list = [ ]
    for d in days:
        for s in stats:
            ds_list.append( (d, s) )

    ds_tumor = dict()
    ds_nat = dict()
    ds_person = dict()
    for ds in ds_list:
        ds_tumor[ds] = [ ]
        ds_nat[ds] = [ ]
        ds_person[ds] = [ ]
        
    for (p, g, d) in pgd_data:
        if (g != 'Tumor'):
            continue
        if (p, 'NAT', d) not in pgd_data:
            continue
        m = 'mean'
        for s in stats:
            ds = (d, s)
            ds_tumor[ds].append(pgd_data[(p,'Tumor',d)][s][m])
            ds_nat[ds].append(pgd_data[(p,'NAT',d)][s][m])
            ds_person[ds].append(p)
    
    with PdfPages(myfile) as pdf:
        for ds in ds_list:
            
            n = len(ds_nat[ds])
            logger.info('%s has %d examples', str(ds), n)
            if (n < 2):
                continue

            plt.figure(figsize=[8,6])
            
            plt.subplot(111)
            plt.scatter(ds_nat[ds], ds_tumor[ds])
            for i in range(len(ds_nat[ds])):
                plt.text(ds_nat[ds][i], ds_tumor[ds][i], ds_person[ds][i])
            plt.xlabel('NAT')
            plt.ylabel('Tumor')
            (r, p) = spstats.pearsonr(ds_nat[ds],ds_tumor[ds])
            plt.title('%s %s r=%.2f p=%.2f' % (ds[0], ds[1], r, p))

            pdf.savefig()
            plt.close()

def groupday_anova(gd_data):
    gd_list = sorted(gd_data.keys())
    for gd in gd_list:
        # logger.info('anova for %s', str(gd))
        #for p in gd_data[gd].keys():
        #    logger.info('%s %s', p, str(gd_data[gd][p]))
        args = [ ]
        for p in gd_data[gd]:
            args.append( np.array(gd_data[gd][p]) )
        #fstat = spstats.f_oneway(*gd_data[gd].values())
        #kw = spstats.kruskal(*gd_data[gd].values())
        fstat = spstats.f_oneway(*args)
        kw = spstats.kruskal(*args)
        logger.info('%s anova results: %s %s', str(gd), str(fstat), str(kw))
                
def main():
    parser = argparse.ArgumentParser(description='Convert organoid data to individual data',
                                     epilog='Sample call: see run.sh',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('analysisdir', help='analysis directory')
    args = parser.parse_args()
    
    logger.info('analysisdir %s', args.analysisdir)

    # check that the output directory exists; if not, create it
    assert(os.path.isdir(args.analysisdir)), 'analysis directory missing'
    
    # read the table of organoid data
    organoid_table = os.path.join(args.analysisdir, 'organoid_table.txt')
    fp = open(organoid_table, 'r')
    header = fp.readline().strip()
    logger.info('header: %s', header)
    organoid_list = [ ]
    for line in fp:
        toks = line.strip().split('\t')
        assert(len(toks) == 7)
        organoid_list.append(toks)
    fp.close()
    logger.info('%d organoids', len(organoid_list))
    
    GROUPS = ['Normal', 'NAT', 'Tumor']
    DAYS = ['Day0', 'Day6']
    STATS = ['sum', 'sum1', 'sum2', 'noncirc']
    MOMENTS = ['cnt', 'mean', 'sd']

    person_dict = dict()
    
    # the key is (person, group, day) to match the file naming
    key_organoid_data = dict()
    for row in organoid_list:
        (fullpath, dirname, filename, sum, sum1, sum2, circ) = row
        (grouppath, personday) = os.path.split(dirname)
        # logger.info('personday %s', personday)
        # (person, day) = personday.split('_')
        toks = personday.split('_')
        person = toks[0]
        day = toks[-1]
        if (len(toks) == 3):
            group = toks[1]
        else:
            group = 'Tumor'
        person_dict[person] = True
        logger.info('group %s person %s day %s file %s', group, person, day, fullpath)
        assert(group in GROUPS), 'bad group'
        assert(day in DAYS), 'bad day'
        mykey = (person, group, day)
        if mykey not in key_organoid_data:
            key_organoid_data[mykey] = dict()
        key_organoid_data[mykey][fullpath] = dict();
        key_organoid_data[mykey][fullpath]['sum'] = float(sum)
        key_organoid_data[mykey][fullpath]['sum1'] = float(sum1)
        key_organoid_data[mykey][fullpath]['sum2'] = float(sum2)
        key_organoid_data[mykey][fullpath]['noncirc'] = 1.0 - float(circ)

    persons = sorted(person_dict.keys())    

    # now summarize all the organoids for each (p, g, d)
    MOMENTS = ['cnt', 'mean', 'sd']
    MINORG = 10
    pgd_data = dict()
    gd_data = dict()
    for k in key_organoid_data:
        norg = len(key_organoid_data[k].keys())
        if (norg < MINORG):
            logger.info('%s has %d organoids, skipping', str(k), norg)
            continue
        logger.info('%s has %d organoids ...', str(k), norg)
        pgd_data[k] = dict()
        (person, group, day) = k
        gd = (group, day)
        if gd not in gd_data:
            gd_data[gd] = dict()
        gd_data[gd][person] = [ ]
        for org in key_organoid_data[k]:
            gd_data[gd][person].append( key_organoid_data[k][org]['sum'] )
        for s in STATS:
            pgd_data[k][s] = dict()
            for m in MOMENTS:
                pgd_data[k][s][m] = 0
        for s in STATS:
            cnt = 0.0
            sum = 0.0
            sumsq = 0.0
            for org in key_organoid_data[k]:
                x = key_organoid_data[k][org][s]
                cnt += 1.0
                sum += x
                sumsq += x*x
            variance = 0.0
            mean = 0.0
            if (cnt > 0):
                mean = sum / cnt
            if (cnt > 1):
                variance = (sumsq - ((sum * sum)/cnt))/(cnt - 1.0)
            sd = math.sqrt(variance)
            pgd_data[k][s]['cnt'] = cnt    
            pgd_data[k][s]['mean'] = mean    
            pgd_data[k][s]['sd'] = sd    
    
    # diagnostic plots for each person
    #organoid_filename = os.path.join(args.analysisdir, 'organoids_by_person.pdf')
    #plot_organoids(organoid_filename, key_organoid_data, persons, GROUPS, DAYS)

    # nat vs tumor
    # tumornatscatter_filename = os.path.join(args.analysisdir, 'tumornat_scatter.pdf')
    # plot_tumornatscatter(tumornatscatter_filename, pgd_data)
    
    # organoid histograms
    # groupday_filename = os.path.join(args.analysisdir, 'groupday_organoid_histogram.pdf')
    # plot_groupday_organoid(groupday_filename, key_organoid_data, GROUPS, DAYS, STATS)
    
    # person histograms
    # personday_filename = os.path.join(args.analysisdir, 'groupday_person_histogram.pdf')
    # plot_groupday_person(personday_filename, pgd_data, GROUPS, DAYS, STATS, MOMENTS)
    tumorday6file = os.path.join(args.analysisdir, 'tumor_day6_histogram.pdf')
    plot_tumor_day6_histogram(tumorday6file, pgd_data)
    groupday_anova(gd_data)
    
    # person scatter, day 0 vs day 6
    # personscatter_filename = os.path.join(args.analysisdir, 'person_scatter.pdf')
    # plot_personscatter(personscatter_filename, pgd_data, GROUPS, DAYS, ['sum','noncirc'])
    
    filename = os.path.join(args.analysisdir, 'pgd_data.txt')
    fp = open(filename, 'w')
    fp.write('\t'.join(['person','group','day','cnt','sum','sd','noncirc'])+'\n')
    for p in persons:
        for g in GROUPS:
            for d in DAYS:
                k = (p,g,d)
                if k not in pgd_data:
                    continue
                ptr = pgd_data[k]
                fp.write('%s\t%s\t%s\t%d\t%f\t%f\t%f\n' % (p,g,d,ptr['sum']['cnt'],
                                                           ptr['sum']['mean'],
                                                           ptr['sum']['sd'],
                                                           ptr['noncirc']['mean']))
    fp.close()
    


if __name__ == "__main__":
    main()

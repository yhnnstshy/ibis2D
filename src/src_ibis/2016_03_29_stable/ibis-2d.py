#!/usr/bin/env python
import argparse

import logging
logging.basicConfig(format='%(levelname)s %(name)s.%(funcName)s: %(message)s')
logger = logging.getLogger('ibis-2d')
logger.setLevel(logging.INFO)


    
def main():
    parser = argparse.ArgumentParser(description='Invasive boundary image score, 2D',
                                     epilog='Sample call: see run.sh',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('infile', help='input file, 2 columns, no header')
    parser.add_argument('outfile', help='output file')
    parser.add_argument('n', help='number of points for FFT', type=int, default=4096)
    args = parser.parse_args()
    
    logger.info('infile %s outfile %s n %d', args.infile, args.outfile, n)


if __name__ == "__main__":
    main()

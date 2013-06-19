#!/usr/bin/python

'''Created by Daniel Sindhikara, sindhikara@gmail.com
Program converts various volumetric data formats.

For now only inputs UxDATA and TINKER guv files.
For now only outputs .dx files.
'''  

from __future__ import division

import grid.grid as grid
import sys
import argparse
# from math import *
# import matplotlib as plt
# import pylab as pl
# import numpy as np
# import os

# <codecell>


# <codecell>

def main():
    parser = \
        argparse.ArgumentParser(description='Converts volumetric data.'
                                )
    parser.add_argument('inputfile', type=str, help='Input volumetric data file')
    parser.add_argument('-iuvdata', action="store_true", default=False, \
                        help='Inputfile is UxDATA file.')
    parser.add_argument('-itguv', action="store_true", default=False, \
                        help='Inputfile is TINKER guv file.')
    parser.add_argument('--disttypes', type=str, \
                         help='Type of distributions to output. E.g. '+\
                              '"--disttypes g" or "--disttypes gct"',\
                         default = 'g')
    
    args = parser.parse_args()
    print "Will write newfiles as extensions of original filename."
    
    if args.iuvdata and args.itguv:
        exit("Error, pick only one input format!")
    
    if args.iuvdata:
        grids = grid.data2Grids(args.inputfile, disttypes=args.disttypes)
        for gridname, mygrid in grids.iteritems():
            outname = args.inputfile + '.' + gridname + '.dx'
            mygrid.writedx(outname)
    elif args.itguv:
        grids = grid.TKRguv2Grids(args.inputfile)
        for gridnum, mygrid in enumerate(grids):
            outname = args.inputfile + '.' + str(gridnum) + '.dx'
            mygrid.writedx(outname)


if __name__ == '__main__':
    main()


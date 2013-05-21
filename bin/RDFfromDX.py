#!/usr/bin/env python
import grid.grid as grid
import sys
print "# Input format: RDFfromDX.py <dxfilename> <x> <y> <z> <delta> <max>"

dxfilename, x, y, z, delta, maxval = sys.argv[1:]
griddata = grid.dx2Grid(dxfilename)
rdf = griddata.interpRDF([float(x),float(y),float(z)], float(delta), float(maxval))
for i, value in enumerate(rdf):
    print i*float(delta), value

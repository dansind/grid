#!/usr/bin/env python
import grid.grid as grid
import numpy as np
import os

datadir = os.path.join(os.path.dirname(grid.__file__),"tests","data")
# Test reading/writing
print "Testing reading/writing dx files.."
dxfilename = os.path.join(datadir, "dxfiles", 
                          "AlaDP_3DRISM_smallbuffer.dx.gz") 
originalgrid = grid.dx2Grid(dxfilename)

maxvalue = originalgrid.distribution.max()
maxindices = tuple(np.argwhere(originalgrid.distribution == maxvalue)[0])
originalgrid.distribution[maxindices] = maxvalue * 2.0
originalgrid.writedx("modified.dx")
newgrid = grid.dx2Grid("modified.dx")
newmax = newgrid.distribution.max()
assert newmax == maxvalue * 2.0
#cleanup
os.remove("modified.dx")

# Test shells
print "Testing shell-related utilities.."
storedprecomputedshells = grid.readshellindices()
newshells = grid.precomputeshellindices(40)
for storedshell, newshell in zip(storedprecomputedshells, newshells):
    for storedpoint, newpoint in zip(storedshell, newshell):
        for storedindex, newindex in zip(storedpoint, newpoint):
            assert storedindex == newindex



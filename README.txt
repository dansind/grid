===========
Grid
===========

Grid contains objects and functions pertaining to 3D volumetric (grid) data.
It was specifically written for 3D-RISM distribution data but is
by no means limited to such data. 
Grid requires numpy.

Typical usage often looks like this::

    #!/usr/bin/env python

    import grid

    mygrid = grid.dx2Grid("mydxfilename")



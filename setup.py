try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Volumetric data module',
    'long_description': open('README.txt').read(),
    'author': 'Daniel J. Sindhikara',
    #'url':'https://github.com/dansind/grid',
    #'download_url': 'https://github.com/dansind/grid/tarball/master',
    'author_email': 'sindhikara@gmail.com',
    'version': '0.4.4',
    'install_requires': ['numpy', 'argparse', ], #
    #'py_modules': ['grid'], 
    'packages': ["grid","grid.tests"],
    'package_data': {'grid': ['data/shells.json', 'data/points/*', "tests/data/dxfiles/*"], 
                     "grid.tests": ["data/dxfiles/*", "data/UxDATAfiles/*", "data/TKRguv/*"]},
    'scripts': ["grid/tests/grid_tests.py", "bin/RDFfromDX.py", "bin/gridconvert.py"],
    'license': 'LGPL',
    'name': 'grid'
}

setup(**config)  

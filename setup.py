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
    'version': '0.1.8',
    'install_requires': ['numpy'], #
    #'py_modules': ['grid'], 
    'packages': ["grid","grid.tests"],
    'package_data': {'grid': ['data/shells.json', 'data/points/*', "tests/data/dxfiles/*"], 
                     "grid.tests": ["data/dxfiles/*"]},
    'scripts': ["grid/tests/grid_tests.py", "bin/RDFfromDX.py"],
    'license': 'LGPL',
    'name': 'grid'
}

setup(**config)  

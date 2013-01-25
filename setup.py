try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Volumetric data module',
    'long_description': open('README.txt').read(),
    'author': 'Daniel J. Sindhikara',
    'url':'www.dansindhikara.com',
    # 'download_url':'Where to download it.',
    'author_email': 'sindhikara@gmail.com',
    'version': '0.1',
    'install_requires': ['numpy'], #
    #'py_modules': ['grid'], 
    'packages': ["grid","grid.tests"],
    'package_data': {'grid': ['data/shells.json', "tests/data/dxfiles/*"], 
                     "grid.tests": ["data/dxfiles/*"]},
    'scripts': ["grid/tests/grid_tests.py"],
    'license': 'LGPL',
    'name': 'grid'
}

setup(**config)  

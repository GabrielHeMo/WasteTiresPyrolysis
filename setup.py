# -*- coding: utf-8 -*-
# CUWP: Chemical Upcycling of Waste Plastics Process Models
# Copyright (C) 2025-2027, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the MIT open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/CUWP/blob/master/LICENSE.txt
# for license details.
from setuptools import setup

setup(
    name='pyrolysis',
    packages=['pyrolysis'],
    license='BSD3',
    version='0.0.1',
    description="Pyrolysis of waste tires",
    long_description=open('README.md', encoding='utf-8').read(),
    author='Yoel Cortes-Pena',
    install_requires=[
        'biosteam==2.52.20',
        'thermosteam==0.52.18',
        'biorefineries',
        'thermo==0.6.0',
        'fluids==1.3.0',
        'chemicals==1.5.0',
        'flexsolve==0.5.10',
        'pyomo==6.10.0',
        'ipopt==1.0.3',
    ],
    python_requires=">=3.12",
    package_data={
        'pyrolysis': [
            'pyrolysis',
            'pyrolysis/*',
        ]
    },
    platforms=['Windows', 'Mac', 'Linux'],
    author_email='yoelcortes@gmail.com',
    url='https://github.com/GabrielHeMo/WasteTiresPyrolysis',
    download_url='https://github.com/GabrielHeMo/WasteTiresPyrolysis',
    classifiers=['Development Status :: 3 - Alpha',
                 'Environment :: Console',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.12',
                 'Programming Language :: Python :: 3.13',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Chemistry',
                 'Topic :: Scientific/Engineering :: Mathematics'],
    keywords='pyrolysis chemical process simulation plastic thermochemical conversion',
)

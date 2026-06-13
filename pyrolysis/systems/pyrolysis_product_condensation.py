# -*- coding: utf-8 -*-
"""
"""
import pyrolysis
import biosteam as bst
import numpy as np


__all__ = (
    'create_pyrolysis_product_condensation_system',
)

@bst.SystemFactory(
    ins=[dict(ID='pyrolysis_product')],
    outs=[dict(ID='syngas'),
          dict(ID='naphtha'),
          dict(ID='fuel_oil')]
)
def create_pyrolysis_product_condensation_system(ins, outs):
    feed, = ins
    syngas, naphtha, fuel_oil = outs
    F1 = bst.Flash(ins=feed, outs=['', fuel_oil], T=160 + 273.15, P=101325 * 0.2)
    F2 = bst.Flash(ins=F1-0, outs=[syngas, naphtha], T=30 + 273.15, P=101325 * 0.2)

def test_pyrolysis_product_condensation_system():
    from pyrolysis import create_chemicals
    bst.settings.set_thermo(create_chemicals(), pkg='ideal gas')
    feed = bst.Stream(phase='g', T=500 + 273.15)
    IDs = (
        'Rubber',
         'Ash',
         'Char',
         'Ca(OH)2',
         'CaSO4',
         'SO2',
         'carbon',
         'molecular hydrogen',
         'molecular nitrogen',
         'sulfur',
         'molecular oxygen',
         'oxidane',
         'methane',
         'ethane',
         'ethene',
         'propane',
         'propene',
         'butane',
         'but-1-ene',
         'but-1-yne',
         'carbon dioxide',
         'carbon monoxide',
         'hydrogen sulfide',
         'hexane',
         'pentane',
         'pent-2-yne',
         '1-methylcyclopentene',
         '3-methylcyclopentene',
         '2,5-dimethylhexa-1,5-diene',
         '2,2,3-trimethylpentane',
         '1,1-dimethylcyclopentane',
         '2,3,4-trimethylpentane',
         '3,4-dimethylhexane',
         'ethylcyclopentane',
         'methylcyclohexane',
         '1,1-dimethylcyclohexane',
         'oct-1-ene',
         'ethylcyclohexane',
         '1,1,3-trimethylcyclohexane',
         'non-1-ene',
         '2-methyloct-1-ene',
         '(4R)-1-methyl-4-prop-1-en-2-ylcyclohexene',
         '2,6,6-trimethylbicyclo[3.1.1]hept-2-ene',
         'Limonene',
         'benzene',
         'toluene',
         'ethylbenzene',
         '1,3-xylene',
         'styrene',
         'cumene',
         '1-ethyl-4-methylbenzene',
         'propylbenzene',
         '1-ethyl-2-methylbenzene',
         '1-ethyl-3-methylbenzene',
         '1,2,3-trimethylbenzene',
         'phenol',
         'benzonitrile',
         'prop-1-enylbenzene',
         '1-methyl-4-propan-2-ylbenzene',
         '2,3-dihydro-1H-indene',
         '1H-indene',
         '1-ethyl-2,3-dimethylbenzene',
         'm-Cymene',
         '1-ethyl-2,4-dimethylbenzene',
         'o-Cymene',
         '5-methyl-2,3-dihydro-1H-indene',
         '1,2,3,5-tetramethylbenzene',
         '1,2,3,4-tetramethylbenzene',
         '1-ethyl-2-propan-2-ylbenzene',
         '2,3-dimethylphenol',
         '1-methyl-2,3-dihydro-1H-indene',
         'benzoic acid',
         '2-methyl-2,3-dihydro-1H-indene',
         '1-methyl-1H-indene',
         '1,2,3,4-tetrahydronaphthalene',
         '4-methyl-2,3-dihydro-1H-indene',
         'naphthalene',
         '4-propan-2-ylphenol',
         'benzothiazole',
         '5-methyl-1,2,3,4-tetrahydronaphthalene',
         'hexylbenzene',
         '1-methylnaphthalene',
         '2-methylnaphthalene',
         '1,2,3-trimethyl-1H-indene',
         "1,1'-biphenyl",
         '1-ethylnaphthalene',
         '2-ethylnaphthalene',
         '1,8-dimethylnaphthalene',
         '1,5-dimethylnaphthalene',
         '2,7-dimethylnaphthalene',
         '2-methylquinoline',
         'tetradec-1-ene',
         'pentadecane',
         '1,2,3-trimethylnaphthalene',
         'butylbenzene',
         '1,2,5-trimethylnaphthalene',
         '9H-fluorene',
         '2,4-dimethyl-1-phenylbenzene',
         'pentadec-1-ene',
         'hexadecane',
         'anthracene',
         '4-methylphenanthrene',
         'pentadecanoic acid',
         '3-methylphenanthrene',
         '2-methylphenanthrene',
         'nonadecane',
         '1-methyl-7-propan-2-ylphenanthrene',
         'icosane',
         'henicosane',
         'docosane',
         'tetracosane',
         'undecane',
         'ammonia',
         'cyclohexane'
    )
    feed.imol[IDs] = np.array([
        0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
        0.000e+00, 8.595e-05, 5.046e-07, 0.000e+00, 6.464e-07, 5.204e+00,
        7.176e+00, 5.101e-01, 3.511e+00, 2.038e-01, 4.375e-01, 7.180e-02,
        1.059e-01, 1.538e-01, 7.467e-01, 6.344e-01, 2.396e+00, 4.904e-01,
        5.542e-01, 2.587e-02, 7.902e-02, 7.902e-02, 1.072e-01, 7.018e-01,
        5.885e-01, 5.798e-01, 1.343e+00, 3.139e-01, 1.765e+00, 3.433e-02,
        2.403e-02, 2.094e-01, 9.154e-03, 5.187e-02, 5.004e-01, 1.921e-02,
        3.512e-02, 3.397e-01, 1.591e-02, 5.271e-02, 2.718e-02, 3.209e-02,
        2.921e-02, 5.147e-03, 4.060e-03, 1.596e-02, 1.309e-02, 1.329e-02,
        2.040e-03, 4.781e-02, 1.051e-02, 3.273e-03, 1.587e-02, 5.322e-03,
        9.221e-03, 1.394e-03, 3.171e-03, 1.637e-03, 2.795e-03, 2.482e-03,
        1.579e-03, 1.649e-03, 7.143e-04, 1.824e-02, 3.651e-03, 5.282e-02,
        1.777e-02, 3.111e-03, 2.703e-03, 2.083e-03, 1.257e-01, 2.694e-03,
        2.388e-01, 1.020e-01, 8.022e-02, 3.775e-01, 4.735e-01, 2.269e-01,
        4.294e-01, 2.448e-01, 9.404e-02, 2.143e-01, 2.101e-01, 3.016e-01,
        1.066e+00, 7.981e-01, 3.363e-01, 1.917e-01, 5.043e-03, 6.293e-02,
        1.443e-01, 1.555e-01, 1.126e-01, 2.601e-01, 1.149e-01, 1.962e-01,
        4.031e+00, 1.180e-01, 2.065e-01, 5.784e-02, 8.662e-02, 6.114e-02,
        6.665e-02, 4.940e-02, 1.144e-02, 2.809e-01, 0.000e+00, 0.000e+00
    ])
    condensation_sys = create_pyrolysis_product_condensation_system(ins=feed)
    condensation_sys.simulate()
    # No formal tests here. Just make sure it simulates
    # The 
    
if __name__ == '__main__':
    test_pyrolysis_product_condensation_system()
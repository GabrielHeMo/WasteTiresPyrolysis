import biosteam as bst
import pandas as pd 
import numpy as np 
from scipy.integrate import solve_ivp
from chemicals.utils import R
from numba import njit
import os 

__all__ = (
    'PyrolysisFractionator', 
)

class PyrolysisFractionator(bst.MESHDistillation):
    """
    """
    
    def _init(self,  
            N_stages=5, 
            feed_stages=[2], 
            P=[101325 * 0.2 + 690.0 * i for i in range(5)], # Vacuum operation to not overheat and decompose heavier fraction 
            vapor_side_draws=[(3, 0.8)],
            stage_specifications={
                0: ('T', 40 + 273.15),
                -1: ('T', 200 + 273.15),
            },
            **kwargs,
        ):
        super()._init(
            N_stages=N_stages, 
            feed_stages=feed_stages, 
            P=P,
            vapor_side_draws=vapor_side_draws,
            stage_specifications=stage_specifications,
            **kwargs)

    
def test_pyrolysis_fractionator():
    from pyrolysis import create_chemicals, PyrolysisReactor
    bst.settings.set_thermo(create_chemicals(), pkg='ideal gas')
    feed = bst.Stream(Tire=6250, units='kg/hr')
    R1 = PyrolysisReactor(ins=feed, tau=14)
    R1.simulate()
    vapor, metals, char, *_ = R1.outs
    vapor.vle(T=160 + 273.15, P=0.2 * 101325 + 690.0 * 4)
    PF = PyrolysisFractionator(ins=vapor, outs=('syngas', 'heavy_oil', 'naptha'))
    PF.sequential_runs_init = 0
    PF.convergence_analysis(algorithm='inside out', legend=False)
    PF.algorithms = ('inside out', 'simultaneous correction')
    PF.methods = ('fixed-point', 'hybr')
    PF.simulate()
    
if __name__ == '__main__':
    test_pyrolysis_fractionator()
import biosteam as bst
from typing import Optional
from biosteam.units.decorators import cost

__all__ = (
    'Hydrotreater', 
)

class HydrogenationReactions:
    __slots__ = ('reactions',)
    
    class HydrogenationReaction:
        __slots__ = ('atom', 'product', 'hydrogen_coefficient', 'product_coefficient')
        def __init__(self, atom, product, hydrogen_coefficient, product_coefficient):
            self.atom = atom
            self.product = product
            self.hydrogen_coefficient = hydrogen_coefficient
            self.product_coefficient = product_coefficient

        def _info(self, header=True):
            line = f"{self.atom} + "
            if self.hydrogen_coefficient == 1:
                line += 'H2 -> '
            else:
                line += f'{self.hydrogen_coefficient:.5g} H2 -> '
            if self.product_coefficient == 1:
                line += self.product
            else:
                line += f'{self.product_coefficient:.5g} {self.product}'
            return f"{type(self).__name__}:\n" + line if header else line

        def show(self):
            print(self._info())
        
        _ipython_display_ = show
    
    def __init__(self, reactions=None):
        self.reactions = [] if reactions is None else reactions

    def add(self, atom, product, hydrogen_coefficient, product_coefficient):
        self.reactions.append(
            self.HydrogenationReaction(atom, product, hydrogen_coefficient, product_coefficient)
        )
        
    def __call__(self, atoms):
        products = {}
        H2_consumption = 0
        for i in self.reactions:
            value = atoms.get(i.atom, 0)
            products[i.product] = value * i.product_coefficient + products.get(i.product, 0)
            H2_consumption += value * i.hydrogen_coefficient
        products['H2'] = atoms['H'] / 2 - H2_consumption
        return products
    
    def _info(self):
        return f"{type(self).__name__}:\n" + '\n'.join([i._info(header=False) for i in self.reactions])

    def show(self):
        print(self._info())
    
    _ipython_display_ = show
    
# # Traditional requirement
# def hydrotreater_hydrogen_requirement(feed): # H2 kmol / hr
#     
#     atoms = feed.get_atomic_flows()
#     # C1HaObNcSd + x H2 -> C1H2.1 + b H2O + c NH3 + d H2S
#     stoichiometry = [
#         ('C', 2.1),
#         ('H', -1),
#         ('O', 2),
#         ('N', 3),
#         ('S', 2),
#     ]
#     # Amount of hydrogen consumed is 10% above stoichiometric amount due to 
#     # hydrocracking.
#     return  1.1 * sum([j * atoms.get(i, 0) for i, j in stoichiometry]) / 2.

# DO NOT DELETE: This code computes the cost coefficients from real data
# # cost = Cb * S ** n
# # log(cost) = n log(S) + log(Cb)
# import numpy as np
# processing_capacity = np.array([2500, 6250]) * 907.185 / 24 * 0.39 * (1 - 0.23) # kg / hr
# capital_cost = np.array([3780000 + 1970000 + 110000/2, 7970000 + 4020000 + 196000/2])
# logcost = np.log(capital_cost)
# logS = np.log(processing_capacity)
# n, logCb = np.polyfit(logS, logcost, 1)
# Cb = np.exp(logCb)

@cost('Flow rate', units='kg/hr',
      cost=1581.8954434501595, CE=567.3,
      n=0.8005031419670899, S=1, BM=1)
class Hydrotreater(bst.Unit):
    """
    Create a Gibbs equilibrium reactor for hydrotreating pyrolysis oil.
    
    Parameters
    ----------
    ins : 
        * [0] Raw pyrolysis oil.
        * [1] Hydrogen produced on-site.
    outs :  
        Vapor product.
    T : float   
        Operating temperature [K].
    P : float   
        Operating pressure [bar].
    
    """
    _N_ins = 2
    _N_outs = 1
    _ins_size_is_fixed = False
    _outs_size_is_fixed = True
    
    def _init(self, 
            T: Optional[float]=None,
            P: Optional[float]=None,
            product_distribution: Optional[dict[str, float]]=None,
            H2_efficiency: Optional[float]=None,
        ):
        self.T = 350 + 273.15 if T is None else T 
        self.P = 100 * 101325 if P is None else P
        self.H2_efficiency = 0.99 if H2_efficiency is None else H2_efficiency
        if product_distribution is None:
            naphtha = 55
            diesel_LHO = 1630 + 158
            total = naphtha + diesel_LHO
            product_distribution = {
                'cyclohexane': naphtha / total,
                'dodecane': diesel_LHO / total,
            }
        self.reactions = reactions = HydrogenationReactions()
        reactions.add('O', 'H2O', 1, 1)
        reactions.add('N', 'NH3', 1.5, 1)
        reactions.add('S', 'H2S', 1, 1)
        chemicals = self.chemicals
        for ID, x in product_distribution.items():
            atoms = chemicals[ID].atoms
            product_coefficient = x / atoms['C']
            hydrogen_coefficient = atoms['H'] / 2 * product_coefficient
            reactions.add('C', ID, hydrogen_coefficient, product_coefficient)
    
    def hydrogen_requirement(self, stream):
        atoms = stream.get_atomic_flows()
        products = self.reactions(atoms)
        H2_consumption = max(-products['H2'], 0)
        H2_needed = H2_consumption / self.H2_efficiency
        return H2_needed
    
    def _run(self):
        feed, hydrogen = ins = self.ins
        product, = self.outs
        hydrogen.phase = product.phase = 'g'
        product.T = self.T
        product.P = self.P
        product.mix_flows(ins)
        atoms = product.get_atomic_flows()
        products = self.reactions(atoms)
        if products['H2'] < 0: raise ValueError('not enough hydrogen')
        product.empty()
        for i, j in products.items(): product.imol[i] = j
        
    def _design(self):
        self.design_results['Flow rate'] = self.ins[0].F_mass
        self.add_heat_utility(self.Hnet, self.T)
    

def test_hydrotreater():
    import pyrolysis
    import numpy as np
    bst.settings.set_thermo(pyrolysis.create_chemicals(), pkg='ideal gas')
    pyrolysis_oil = bst.Stream(
        ID=None, Rubber=1, phase='g', T=433.15, P=101325, 
    )
    hydrogen = bst.Stream(phase='g')
    HT = pyrolysis.Hydrotreater(
        ins=[pyrolysis_oil, hydrogen], 
    )
    hydrogen.imol['H2'] = HT.hydrogen_requirement(pyrolysis_oil)
    HT.simulate()
    np.testing.assert_allclose(
        HT.outs[0].imol['H2', 'H2O', 'H2S', 'NH3', 'cyclohexane', 'dodecane'],
        [0.0004115731955977864,
         0.001950940212281082,
         0.0005408089961773817,
         0.0002476106500807149,
         0.0003590566217275541,
         0.005836302178626062],
        atol=1e-3, rtol=1e-2,
    )

if __name__ == '__main__':
    test_hydrotreater()
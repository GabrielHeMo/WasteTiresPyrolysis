import biosteam as bst
from biosteam.units.design_tools.Gibbs_equilibrium_reaction import minimize_Gibbs_free_energy
from typing import Optional, Iterable
from biosteam.units.decorators import cost

__all__ = (
    'HydrogenGeneration', 
    'NaphthaPartialOxidation',
)

# DO NOT DELETE: This code computes the cost coefficients from real data
# # cost = Cb * S ** n
# # log(cost) = n log(S) + log(Cb)
# import numpy as np
# processing_capacity = np.array([2500, 6250]) * 907.185 / 24 * 0.39 * 0.23 # kg / hr
# capital_cost = np.array([2235000 + 110000/2, 3870000 + 196000/2])
# logcost = np.log(capital_cost)
# logS = np.log(processing_capacity)
# n, logCb = np.polyfit(logS, logcost, 1)
# Cb = np.exp(logCb)


@cost('Flow rate', units='kg/hr',
      cost=10073.488658306569, CE=567.3,
      n=0.5999300797598506, S=1, BM=1)
class HydrogenGeneration(bst.Unit):
    """
    Create a Gibbs equilibrium reactor for hydrogen production from Naphtha.
    This unit operation is also called NaphthaPartialOxidation.
    
    Parameters
    ----------
    ins : 
        * [0] Naphtha feed.
        * [1] Oxygen (pure).
        * [2] Water. It is mixed with the feed to prevent coking.
    outs :  
        Vapor product.
    T : float   
        Operating temperature [K].
    P : float   
        Operating pressure [bar].
    
    """
    _N_ins = 3
    _N_outs = 2
    _ins_size_is_fixed = False
    _outs_size_is_fixed = True
    method_default = 'COBYLA' # Alternatively 'differential evolution'
    auxiliary_unit_names = ('pump',)
    
    def _init(self, 
            T=None,
            P=None,
            oxygen_carbon_ratio=None, # Molecular oxygen
            water_carbon_ratio=None, 
            reactive: Optional[Iterable[str]]=None,
            products: Optional[Iterable[str]]=None, 
            method: Optional[str]=None,
        ):
        self.T = 1300 + 273.15 if T is None else T # Assume non-catalytic
        self.P = 30 * 101325 if P is None else P
        self.water_carbon_ratio = 1 if water_carbon_ratio is None else water_carbon_ratio
        self.oxygen_carbon_ratio = 0.75 if oxygen_carbon_ratio is None else oxygen_carbon_ratio
        if reactive is None:
            reactive = []
            for i in self.chemicals:
                if 'S' in i.atoms or 'N' in i.atoms: continue # Ignore component (will be rejected later anyway)
                if i.phase: continue # Phase is locked
                reactive.append(i.ID)
        self.reactive = reactive
        self.products = ('H2O', 'H2', 'CO', 'CO2', 'CH4')
        self.kWh_per_kgH2 = 0.5 # https://www.sciencedirect.com/science/article/pii/S0360319923020189
        self.H2_recovery_efficiency = 0.9 # https://www.sciencedirect.com/science/article/pii/S0360319923020189
        self.method = self.method_default if method is None else method
    
    def _run(self):
        feed, oxygen, water = self.ins
        hydrogen, other = self.outs
        oxygen.phase = hydrogen.phase = other.phase = 'g'
        hydrogen.T = other.T = self.T
        hydrogen.P = other.P = self.P
        atoms = feed.get_atomic_flows()
        C = atoms['C']
        water.imol['Water'] = water_supply = max(self.water_carbon_ratio * C - feed.imass['Water'], 0)
        O = atoms['O'] - water_supply # Do not count O atoms in water 
        O_demand = 2 * self.oxygen_carbon_ratio * C
        O2_supply = (O_demand - O) / 2
        oxygen.imol['O2'] = O2_supply
        other.empty()
        other.imol[self.reactive] = feed.imol[self.reactive]
        other.imol['O2'] += O2_supply
        other.imol['Water'] += water_supply
        minimize_Gibbs_free_energy(
            other, self.products, 
            method=self.method, 
        )
        H2_produced = other.imol['H2']
        hydrogen.imol['H2'] = H2_recovered = H2_produced * self.H2_recovery_efficiency
        other.imol['H2'] = H2_produced - H2_recovered
        
    def _design(self):
        self.design_results['Flow rate'] = self.ins[0].F_mass
        self.add_power_utility(
            self.kWh_per_kgH2 * sum([i.imol['H2'] for i in self.outs])
        )
        self.add_heat_utility(self.Hnet, self.T)
    
    
NaphthaPartialOxidation = HydrogenGeneration
    
def test_naphtha_partial_oxidation():
    import pyrolysis
    import numpy as np
    bst.settings.set_thermo(pyrolysis.create_chemicals(), pkg='ideal gas')
    naphtha = bst.Stream(
        ID=None, phase='g', T=433.15, P=101325, 
    )
    IDs = ('Rubber', 'Ash', 'Char', 'Ca(OH)2', 'CaSO4', 'SO2', 'carbon', 
           'molecular hydrogen', 'molecular nitrogen', 'sulfur', 
           'molecular oxygen', 'oxidane', 'methane', 'ethane', 'ethene',
           'propane', 'propene', 'butane', 'but-1-ene', 'but-1-yne',
           'carbon dioxide', 'carbon monoxide', 'hydrogen sulfide', 'hexane',
           'pentane', 'pent-2-yne', '1-methylcyclopentene', '3-methylcyclopentene',
           '2,5-dimethylhexa-1,5-diene', '2,2,3-trimethylpentane', 
           '1,1-dimethylcyclopentane', '2,3,4-trimethylpentane',
           '3,4-dimethylhexane', 'ethylcyclopentane', 'methylcyclohexane',
           '1,1-dimethylcyclohexane', 'oct-1-ene', 'ethylcyclohexane', 
           '1,1,3-trimethylcyclohexane', 'non-1-ene', '2-methyloct-1-ene', 
           '(4R)-1-methyl-4-prop-1-en-2-ylcyclohexene',
           '2,6,6-trimethylbicyclo[3.1.1]hept-2-ene', 'Limonene', 'benzene',
           'toluene', 'ethylbenzene', '1,3-xylene', 'styrene', 'cumene', 
           '1-ethyl-4-methylbenzene', 'propylbenzene', '1-ethyl-2-methylbenzene',
           '1-ethyl-3-methylbenzene', '1,2,3-trimethylbenzene', 'phenol',
           'benzonitrile', 'prop-1-enylbenzene', '1-methyl-4-propan-2-ylbenzene', 
           '2,3-dihydro-1H-indene', '1H-indene', '1-ethyl-2,3-dimethylbenzene', 
           'm-Cymene', '1-ethyl-2,4-dimethylbenzene', 'o-Cymene',
           '5-methyl-2,3-dihydro-1H-indene', '1,2,3,5-tetramethylbenzene',
           '1,2,3,4-tetramethylbenzene', '1-ethyl-2-propan-2-ylbenzene', 
           '2,3-dimethylphenol', '1-methyl-2,3-dihydro-1H-indene', 'benzoic acid',
           '2-methyl-2,3-dihydro-1H-indene', '1-methyl-1H-indene',
           '1,2,3,4-tetrahydronaphthalene', '4-methyl-2,3-dihydro-1H-indene',
           'naphthalene', '4-propan-2-ylphenol', 'benzothiazole', 
           '5-methyl-1,2,3,4-tetrahydronaphthalene', 'hexylbenzene',
           '1-methylnaphthalene', '2-methylnaphthalene',
           '1,2,3-trimethyl-1H-indene', "1,1'-biphenyl", '1-ethylnaphthalene',
           '2-ethylnaphthalene', '1,8-dimethylnaphthalene',
           '1,5-dimethylnaphthalene', '2,7-dimethylnaphthalene',
           '2-methylquinoline', 'tetradec-1-ene', 'pentadecane',
           '1,2,3-trimethylnaphthalene', 'butylbenzene',
           '1,2,5-trimethylnaphthalene', '9H-fluorene',
           '2,4-dimethyl-1-phenylbenzene', 'pentadec-1-ene', 'hexadecane',
           'anthracene', '4-methylphenanthrene', 'pentadecanoic acid',
           '3-methylphenanthrene', '2-methylphenanthrene', 'nonadecane',
           '1-methyl-7-propan-2-ylphenanthrene', 'icosane', 'henicosane', 
           'docosane', 'tetracosane', 'undecane', 'ammonia', 'cyclohexane')
    mol = [0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
            0.000e+00, 1.564e-16, 6.675e-13, 0.000e+00, 1.008e-10, 4.095e+00,
            2.340e-03, 1.645e-02, 6.488e-02, 2.543e-02, 4.618e-02, 2.459e-02,
            3.188e-02, 6.105e-02, 1.574e-02, 5.845e-07, 1.523e-01, 3.641e-01,
            3.366e-01, 1.829e-02, 5.927e-02, 5.823e-02, 6.730e-02, 4.697e-01,
            4.328e-01, 3.836e-01, 8.625e-01, 2.197e-01, 1.243e+00, 2.159e-02,
            1.514e-02, 1.154e-01, 4.967e-03, 2.536e-02, 2.477e-01, 6.009e-03,
            1.509e-02, 9.024e-02, 1.197e-02, 3.583e-02, 1.499e-02, 1.718e-02,
            1.455e-02, 2.336e-03, 1.603e-03, 6.574e-03, 4.921e-03, 5.302e-03,
            6.364e-04, 1.280e-02, 2.487e-03, 9.659e-04, 4.867e-03, 1.616e-03,
            2.554e-03, 3.233e-04, 1.007e-03, 4.221e-04, 8.351e-04, 4.830e-04,
            3.148e-04, 2.831e-04, 1.606e-04, 2.134e-03, 8.882e-04, 1.703e-03,
            4.258e-03, 1.043e-04, 4.464e-04, 3.776e-04, 1.645e-02, 2.004e-04,
            2.719e-02, 8.155e-03, 8.247e-03, 2.489e-02, 3.432e-02, 1.812e-02,
            2.024e-02, 1.061e-02, 4.370e-03, 5.614e-03, 7.180e-03, 1.189e-02,
            5.604e-02, 3.801e-02, 8.518e-03, 3.721e-03, 1.375e-03, 1.336e-03,
            2.226e-03, 3.893e-03, 3.107e-03, 3.808e-03, 5.830e-04, 4.591e-04,
            4.539e-03, 8.511e-04, 4.044e-03, 1.742e-04, 1.548e-04, 1.005e-04,
            8.620e-05, 2.814e-05, 2.978e-06, 5.844e-02, 0.000e+00, 0.000e+00]
    naphtha.imol[IDs] = mol
    NPO = NaphthaPartialOxidation(
        ins=[naphtha, 'oxygen', 'water'], 
        T=1300 + 273.15,
        P=30 * 101325,
        oxygen_carbon_ratio=0.75,
        water_carbon_ratio=1, 
    )
    NPO.simulate()
    np.testing.assert_allclose(
        NPO.outs[0].imol['H2'],
        3.883e+01,
        atol=1e-3, rtol=1e-2,
    )
    np.testing.assert_allclose(
        NPO.outs[1].imol['H2', 'H2O', 'CH4', 'CO2', 'CO'],
        [0, 5.201e+01, 3.334e-03, 1.407e+01, 2.870e+01],
        atol=1e-3, rtol=1e-2,
    )
    
if __name__ == '__main__':
    test_naphtha_partial_oxidation()
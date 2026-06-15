import biosteam as bst
import pandas as pd 
import numpy as np 
from scipy.integrate import solve_ivp
from chemicals.utils import R
from numba import njit
import os 

__all__ = (
    'PyrolysisReactor', 
)

@njit(cache=True)
def k4_waste_tire_pyrolysis(T):
    """
    Equation 27 from Ismail et al. (2017).

    Parameters
    ----------
    T : float or array_like
        Temperature [K]

    Returns
    -------
    k4 : float or ndarray
        Lumped rate constant [1/s]
    """
    RT = R * T
    k_ic = 0.195 * np.exp(-15920.0 / RT)
    k_i  = 0.0409 * np.exp(-3730.0 / RT)
    k_ia = 1.21 * np.exp(-39050.0 / RT)
    k_it = 0.0885 * np.exp(-11320.0 / RT)
    return (k_ic * k_i) / (k_ia + k_it + k_ic)

@njit(cache=True)
def pyrolysis_conversion(
        x, coefficients, char_coefficient, stoichiometry, char_stoichiometry, 
        reactant_index, carbon_index, solids_index):
    total = x.sum()
    X = x[reactant_index] / total
    rates = coefficients * X 
    return stoichiometry @ rates + char_stoichiometry * char_coefficient * x[carbon_index] / total
    
class PyrolysisReactor(bst.Unit):
    """
    """
    _N_ins = 3 # Feed, Syngas
    _N_outs = 4 # Vapor, Solids, Unreacted Syngas
    _units = {}

    tire_capacity = 62.5 # Capacity of tires [kg / m3]
    cage_length = 4 #: Cage length [m]
    cage_volume = cage_length ** 3
    cage_capacity = cage_volume * tire_capacity
    
    # DO NOT DELETE: This code computes the cost of correlation from real data
    # tau = 14
    # tire_flows_data = [i * 907.185 / 24 for i in (30, 60, 100, 150)]
    # N_cages_data = [np.ceil(i * tau / cage_capacity) for i in tire_flows_data]
    # cost_data = [6240000, 11870000, 17230000, 24544000]
    # cost_per_cage, facilities_cost = np.polyfit(N_cages_data, cost_data, 1)
    cost_per_cage = 1.111e6
    facilities_cost = 2.231e+06

    def _init(self, T=500 + 273.15, P=101325, tau=14):
        self.T = T #: Operation temperature [K]
        self.P = P #: Operating pressure [Pa]
        self.tau = tau #: Residence time [hr]
        self._load_kinetic_data()

    @property
    def cage_volume(self):
        L = self.cage_length
        return L * L * L

    def _load_kinetic_data(self):
        data_folder = os.path.join(os.path.join(os.path.dirname(__file__), '..'), 'data')
        path = os.path.join(data_folder, 'kinetic_main_original.csv')
        self.df_kinetics  = pd.read_csv(path)
        reactions_list, k_constants, n_constants, energy_constants, reactant_index, elements = [], [], [], [], [], []
        df_kinetics = self.df_kinetics
        H2_index, S_index, O2_index = self.chemicals.get_index(['H2', 'S', 'O2'])
        N_index = self.chemicals.get_index('N2')
        index_map = {1: ('H', H2_index), 2: ('O', O2_index), 3: ('S', S_index)}
        original_tire_composition = dict(
            C=75, H=7, N=0.3, S=1.5, O=2.7, Ash=13.5,
        ) # Used to undo scaling of rate coefficients from the original study (which disregarded moisture too)
        total_mass = sum(original_tire_composition.values())
        original_tire_composition = {i: j/total_mass for i, j in original_tire_composition.items()}
        chemicals = self.chemicals
        self.carbon_index = C_index = self.chemicals.get_index('Carbon')
        for i, available in enumerate(df_kinetics['Available']):
            if not available == 'Yes': continue
            k = df_kinetics['k_constant'].iloc[i]
            n_constants.append(df_kinetics['n'].iloc[i])
            energy_constants.append(df_kinetics['E[kj/mol]'].iloc[i])
            element, index = index_map[df_kinetics['Dependant_index'].iloc[i]]
            k *= original_tire_composition[element]
            product_name = df_kinetics['Common_name'].iloc[i]
            chemical = chemicals[product_name]
            if 'N' in chemical.atoms:
                element = 'N'
                index = N_index
            elif element == 'H' and 'C' in chemical.atoms: 
                if chemical.atoms['H'] / chemical.atoms['C'] < 2:
                    element = 'C'
                    index = C_index
            if ('O' in chemical.atoms or 'S' in chemical.atoms):
                k *= 10 # Rescale to match product distribution
            elif chemical.Tb < 300: 
                k *= 0.3 # Rescale to match product distribution
            k_constants.append(k)        
            elements.append(element)
            reactant_index.append(index)
            stoichiometry = {
                'H2': 1, 'sulfur': 1, 'C': 1,
                'N2': 1, 'O2': 1 , product_name: 1
            }
            reactions_list.append(
                bst.Reaction(
                    stoichiometry,  
                    correct_atomic_balance=True, 
                    reactant=df_kinetics['Reactant'].iloc[i],
                    basis='wt',
                ) 
            )
        # Parámetros de reacción
        self.k = np.array(k_constants) / 100 # Rescale kinetic constants to match product distribution
        self.E = np.array(energy_constants) * 1000 # Energía de activación en kJ/mol a J/mol
        self.n = np.array(n_constants)  # Exponentes de temperatura
        self.reactant_index = np.array(reactant_index)
        self.elements = elements
        self.reactions = bst.ParallelReaction(reactions_list)
        self.stoichiometry = np.array(self.reactions.stoichiometry).T
        self.solids_index = np.array(self.chemicals.get_index([
            'Rubber', 'Ca(OH)2', 'CaSO4', 'Char',
        ]))
        self.decomposition = bst.Reaction(
            'Rubber -> O2 + N2 + Sulfur + H2 + Carbon', X=1, reactant='Rubber',
            correct_atomic_balance=True,
        )
        self.char_formation = bst.Reaction(
            'C -> Char', X=1, reactant='C', correct_atomic_balance=True,
        )
        self.char_stoichiometry = self.char_formation.stoichiometry.to_array()

    def ode(self, t, x):
        return pyrolysis_conversion(
            x, 
            self.rate_coefficients, 
            self.char_coefficient,
            self.stoichiometry,
            self.char_stoichiometry,
            self.reactant_index,
            self.carbon_index,
            self.solids_index,
        )

    def _design(self):
        """ 
        """
        feed = self.ins[0]
        cage_capacity = self.cage_volume * self.tire_capacity
        N_cages = np.ceil(feed.F_mass * self.tau / cage_capacity)
        self.design_results['# Cages'] = N_cages
        self.add_heat_utility(self.duty, self.T, hxn_ok=False)
        
    def _cost(self):
        """ 
        """
        self.baseline_purchase_costs['Pyrolysis system'] = (
            self.cost_per_cage * self.design_results['# Cages'] + self.facilities_cost
        )

    def _run(self):
        feed, syngas, air = self.ins
        vapor, solids, unused_syngas, emissions = self.outs
        air.phase = syngas.phase = emissions.phase = unused_syngas.phase = vapor.phase = 'g'
        solids.phase = 's'
        feed = self.ins[0]
        tire = feed.copy()
        H2O = tire.imass['H2O']
        tire.imol['H2O'] = 0
        new_tire_composition = dict(
            C=tire.get_atomic_flow('C') * 12.011,
            H=tire.get_atomic_flow('H') * 1.008,
            N=tire.get_atomic_flow('N') * 14.01,
            S=tire.get_atomic_flow('S') * 32.07,
            O=tire.get_atomic_flow('O') * 15.999,
            Ash=tire.imass['Ash'],
            H2O=H2O # This time account for moisture content
        )
        tire.imass['H2O'] = H2O
        total_mass = sum(new_tire_composition.values())
        new_tire_composition = {i: j/total_mass for i, j in new_tire_composition.items()}
        rescaling_factor = np.array([new_tire_composition[e] for e in self.elements])
        self.rate_coefficients = (
            self.k / rescaling_factor * self.T**self.n * np.exp(-self.E / (R * self.T))
        )
        char_coefficient = k4_waste_tire_pyrolysis(self.T) / new_tire_composition['C']
        self.char_coefficient = char_coefficient
        self.decomposition(tire)
        sol = solve_ivp(self.ode, [0, self.tau * 60 * 60], tire.z_mass)
        mass_fractions = sol.y[:, -1]
        mass_fractions[mass_fractions < 0] = 0
        vapor.mass[:] = feed.F_mass * mass_fractions / mass_fractions.sum()
        vapor.imol['S', 'C'] = 0 # Remove negligible decomposition products
        solids.copy_flow(vapor, ['Ash', 'Char'], remove=True)
        vapor.T = solids.T = self.T
        vapor.P = solids.P = self.P
    
        # Duty that needs to be satisfied
        self.total_duty = duty = (vapor.Hnet + solids.Hnet) - feed.Hnet
        
        if not syngas.isempty():
            # Get available heat after combustion at the furnace
            combustion = self.chemicals.get_combustion_reactions()
            emissions.copy_like(syngas)
            combustion.force_reaction(emissions)
            O2_consumption = -emissions.imol['O2']
            air.imol['O2', 'N2'] = [O2_consumption, O2_consumption * 0.79 / 0.21]
            emissions.mol += air.mol
            F_emissions = emissions.F_mass
            z_CO2 = emissions.imass['CO2'] / F_emissions
            z_CO2_target = 0.055 # Usually between 4 - 7 for biomass and natural gas (https://www.sciencedirect.com/science/article/pii/S0957582021005127)
            F_emissions_new = z_CO2 * F_emissions / z_CO2_target
            dF_emissions = F_emissions_new - F_emissions
            air.F_mass = F_mass_O2_new = air.F_mass + dF_emissions
            emissions.mol += air.mol * (dF_emissions / F_mass_O2_new)
            emissions.T = 405 # T_emissions
            emissions.P = 3548325.0 # 500 psig furnace
            Q_available = syngas.Hnet - emissions.Hnet
            # Set fraction used
            fraction_used = duty / Q_available
            if fraction_used > 1: # Done
                duty -= Q_available
                unused_syngas.empty()
            elif fraction_used > 0:
                duty = 0
                unused_syngas.copy_like(syngas)
                fraction_unused = (1 - fraction_used)
                unused_syngas.F_mol *= fraction_unused
                emissions.F_mol *= fraction_used
                air.F_mol *= fraction_used
            else:
                # No syngas needed. Theoretically, this should not happen but it might
                # if tire has a lot of oxygen.
                air.empty()
                unused_syngas.copy_like(syngas)
                emissions.empty()
        else:
            air.empty()
            unused_syngas.copy_like(syngas)
            emissions.empty()
        self.duty = duty
        
    
def test_pyrolysis_reactor():
    from pyrolysis import create_chemicals
    bst.settings.set_thermo(create_chemicals(), pkg='ideal gas')
    feed = bst.Stream(Tire=6250, units='kg/hr')
    syngas = bst.Stream(CH4=2, units='kg/hr') # Some syngas
    air = bst.Stream()
    R1 = PyrolysisReactor(ins=[feed, syngas, air], tau=14)
    R1.simulate()
    vapor_oil, solids, unused_syngas, emissions = R1.outs
    
    # Pyrolysis heat requirement
    duty = ((vapor_oil.Hnet + solids.Hnet) - feed.Hnet) / feed.F_mass
    np.testing.assert_allclose(duty, 3365.8082786929644) # Would be less if feed was preheated
    
    # Make sure all syngas is used up
    np.testing.assert_allclose(emissions.imol['CO2'], syngas.imol['CH4'])
    np.testing.assert_allclose(emissions.imass['CO2'] / emissions.F_mass, 0.055)
    np.testing.assert_allclose(unused_syngas.F_mol, 0)
    
    # Test mass balance
    vapor_oil.vle(T=30 + 273.15, P=101325)
    noncondensables = vapor_oil['g']
    condensables = vapor_oil['l']
    F_mass = feed.F_mass
    x_noncondensable = noncondensables.F_mass / F_mass
    x_condensables = condensables.F_mass / F_mass
    x_metals = solids.imass['Ash'] / F_mass
    x_char = solids.imass['Char'] / F_mass
    np.testing.assert_allclose(
        [x_noncondensable, x_condensables, x_metals, x_char],
        [0.0702, 0.5121, 0.135, 0.2827],
        atol=1e-3, rtol=1e-4,
    )
    
    # Test excess syngas
    syngas.imol['CH4'] = 1000
    syngas.T = 773.15
    R1.simulate()    
    vapor_oil, solids, unused_syngas, emissions = R1.outs
    
    np.testing.assert_allclose(emissions.imol['CO2'] + unused_syngas.imol['CH4'], syngas.imol['CH4'])
    np.testing.assert_allclose(emissions.imass['CO2'] / emissions.F_mass, 0.055)
    assert unused_syngas.F_mol / syngas.F_mol > 0.9
    
    
if __name__ == '__main__':
    test_pyrolysis_reactor()
import biosteam as bst
from biosteam.units.decorators import cost
from biosteam.units.design_tools.Gibbs_equilibrium_reaction import minimize_Gibbs_free_energy

__all__ = (
    'RotaryKiln', 
    'GasificationActivation',
)

# DO NOT DELETE: This code computes the cost coefficients from real data
# # cost = Cb * S ** n
# # log(cost) = n log(S) + log(Cb)
# import numpy as np
# processing_capacity = np.array([60, 150]) * 907.185 / 24 * 0.37 # kg / hr
# capital_cost = np.array([3225000 + 82000, 9350000 + 143000])
# logcost = np.log(capital_cost)
# logS = np.log(processing_capacity)
# n, logCb = np.polyfit(logS, logcost, 1)
# Cb = np.exp(logCb)

# Thermal energy consumption (about 12.6e3 kJ per kg tire, assuming 0.37 char yield) for producing activated carbon (through gasification) based on: 
# https://www.sciencedirect.com/science/article/pii/S0921344917303579

# Electric energy consumption (about 72 kW per 2.5 ton of feed) based on specs from commercial rotary kiln:
# https://rotarykilnsupplier.com/activated-carbon-production/

# Burn off of activated carbon:
# 0.784 burn off required for a BET of 775.8 m2 / g
# This is low to medium BET
# https://www.sciencedirect.com/science/article/pii/S0921344917303579
# https://www.sciencedirect.com/science/article/pii/S0048969723056061#s0165

# Steam requirement: 
# 2 steam to activated carbon product (by weight)
# https://rotarykilnsupplier.com/activated-carbon-production/activation-process-of-activated-carbon/

@cost('Flow rate', units='kg/hr',
      cost=1427.3665083571323, CE=567.3,
      n=1.150850067797952, S=1, kW=72/(2.5*907.185), BM=1)
class RotaryKiln(bst.Unit): 
    _N_ins = 3
    _N_outs = 3
    
    def _init(self, 
              burn_off=0.784, 
              kJ_steam_per_kg=12.6e3 * 0.37,
              steam_to_product_demand=2,
              T=900 + 273.15,
              P=101325,
        ):
        self.burn_off = burn_off
        self.kJ_steam_per_kg = kJ_steam_per_kg
        self.steam_to_product_demand = steam_to_product_demand
        self.T = T
        self.P = P
    
    def _run(self):
        char, water, air = self.ins
        activated_carbon, syngas, emissions = self.outs
        activated_carbon.mol[:] = char.mol * (1 - self.burn_off)
        water.imass['Water'] = self.steam_to_product_demand * char.imass['Char']
        syngas.mol[:] = char.mol - activated_carbon.mol + water.mol
        syngas.phase = 'g'
        syngas.T = self.T
        syngas.P = self.P
        minimize_Gibbs_free_energy(
            syngas, ['CO', 'CO2', 'H2', 'CH4', 'H2O'], 
            method='COBYLA', 
        )
        
        
        duty = self.kJ_steam_per_kg * self.ins[0].F_mass
        
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
            syngas.empty()
        elif fraction_used > 0:
            duty = 0
            fraction_unused = (1 - fraction_used)
            syngas.F_mol *= fraction_unused
            emissions.F_mol *= fraction_used
            air.F_mol *= fraction_used
        else:
            raise RuntimeError('unexpected energy requirement')
        self.duty = duty
            
    def _design(self):
        self.design_results['Flow rate'] = self.ins[0].F_mass
        if self.duty: self.add_heat_utility(self.duty, self.T)
    
GasificationActivation = RotaryKiln
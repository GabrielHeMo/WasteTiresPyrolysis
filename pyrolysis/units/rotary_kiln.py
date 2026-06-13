import biosteam as bst
from biosteam.units.decorators import cost

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

# Electric energy consumption (abound 72 kW per 2.5 ton of feed) based on specs from commercial rotary kiln:
# https://rotarykilnsupplier.com/activated-carbon-production/

# Burn off of activated carbon:
# 0.784 burn off required for a BET of 775.8 m2 / g
# This is low to medium BET (between 1.5 to 1.8 USD / kg)
# https://www.sciencedirect.com/science/article/pii/S0921344917303579
# https://www.sciencedirect.com/science/article/pii/S0048969723056061#s0165

@cost('Flow rate', units='kg/hr',
      cost=1427.3665083571323, CE=567.3,
      n=1.150850067797952, S=1, kW=72/(2.5*907.185), BM=1)
class RotaryKiln(bst.Unit): 
    
    def _init(self, burn_off=0.784, kJ_steam_per_kg=12.6e3 * 0.37):
        self.burn_off = burn_off
        self.kJ_steam_per_kg = kJ_steam_per_kg
    
    def _run(self):
        char, = self.ins
        activated_carbon, = self.outs
        activated_carbon.mol[:] = char.mol * (1 - self.burn_off)
        
    def _design(self):
        self.design_results['Flow rate'] = F_mass = self.ins[0].F_mass
        self.add_heat_utility(self.kJ_steam_per_kg * F_mass, 900 + 273.15)
    
GasificationActivation = RotaryKiln
import biosteam as bst
from biosteam.units.decorators import cost

__all__ = (
    'MechanicalActivation', 
)

# DO NOT DELETE: This code computes the cost coefficients from real data
# # cost = Cb * S ** n
# # log(cost) = n log(S) + log(Cb)
# import numpy as np
# processing_capacity = np.array([60, 150]) * 907.185 / 24 * (0.37 + 0.12) # kg / hr
# capital_cost = np.array([4270000 + 98000, 7280000 + 180000])
# logcost = np.log(capital_cost)
# logS = np.log(processing_capacity)
# n, logCb = np.polyfit(logS, logcost, 1)
# Cb = np.exp(logCb)

# Energy consumption (between 1 to 2 kWh/kg) for mechanical activation to carbon black based on: 
# https://www.sciencedirect.com/science/article/pii/S2667378926000295

@cost('Flow rate', units='kg/hr',
      cost=72621.19719824931, CE=567.3,
      n=0.5841488483445471, S=1, kW=1.5, BM=1)
class MechanicalActivation(bst.Splitter): 
    
    def _init(self):
        super()._init(split=1)
        self.isplit['Ash'] = 0
        
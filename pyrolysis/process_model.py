import numpy as np
import pandas as pd
import biosteam as bst
import os 
import pyrolysis
from typing import Optional

__all__ = ('WasteTirePyrolysisProcess',)

class WasteTirePyrolysisProcess(bst.ProcessModel):
    
    class Scenario:
        processing_capacity: Optional[float] = 6250, 'kg/hr of bone dry waste tires'
    
    def update_feedstock(self):
        moisture = self.tire_moisture_content
        ash = self.tire_ash_content
        C = self.rubber_carbon_content
        H = self.rubber_hydrogen_content
        O = self.rubber_oxygen_content
        N = self.rubber_nitrogen_content
        S = self.rubber_sulfer_content
        pyrolysis.adjust_tire_composition(
            H=H, O=O, N=N, C=C, S=S, # Elemental composition
            rubber=100 - ash - moisture, ash=ash, moisture=moisture, # Overall composition
        )
        original = self.feedstock.imol['Tire']
        new = 100 * self.processing_capacity / (100 - moisture)
        if original:
            self.system.rescale(self.feedstock, new / original)
        else:
            self.feedstock.imol['Tire'] = new
    
    def create_thermo(self):
        return bst.Thermo(pyrolysis.create_chemicals(), pkg='ideal gas')
    
    def create_system(self):
        return pyrolysis.create_waste_tire_pyrolysis_system()
    
    def create_model(self):
        bst.settings.define_impact_indicator('GWP', 'kg*CO2e/kg')
        emissions = (self.emissions, self.emissions_1, self.emissions_2)
        self.system.define_process_impact(
            key='GWP', name='emissions', basis='kg', 
            inventory=lambda: sum([i.imass['CO2'] for i in emissions]), 
            CF=1
        )
        self.system.set_tolerance(
            mol=1e-3, rmol=1e-3, maxiter=200, subsystems=True,
            method='fixed-point'
        )
        
        # GREET 2023
        self.diesel.set_CF('GWP', 0.5169) 
        self.LFO.set_CF('GWP', 0.5169)
        self.metals.set_CF('GWP', 1.9872)
        self.tea = tea = pyrolysis.create_tea(self.system)
        model = bst.Model(self.system)
        parameter = model.parameter
        indicator = model.indicator
        
        data = pyrolysis.get_component_data()
        IDs = data['Component ID'].values
        Ignition = data['Ignition'].values
        Flash = data['Flash'].values
        
        @indicator(units='damage*meter')
        def FEDI():
            pyrolysis_reactor = self.pyrolysis_reactor
            product = pyrolysis_reactor.outs[0]
            LHV = product.LHV / product.F_mass  # KJ/kg
            z_frac = product.imass[IDs]
            z_frac /= z_frac.sum()
            Ignition_mixture = (z_frac * Ignition).sum() + 273.15
            Flash_point_mixture = (z_frac * Flash).sum() + 273.15
            Mass = product.get_total_flow('kg/s')
            Pequipment = product.get_property('P', 'kPa') #  [Pa] to [kPa]
            Vol = pyrolysis_reactor.design_results['# Cages'] * pyrolysis_reactor.cage_volume
            F1 = 0.1 * Mass * LHV / 3.148
            F2 = 6 / 3.148 * Pequipment*Vol
            Tope = pyrolysis_reactor.T
            F3 = 0 # Its a gas, not a liquid, so 0 by definition
            if Tope > Flash_point_mixture and Tope < 0.75 * Ignition_mixture: 
                pn1 = (1.45 +  1.75) / 2
            elif Tope > 0.75 * Ignition_mixture:
                pn1 = 1.95
            else:
                pn1 = 1.1
            # Penalty 2
            pn2 = 1 
            F = F2
            pn2 = 1.1
            F = F3          
            F4  = (Mass * pyrolysis_reactor.total_duty) / 3.148
    
            pn4 = 1 + 0.25*3
            pn3, pn5, pn6, pn7 = 1, 1, 1, 1.45
            Damage_Potential = (F1*pn1 + F*pn2 + F4*pn7) * pn3 * pn4 * pn5 * pn6
            Fedi_val = 4.76 * Damage_Potential**(1/3)
            return Fedi_val
    
        @indicator(units='10^6 * USD')
        def TCI(): return tea.TCI / 1e6 
    
        @indicator(units='%')
        def IRR(): return 100 * tea.solve_IRR() 
        
        @indicator(units='kg*CO2e/kg')
        def GWP(): return self.system.get_product_impact(self.feedstock, 'GWP', allocation_method='displacement')
        
        @parameter(
            element='tire', units='dry kg/hr', 
            bounds=(2500, 6250), distribution='uniform',
            coupled=True,
        )
        def set_processing_capacity(processing_capacity):
            self.processing_capacity = processing_capacity
        
        if self.scenario.processing_capacity is not None: 
            set_processing_capacity.active = False
            self.processing_capacity = self.scenario.processing_capacity
        
        @parameter(
            element='tire', units='wt%', 
            bounds=(0.4, 2), baseline=1.0, distribution='triangular',
            coupled=True,
        )
        def set_moisture_content(moisture_content):
            self.tire_moisture_content = moisture_content
            
        @parameter(
            element='tire', units='wt%',  
            bounds=(0, 9.89), baseline=2.5, distribution='triangular',
            coupled=True,
        )
        def set_ash_content(ash_content):
            self.tire_ash_content = ash_content
            
        @parameter(
            element='tire', units='wt%',  
            bounds=(75, 89.9), baseline=83.3, distribution='triangular',
            coupled=True,
        )
        def set_rubber_carbon_content(rubber_carbon_content):
            self.rubber_carbon_content = rubber_carbon_content
    
        @parameter(
            element='tire', units='wt%',
            bounds=(6.56, 7.99), baseline=7.5, distribution='triangular',
            coupled=True,
        )
        def set_rubber_hydrogen_content(rubber_hydrogen_content):
            self.rubber_hydrogen_content = rubber_hydrogen_content
    
        @parameter(
            element='tire', units='wt%', 
            bounds=(1.29, 10.79), baseline=4.5, distribution='triangular', 
            coupled=True,
        )
        def set_rubber_oxygen_content(rubber_oxygen_content):
            self.rubber_oxygen_content = rubber_oxygen_content
    
        @parameter(
            element='tire', units='wt%', 
            bounds=(0.3, 1.0), baseline=0.6, distribution='triangular', 
            coupled=True,
        )
        def set_rubber_nitrogen_content(rubber_nitrogen_content):
            self.rubber_nitrogen_content = rubber_nitrogen_content
    
        @parameter(
            element='tire', units='wt%', 
            bounds=(0.87, 2.46), baseline=1.6, distribution='triangular',
            coupled=True,
        )
        def set_rubber_sulfer_content(rubber_sulfer_content):
            self.rubber_sulfer_content = rubber_sulfer_content
            self.update_feedstock()
            
        @parameter(
            element='Pyrolysis reactor', units='K', 
            bounds=(500 + 273.15, 800 + 273.15), baseline=550 + 273.15, distribution='uniform',
            coupled=True,
        )
        def set_pyrolysis_reactor_temperature(temperature):
            self.pyrolysis_reactor.T = temperature
    
        kg_per_ton = 907.185
    
        # https://archive.epa.gov/epawaste/conserve/materials/tires/web/pdf/tires.pdf
        @parameter(
            element='Tire', units='USD/kg',
            bounds=(35 / kg_per_ton, 108 / kg_per_ton),
            baseline=50 / kg_per_ton,
            distribution='uniform'
        )
        def set_tire_tipping_fee(tire_tipping_fee):
            self.feedstock.price = -tire_tipping_fee
        
        # Whole sale price
        # https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=pet&s=ema_epd2d_pwg_nus_dpg&f=m
        @parameter(
            element='Diesel', units='USD/gal',
            bounds=(0.878, 3.582),
            baseline=3.582,
            distribution='uniform' 
        )
        def set_diesel_price(diesel_price):
            self.diesel.price = diesel_price / 3.22 # gal to kg
        
        @parameter(
            element='LFO', units='USD/gal',
            bounds=(0.878, 3.582),
            baseline=3.582,
            distribution='uniform',
        )
        def set_LFO_price(LFO_price):
            self.LFO.price = LFO_price / 3.22 # gal to kg
        
        # 750 USD / ton
        # https://www.alibaba.com/product-detail/Coal-Granular-Commercial-Bulk-Coal-Based_1600679536520.html?spm=a2700.7724857.0.0.3f2f7f5fVPa2vq
        
        # 1,050 USD / ton
        # https://www.alibaba.com/product-detail/Good-Price-Coal-Based-Columnar-Activated_1601391864743.html?spm=a2700.7724857.0.0.3f2f7f5fVPa2vq
        
        # 960.00 USD / ton
        # https://yrdcarbon.en.made-in-china.com/product/exEYRISrJvck/China-Coal-Based-Powdered-Activated-Carbon-Price-Per-Ton-for-Power-Plant.html
        
        @parameter(
            element='Activated carbon', units='USD/kg',
            bounds = (750 / kg_per_ton, 1050 / kg_per_ton),
            baseline = 960 / kg_per_ton,
            distribution='uniform'
        )
        def set_carbon_price(price):
            self.activated_carbon.price = price            
    
        # 215 USD / ton
        # https://jrsadvancedrecyclers.com/scrap-metal-prices/#steel
        metal_price = 215 / kg_per_ton
        
        @parameter(
            element='Metals', units='USD/kg',
            bounds=(metal_price * 0.80, metal_price * 1.2),
            baseline=metal_price,
            distribution='triangular'
        )
        def set_metals_price(price):
            self.metals.price = price
        
        @parameter(
            element='Activated carbon', units='wt %',
            bounds=(48.1, 78.4),
            baseline=78.4,
            distribution='uniform',
            coupled=True,
        )
        def set_burn_off(burn_off):
            self.rotary_kiln.burn_off = burn_off / 100
        
        @parameter(
            element='Mechanical activation', units='kWh/kg',
            bounds=(1, 2),
            baseline=1.5,
            distribution='uniform'
        )
        def set_mechanical_activation_power(power):
            MA = self.mechanical_activation
            MA.cost_items[MA.line].kW = power
        
        # WE DO NOT USE THIS REFERENCE because it includes end of life emissions (so we avoid double counting)
        # Cradle to gate: Comparative life cycle assessment of biomass-based and coal-based activated carbon production
        # https://www.researchgate.net/publication/363140955_Comparative_life_cycle_assessment_of_biomass-based_and_coal-based_activated_carbon_production/fulltext/636c9025431b1f530086795e/Comparative-life-cycle-assessment-of-biomass-based-and-coal-based-activated-carbon-production.pdf
        # These include end of life emissions:
        # GWP_AC_coal = 8.6 # Pyrolysis and physical activation
        # GWP_AC_wood = 1.08 # Pyrolysis and physical activation
        
        # Use cradle to gate from Gu et al (does not include end of life emissions)
        # https://www.sciencedirect.com/science/article/pii/S0301479722019296?via%3Dihub#bib7
        
        self.activated_carbon.set_CF('GWP', 3.41) 
        
        return model


def test_process_model():
    process = WasteTirePyrolysisProcess(processing_capacity=6250)
    np.testing.assert_allclose(process.FEDI(), 965.842264949065, rtol=1e-3)
    np.testing.assert_allclose(process.TCI(), 121.42411596282324, rtol=1e-3)
    np.testing.assert_allclose(process.IRR(), 2.8789219652373093, rtol=1e-3) 
    np.testing.assert_allclose(process.GWP(), -0.4012597038401888, rtol=1e-3)
    
    
if __name__ == '__main__':
    test_process_model()
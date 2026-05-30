
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

import biosteam as bst
from .system import create_system
import chaospy
from chaospy import distributions as shape

import os 
# __all__ = ('run_montecarlo')

def _create_model(context_sys, settings, tea):
    model = bst.Model(context_sys)

    @model.indicator(units='damage*meter')
    def FEDI():
        df_data = pd.read_csv('Data_components_clean.csv', encoding='latin1')
        components = df_data['ComponentID'][2:].to_list()
        
        ng = bst.F.MainReactor.outs[0].copy() #R2.outs[0].copy()
        combustion_rxns = settings.chemicals.get_combustion_reactions()
        final_flowsa = bst.F.MainReactor.final_flows[-1,:]
        # Salida de reactor
        flows_total =  [] 
        for i ,c in enumerate(settings.chemicals): 
            if i > 2 and i < len(settings.chemicals)-2:
                flows_total.append(( c.common_name , final_flowsa[i]))
        ng = bst.Stream(None, phase = 'g',  T=bst.F.MainReactor.Treac, 
                        P=bst.F.MainReactor.Preac, units='kg/hr', **dict(flows_total))
        product = ng.copy()
        combustion_rxns.force_reaction(product)
        O2 = max(-product.imol['O2'], 0.)
        ng.imol['O2'] = O2
        product.imol['O2'] = 0
        product.phase = 'g'
        Enthalpy_comb = abs(product.Hnet - ng.Hnet)/bst.F.MainReactor.outs[0].F_mol  # KJ/mol
        

        z_frac = np.array(ng.mol)[3:-2]/sum(np.array(ng.mol)[3:-2])
        Ignition_mixture =sum(z_frac*df_data['Ignition'][1:-2].to_numpy() ) + 273.15
        Flash_point_mixture = sum(z_frac*df_data['Flash'][1:-2].to_numpy() ) + 273.15

        Mass = bst.F.MainReactor.outs[0].F_mass/3600  #  [kg/hr] to   [Kg/sec] de vapores
        Pequipment = bst.F.MainReactor.Preac/1000  #  [Pa] to [kPa]
        Vol =   (4*4*4)*22  # [m3]
        VapPress  =    bst.F.MainReactor.Preac/1000  #  752180.689431543   # [kPa] Average Value taken from Aspen
        F1 = 0.1*(Mass*(Enthalpy_comb))/3.148
        F2 = (6/3.148)*Pequipment*Vol
        Tope = bst.F.MainReactor.Treac
        F3 = (1e-3)*(1/Tope)*((Pequipment-VapPress)**2)*Vol   # La temperatura es de la operación
        if Tope > Flash_point_mixture and Tope < 0.75*Ignition_mixture: 
            pn1 = (1.45 +  1.75)/2
        elif Tope > 0.75*Ignition_mixture:
            pn1 = 1.95
        else:
            pn1 = 1.1
        # Penalty 2
        if VapPress > 101.325 and Pequipment > VapPress :
            pn2 = 1 + (Pequipment-VapPress)*0.6/Pequipment
            F = F2 + F3
        else:
            pn2 = 1 + (Pequipment-VapPress)*0.4/Pequipment
            F = F2
        if VapPress < 101.325 and 101.325 < Pequipment:
            pn2 = 1 + (Pequipment-VapPress)*0.2/Pequipment
            F = F3
        else:
            pn2 = 1.1
            F = F3          
        F4  =   (Mass*abs(bst.F.MainReactor.duty))/3.148

        pn4 = 1 + 0.25*(3 + 0)
        pn3,pn5,pn6,pn7 = 1,1,1, 1.45
        # pn1, pn2 = 1, 1
        # print('F1', F1, 'F', F, 'F4', F4)
        Damage_Potential = (F1*pn1 + F*pn2 + F4*pn7)*(pn3*pn4*pn5*pn6)
        # print('Damage_Potential', Damage_Potential)
        Fedi_val = 4.76*(Damage_Potential**(1/3))
        # print('Fedi_val',Fedi_val)
        return Fedi_val

    @model.indicator(units='USD/kg')
    def MSP_Diesel(): return tea.solve_price(bst.F.Diesel)

    @model.indicator(units='USD/kg')
    def MSP_LFO(): return tea.solve_price(bst.F.LFO)

    @model.indicator(units='USD/kg')
    def MSP_CarbonActivated(): return tea.solve_price(bst.F.CarbonActivated)

    @model.indicator(units='USD/kg')
    def MSP_Metals(): return tea.solve_price(bst.F.Metals)

    @model.indicator(units='10^6 * USD')
    def TCI(): return tea.TCI / 1e6 # total capital investment

    @model.indicator(units='10^6 * USD')
    def NPV(): return tea.NPV/ 1e6 # total capital investment

    @model.indicator(units='%')
    def IRR(): return round(tea.solve_IRR()*100,2) # Investment return ratio 

    # GWP USANDO MASA

    def _gwp_mass_per_kg_for_products(products):
        GWP = 'GWP 100yr'
        bst.F.Tyre_Stream.set_CF(GWP, 0)
        bst.F.CarbonActivated.set_CF(GWP,0)
        bst.F.Metals.set_CF(GWP,0)
        bst.F.Diesel.set_CF(GWP,0)
        bst.F.LFO.set_CF(GWP,0)
        bst.F.Oxygen.set_CF(GWP,0.18)
        bst.F.natural_gas.set_CF(GWP,0.33) # Antes 1.5 Correcto 0.33
        # products = (bst.F.CarbonActivated, bst.F.Metals, bst.F.Diesel, bst.F.LFO)
        # Total impact anual (feeds + net electricity); NO incluye "créditos" de coproductos
        total_impact = (context_sys.get_total_feeds_impact(GWP)
                        + context_sys.get_net_electricity_impact(GWP))
        total_prod_mass = sum(context_sys.get_mass_flow(s) for s in products)
        if total_prod_mass <= 0:
            return 0 # float("nan")
        # Como estás asignando por masa dentro de ese set, kgCO2e/kg es el mismo para todos
        return float(total_impact / total_prod_mass)

    @model.indicator(units='kg-CO2e/kg')
    def GWP_mass_CA():
        products = (bst.F.CarbonActivated) 
        # Total impact anual (feeds + net electricity); NO incluye "créditos" de coproductos
        gwp_mass_per_kg_for_products = _gwp_mass_per_kg_for_products(products)
        return gwp_mass_per_kg_for_products

    @model.indicator(units='kg-CO2e/kg')
    def GWP_mass_Metals():
        products = (bst.F.Metals)
        # Total impact anual (feeds + net electricity); NO incluye "créditos" de coproductos
        gwp_mass_per_kg_for_products = _gwp_mass_per_kg_for_products(products)
        return gwp_mass_per_kg_for_products

    @model.indicator(units='kg-CO2e/kg')
    def GWP_mass_Diesel():
        products = (bst.F.Diesel) 
        # Total impact anual (feeds + net electricity); NO incluye "créditos" de coproductos
        gwp_mass_per_kg_for_products = _gwp_mass_per_kg_for_products(products)
        return gwp_mass_per_kg_for_products

    @model.indicator(units='kg-CO2e/kg')
    def GWP_mass_LFO():
        products = (bst.F.LFO) 
        # Total impact anual (feeds + net electricity); NO incluye "créditos" de coproductos
        gwp_mass_per_kg_for_products = _gwp_mass_per_kg_for_products(products)
        return gwp_mass_per_kg_for_products

    # GWP CON ENERGIA
    def _gwp_energy_per_kg(stream):
        GWP = 'GWP 100yr'
        bst.F.Tyre_Stream.set_CF(GWP, 0)
        bst.F.CarbonActivated.set_CF(GWP,0)
        bst.F.Metals.set_CF(GWP,0)
        bst.F.Diesel.set_CF(GWP,0)
        bst.F.LFO.set_CF(GWP,0)
        bst.F.Oxygen.set_CF(GWP,0.18)
        bst.F.natural_gas.set_CF(GWP,0.33) # Antes 1.5 Correcto 0.33
        if stream.F_mass <= 0:
            return float("nan")
        GWP_per_GGE = context_sys.get_property_allocated_impact(
            key=GWP, name='energy', basis='GGE'
        )  # kgCO2e / GGE
        try:
            gge_per_hr = stream.get_property('LHV', 'GGE/hr')  # puede ser 0 si no aplica
        except Exception:
            return float("nan")
        return float(GWP_per_GGE * gge_per_hr / stream.F_mass)  # kgCO2e/kg

    @model.indicator(units='kg-CO2e/kg')
    def GWP_energy_activatedcarbon():
        return _gwp_energy_per_kg(bst.F.CarbonActivated)

    @model.indicator(units='kg-CO2e/kg')
    def GWP_energy_metals():
        return _gwp_energy_per_kg(bst.F.Metals)

    @model.indicator(units='kg-CO2e/kg')
    def GWP_energy_diesel():
        return _gwp_energy_per_kg(bst.F.Diesel)

    @model.indicator(units='kg-CO2e/kg')
    def GWP_energy_LFO():
        return _gwp_energy_per_kg(bst.F.LFO)

    # GWP por revenue
    def _gwp_revenue_per_kg(stream):
        GWP = 'GWP 100yr'
        bst.F.Tyre_Stream.set_CF(GWP, 0)
        bst.F.CarbonActivated.set_CF(GWP,0)
        bst.F.Metals.set_CF(GWP,0)
        bst.F.Diesel.set_CF(GWP,0)
        bst.F.LFO.set_CF(GWP,0)
        bst.F.Oxygen.set_CF(GWP,0.18)
        bst.F.natural_gas.set_CF(GWP,0.33) # Antes 1.5 Correcto 0.33  
        GWP_per_USD = context_sys.get_property_allocated_impact(
            key=GWP, name='revenue', basis='USD'
        )  # kgCO2e / USD
        price = stream.price or 0.0  # USD/kg
        return float(GWP_per_USD * price)  # kgCO2e/kg

    @model.indicator(units='kg-CO2e/kg')
    def GWP_revenue_activatedcarbon():
        return _gwp_revenue_per_kg(bst.F.CarbonActivated)

    @model.indicator(units='kg-CO2e/kg')
    def GWP_revenue_metals():
        return _gwp_revenue_per_kg(bst.F.Metals)

    @model.indicator(units='kg-CO2e/kg')
    def GWP_revenue_diesel():
        return _gwp_revenue_per_kg(bst.F.Diesel)

    @model.indicator(units='kg-CO2e/kg')
    def GWP_revenue_LFO():
        return _gwp_revenue_per_kg(bst.F.LFO)

    @model.indicator(units='%wt')
    def Carbon_Yield(): return bst.F.MainReactor.carbon_yield 

    @model.indicator(units= 'conversion')
    def RealConversion(): return bst.F.MainReactor.real_Carbonconversion

    # Este es necesario para los calculos 
    @model.parameter(element='P_moisture', units='wt%',  
                    bounds=(0.4, 12.13),
                    baseline=4.0,
                    distribution='triangular',
                    coupled=True)
    def set_P_moisture(value_P_moisture):
        moisture = round(value_P_moisture, 2)

        # 1) actualizar humedad en el reactor
        bst.F.MainReactor.P_moisture = moisture

        # 2) mantener constante la capacidad seca nominal
        dry_capacity = bst.F.MainReactor.processing_capacity  # kg/hr dry
        wet_capacity = dry_capacity / (1 - moisture/100)

        # 3) actualizar la corriente de alimentación húmeda
        bst.F.Tyre_Stream.imass['Tyre'] = wet_capacity

    # Esta si se usa: P_ash = U_ash
    @model.parameter(element='Ash', units='wt%',  
                    bounds=(0, 9.89), 
                    baseline=2.5, 
                    distribution='triangular',
                    coupled=True)
    def set_U_ash(value_ash):
        bst.F.MainReactor.U_ash = round(value_ash,2)
        # bst.F.DescomposerR1.U_ash = round(value_ash,2)

    # Parametros independientes U
    @model.parameter(element='U_carbon', units='wt%',  
                    bounds=(75, 89.9), 
                    baseline=83.3, 
                    distribution='triangular',
                    coupled=True)
    def set_U_carbon(value_c):
        # bst.F.DescomposerR1.U_carbon = round(value_c,2)
        bst.F.MainReactor.U_carbon = round(value_c,2)

    @model.parameter(element='U_h', units='wt%',
                    bounds=(6.56, 7.99), 
                    baseline=7.5, 
                    distribution='triangular',
                    coupled=True)
    def set_U_h(value_h):
        bst.F.MainReactor.U_h = round(value_h,2)

    @model.parameter(element='U_o', units='wt%', 
                    bounds=(1.29, 10.79), 
                    baseline=4.5, 
                    distribution='triangular', 
                    coupled= True)
    def set_U_o(value_o):
        bst.F.MainReactor.U_o = round(value_o,2)

    @model.parameter(element='U_n', units='wt%', 
                    bounds=(0.3, 1.0), 
                    baseline=0.6, 
                    distribution='triangular', 
                    coupled= True)
    def set_U_n(value_n):
        bst.F.MainReactor.U_n = round(value_n,2)

    @model.parameter(element='U_s', units='wt%', 
                    bounds=(0.87, 2.46),
                    baseline=1.6, 
                    distribution='triangular',
                    coupled=True)
    def set_U_s(value_s):
        bst.F.MainReactor.U_s = round(value_s,2) 
        
    Treactor_R2 = bst.F.MainReactor.Treac
    lb , ub = 500 + 273.15  ,  800 + 273.15
    @model.parameter(element = 'Treaction', units = 'K', 
                     bounds = (lb,ub) , baseline= 550 + 273.15, distribution='triangular')
    def set_Treaction_R2(Treaction):
        bst.F.MainReactor.Treac = Treaction

    carbon_conversion_R2 = bst.F.MainReactor.carbon_conversion
    lb , ub = 30 , 60
    @model.parameter(element = 'carbonconversion', units = '%wt', 
                     bounds = (lb,ub) , 
                     baseline= carbon_conversion_R2, 
                     distribution='triangular', coupled=True)
    def set_conversion_R2(conversion_val):
        bst.F.MainReactor.carbon_conversion = round(conversion_val,1)

    feedstock = bst.F.Tyre_Stream      # Es con el ID de la corriente 
    lb = feedstock.price * 0.80
    ub = feedstock.price * 1.20
    @model.parameter(
        element='Tyre', units='USD/kg',
        bounds=(lb, ub),
        baseline=feedstock.price,
        distribution='triangular')
    def set_tyre_price(tyre_price):
        feedstock.price = tyre_price

    # Variar el precio del Diesel 
    product_diesel = bst.F.Diesel # The feedstock stream
    lb = product_diesel.price * 0.8 # Minimum price
    ub = product_diesel.price * 1.2 # Maximum price
    @model.parameter(
        element='Diesel', units='USD/kg',
        bounds=(lb, ub),
        baseline=product_diesel.price,
        distribution='triangular' 
    )
    def set_diesel_price(product_diesel_price):
        product_diesel.price = product_diesel_price

    # Variar el precio de LFO
    product_LFO = bst.F.LFO
    lb = product_LFO.price * 0.8 # Minimum price
    ub = product_LFO.price * 1.2 # Maximum price
    @model.parameter(
        element='LFO', units='USD/kg',
        bounds=(lb, ub),
        baseline=product_LFO.price,
        distribution='triangular' # Defaults to shape.Triangular(lower=lb, midpoint=baseline, upper=ub)
    )
    def set_lfo_price(product_lfo_price):
        product_LFO.price = product_lfo_price

    product_CarbonActivated = bst.F.CarbonActivated
    lb = product_CarbonActivated.price * 0.8 # Minimum price
    ub = product_CarbonActivated.price * 1.2 # Maximum price
    @model.parameter(
        element='Carbon', units='USD/kg',
        bounds = (lb, ub),
        baseline = product_CarbonActivated.price,
        distribution='triangular' )
    def set_carbon_price(product_carbon_price):
        product_CarbonActivated.price = product_carbon_price            


    product_Metals = bst.F.Metals
    lb = product_Metals.price * 0.8 # Minimum price
    ub = product_Metals.price * 1.2 # Maximum price
    @model.parameter(
        element='Metals', units='USD/kg',
        bounds = (lb, ub),
        baseline = product_Metals.price,
        distribution='triangular' )
    def set_metals_price(product_metals_price):
        product_Metals.price = product_metals_price            
  
    return model


def run_montecarlo(PC , file_ps, file_corr,  settings,  N_samples = 5000): 
    # Change Number of Samples if more data need to be collected
    output_folder = 'results'
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Construct the full relative path
    output_pat_ps = os.path.join(output_folder, file_ps)
    output_pat_corr = os.path.join(output_folder, file_corr)

    # create system again
    context_sys, tea, NPV , IRR_val, _ , _ = create_system(processing_capacity=PC,  x = 55, y = 3.5, settings= settings)
    # create model
    model =  _create_model(context_sys , settings,  tea)
    np.random.seed(42) # For consistent results
    rule = 'L' # For Latin-Hypercube sampling
    samples = model.sample(N_samples, rule)
    # Standardize tire compositions
    for i in range(samples.shape[0]):
        samples[i,1:7] = (samples[i,1:7]/samples[i,1:7].sum())*100
    model.load_samples(samples, sort=True)
    model.exception_hook = 'raise'
    model.evaluate( notify = 100 )
    full_problem_space = model.table.copy() # All evaluations are stored as a pandas DataFrame
    full_problem_space.to_csv(output_pat_ps)


    df_rho, df_p = model.spearman_r()
    # solo guardar df_rho para despues graficar ! 
    df_rho.to_csv( output_pat_corr)

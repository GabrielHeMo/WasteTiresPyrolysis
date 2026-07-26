# -*- coding: utf-8 -*-
"""
"""
import biosteam as bst
import pyrolysis

__all__ = (
    'create_waste_tire_pyrolysis_system',
)

@bst.SystemFactory(
    ins=[dict(ID='feedstock', Tire=6250, units='kg/hr'),
         dict(ID='fresh_hydrogen', units='kg/hr', price=4.5)],
    outs=[dict(ID='diesel', price=1.06),
          dict(ID='LFO', price=1.06),
          dict(ID='metals', price=0.237),
          dict(ID='activated_carbon', price=1.5)]
)
def create_waste_tire_pyrolysis_system(ins, outs):
    feed, fresh_hydrogen = ins
    diesel, LFO, metals, activated_carbon = outs
    syngas_recycle = bst.Stream()
    R1 = pyrolysis.PyrolysisReactor(ins=[feed, syngas_recycle], tau=14)
    R1.register_alias('pyrolysis_reactor')
    vapor, solids, unused_syngas, emissions = R1.outs
    emissions.ID = 'emissions'
    condensation_sys = pyrolysis.create_pyrolysis_product_condensation_system(
        ins=vapor, outs=[syngas_recycle, 'naphtha', 'fuel_oil']
    )
    syngas_recycle, naphtha, fuel_oil = condensation_sys.outs
    P_naphtha = 30 * 101325
    P1 = bst.Pump(ins=naphtha, P=P_naphtha)
    residual_naphtha = bst.Stream()
    P2 = bst.Pump(ins=residual_naphtha, P=P_naphtha)
    naphtha_mixer = bst.Mixer(ins=[P1-0, P2-0])
    naphtha_splitter = bst.Splitter(ins=naphtha_mixer-0, split=1.0)
    HG = pyrolysis.HydrogenGeneration(ins=[naphtha_splitter-0, 'oxygen', 'water'])
    _, oxygen, water = HG.ins
    oxygen.price = 0.09
    water.price = 0.00021133774 # 0.8 USD / 1000 gal
    P_hydrotreater = 100 * 101325
    HX1 = bst.HXutility(ins=HG-0, T=40 + 273.15)
    C1 = bst.IsentropicCompressor(ins=HX1-0, P=P_hydrotreater)
    HX2 = bst.HXutility(ins=C1-0, T=40 + 273.15)
    H2_mixer = bst.Mixer(ins=[HX2-0, fresh_hydrogen]) 
    P3 = bst.Pump(ins=fuel_oil, P=P_hydrotreater)
    HT = pyrolysis.Hydrotreater(ins=[P3-0, H2_mixer-0])
    ignored = [
        P1, P2, naphtha_mixer, naphtha_splitter, HG,
        HG, HX1, C1, HX2, H2_mixer
    ]
    for unit in ignored:
        @unit.add_specification
        def do_nothing(): pass
    
    @HT.add_specification
    def adjust_naphtha_recycling():
        fresh_hydrogen.imol['H2'] = 0
        
        # Find maximum hydrogen production
        naphtha_splitter.split[:] = 1
        naphtha_to_before_mixer = [
            P1, P2, naphtha_mixer, naphtha_splitter, HG,
            HG, HX1, C1, HX2
        ]
        for i in naphtha_to_before_mixer: i._run()
        HT_feed = HT.ins[0]
        hydrogen_produced = HG.outs[0].imol['H2']
        hydrogen_consumed = HT.hydrogen_requirement(HT_feed) + 1e-6
        
        # Adjust fresh hydrogen or naphtha split accordingly
        H2_fresh = hydrogen_consumed - hydrogen_produced
        if H2_fresh > 0:
            fresh_hydrogen.imol['H2'] = H2_fresh
            for i in [H2_mixer, HT]: i._run()
        else:
            naphtha_splitter.split[:] = hydrogen_consumed / hydrogen_produced
            for i in [naphtha_splitter, HG, HX1, C1, HX2, H2_mixer, HT]: i._run()
    
    T1 = bst.IsentropicTurbine(ins=HT-0, P=101325)
    off_gas = bst.Stream()
    distillation_sys = pyrolysis.create_fractional_distillation_system(
        ins=T1-0, outs=(off_gas, residual_naphtha, diesel, LFO), mockup=True
    )    
    MA = pyrolysis.MechanicalActivation(ins=solids, outs=['pretreated_char', metals])
    MA.register_alias('mechanical_activation')
    char, ash = MA.outs
    water = bst.Stream(
        price=0.00021133774, # 0.8 USD / 1000 gal
    )
    char_syngas = bst.Stream()
    RK = pyrolysis.RotaryKiln(ins=[char, water, 'air'], outs=[activated_carbon, char_syngas, 'emissions'])
    RK.register_alias('rotary_kiln')
    combustible_mixer = bst.Mixer(ins=[naphtha_splitter-1, HG-1, unused_syngas, char_syngas, off_gas], outs='gas_to_boiler')
    bst.BoilerTurbogenerator(ins=[combustible_mixer-0])
    bst.CoolingTower()

def test_waste_tire_pyrolysis_system():
    import numpy as np
    import pyrolysis
    import biosteam as bst
    bst.settings.set_thermo(pyrolysis.create_chemicals(), pkg='ideal gas')
    sys = pyrolysis.create_waste_tire_pyrolysis_system()
    sys.simulate()
    tea = pyrolysis.create_tea(sys)
    feed, = sys.ins
    diesel, LFO, metals, activated_carbon = sys.outs
    np.testing.assert_allclose(
        [feed.F_mass, diesel.F_mass, LFO.F_mass, metals.F_mass, activated_carbon.F_mass],
        [6250.0, 1315.95218259095, 127.55855512231291, 843.7500000000001, 381.6095151143242],
    )
    np.testing.assert_allclose(
        tea.solve_IRR(),
        0.051151, # Without tipping fee (for now).
    )
    
if __name__ == '__main__':
    test_waste_tire_pyrolysis_system()
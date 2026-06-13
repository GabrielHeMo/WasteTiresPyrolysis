# -*- coding: utf-8 -*-
"""
"""
import biosteam as bst
import pyrolysis

__all__ = (
    'create_pyrolysis_system',
)

@bst.SystemFactory(
    ins=[dict(ID='feedstock', Tire=6250, units='kg/hr'),
         dict(ID='hydrogen', phase='g')],
    outs=[dict(ID='diesel', price=1.06),
          dict(ID='LFO', price=1.06),
          dict(ID='metals', price=0.237),
          dict(ID='activated_carbon', price=1.5)]
)
def create_pyrolysis_system(ins, outs):
    feed, hydrogen, = ins
    diesel, LFO, metals, activated_carbon = outs
    syngas_recycle = bst.Stream()
    R1 = pyrolysis.PyrolysisReactor(ins=[feed, syngas_recycle], tau=14)
    vapor, solids, unused_syngas, emissions = R1.outs
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
    H2_mixer = bst.Mixer(ins=[HX2-0, hydrogen]) 
    P3 = bst.Pump(ins=fuel_oil, P=P_hydrotreater)
    HT = pyrolysis.Hydrotreater(ins=[P3-0, H2_mixer-0])
    
    @HT.add_specification
    def adjust_naphtha_recycling():
        # Find maximum hydrogen production
        naphtha_splitter.split[:] = 1
        naphtha_to_HG = [P1, P2, naphtha_mixer, naphtha_splitter, HG]
        HG_to_HT = [HX1, C1, HX2, H2_mixer, HT] 
        naphtha_to_HT = naphtha_to_HG + HG_to_HT
        for i in naphtha_to_HG: i.run()
        HT_feed = HT.ins[0]
        hydrogen_produced = HG.outs[0].imol['H2']
        hydrogen_consumed = HT.hydrogen_requirement(HT_feed)
        
        # Adjust fresh hydrogen or naphtha split accordingly
        if hydrogen_consumed > hydrogen_produced:
            hydrogen.imol['H2'] = hydrogen_consumed - hydrogen_produced
            for i in HG_to_HT: i.run()
        else:
            hydrogen.imol['H2'] = 0
            naphtha_splitter.split[:] = hydrogen_consumed / hydrogen_produced
            for i in naphtha_to_HT: i.run()
    
    T1 = bst.IsentropicTurbine(ins=HT-0, P=101325)
    off_gas = bst.Stream()
    distillation_sys = pyrolysis.create_fractional_distillation_system(
        ins=T1-0, outs=(off_gas, residual_naphtha, diesel, LFO), mockup=True
    )    
    MA = pyrolysis.MechanicalActivation(ins=solids, outs=['pretreated_char', metals])
    char, ash = MA.outs
    RK = pyrolysis.RotaryKiln(ins=char, outs=activated_carbon)
    combustible_mixer = bst.Mixer(ins=[naphtha_splitter-1, HG-1, unused_syngas, off_gas], outs='gas_to_boiler')
    bst.BoilerTurbogenerator(ins=[combustible_mixer-0])
    bst.CoolingTower()


def test_pyrolysis_system():
    import pyrolysis
    import biosteam as bst
    from biorefineries.tea import create_cellulosic_ethanol_tea
    bst.settings.set_thermo(pyrolysis.create_chemicals(), pkg='ideal gas')
    sys = pyrolysis.create_pyrolysis_system()
    sys.simulate()
    tea = create_cellulosic_ethanol_tea(sys)
    breakpoint()
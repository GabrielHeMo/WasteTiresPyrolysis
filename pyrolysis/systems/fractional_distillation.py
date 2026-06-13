# -*- coding: utf-8 -*-
"""
"""
import pyrolysis
import biosteam as bst
import numpy as np


__all__ = (
    'create_fractional_distillation_system',
)

@bst.SystemFactory(
    ins=[dict(ID='hydrotreated_feed')],
    outs=[dict(ID='off_gas'),
          dict(ID='residual_naphtha'),
          dict(ID='diesel'),
          dict(ID='LFO')]
)
def create_fractional_distillation_system(ins, outs):
    feed, = ins
    off_gas, residual_naphtha, diesel, LFO = outs
    heavy = bst.Stream()
    HXi = bst.HXprocess(ins=[feed, heavy])
    F1 = bst.Flash(ins=HXi-0, outs=[off_gas, heavy], T=40 + 273.15, P=101325)
    D1 = bst.BinaryDistillation(
        ins=HXi-1, outs=[residual_naphtha, 'heavy_product'], 
        LHK=('cyclohexane', 'dodecane'),
        Hr = 0.999999,
        Lr = 0.999999,
        P=101325 * 0.2,
        k=1.25,
        partial_condenser=False,
    )
    D1.check_LHK = False
    
    @D1.add_specification
    def water_to_distillate():
        D1.run()
        residual_naphtha, heavy_product = D1.outs
        residual_naphtha.imol['Water'] += heavy_product.imol['Water']
        heavy_product.imol['Water'] = 0
    
    S1 = bst.Splitter(
        ins=D1-1, 
        outs=(diesel, LFO),
        split=1 - 0.0883668903803132,
    )

def test__fractional_distillation_system():
    pass
    
if __name__ == '__main__':
    pass
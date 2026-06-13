import pandas as pd 
import biosteam as bst
import numpy as np 
import os
from chemicals.elements import periodic_table

__all__ = (
    'create_chemicals',
)

def User_mu_model_isodurene(T , C1 = -12.343  , C2 = 1688.4, C3 = -0.0041458 , C4 = 0, C5 = 0):  # C10H14E6
   if T > 249.46 or T < 471.15:
      return  np.exp(C1 + C2 / T + C3 * np.log(T) +  C4 * T**C5 )
   elif T < 249.46 or T > 471.15 :
      T_lim = 471.15
      mu_lim = np.exp(C1 + C2 /T_lim + C3 * np.log(T_lim) +  C4 * T_lim**C5 )
      dln_eta_dT = (-C2 / T_lim**2) + (C3 / T_lim) + (C4 * C5 * T_lim**(C5 - 1))
      deta_dT = mu_lim * dln_eta_dT
      return mu_lim  + deta_dT * (T - T_lim)  # linear extrapolation

def User_mu_model_hexylbenzene(T , C1 = 77.453  , C2 = -7677.7, C3 = -11.908, C4 = 851050, C5 = -1.9982):  # C12H18
   if T > 206.3 or T < 683.15:
      return  np.exp(C1 + C2 / T + C3 * np.log(T) +  C4 * T**C5 )
   elif T < 206.3 or T > 683.15:  
      T_lim = 683.15
      mu_lim = np.exp(C1 + C2 /T_lim + C3 * np.log(T_lim) +  C4 * T_lim**C5 )
      dln_eta_dT = (-C2 / T_lim**2) + (C3 / T_lim) + (C4 * C5 * T_lim**(C5 - 1))
      deta_dT = mu_lim * dln_eta_dT
      return mu_lim  + deta_dT * (T - T_lim)

def User_mu_model_methylnaphthalene(T , C1 = 36.99  , C2 = -2115.1, C3 = -6.6142 , C4 = 25556000000, C5 = -4.0889):  
   if T > 242.67 or T < 517.83:
      return  np.exp(C1 + C2 / T + C3 * np.log(T) +  C4 * T**C5 )
   elif T < 242.67 or T > 517.83 :
      T_lim = 517.83
      mu_lim = np.exp(C1 + C2 /T_lim + C3 * np.log(T_lim) +  C4 * T_lim**C5 )
      dln_eta_dT = (-C2 / T_lim**2) + (C3 / T_lim) + (C4 * C5 * T_lim**(C5 - 1))
      deta_dT = mu_lim * dln_eta_dT
      return mu_lim  + deta_dT * (T - T_lim)

def User_mu_model_ethylnaphthalene(T , C1 = -127.59 , C2 = 6980.2 , C3 =17.487, C4 = -1.214E-05, C5 = 2):  # C12H12
   if T > 259.34 or T < 620.8:
      return  np.exp(C1 + C2 / T + C3 * np.log(T) +  C4 * T**C5 )
   elif T < 259.34 or T > 620.8:
      T_lim = 620.8
      mu_lim = np.exp(C1 + C2 /T_lim + C3 * np.log(T_lim) +  C4 * T_lim**C5 )
      dln_eta_dT = (-C2 / T_lim**2) + (C3 / T_lim) + (C4 * C5 * T_lim**(C5 - 1))
      deta_dT = mu_lim * dln_eta_dT
      return mu_lim  + deta_dT * (T - T_lim)

nonequilibrium_components = frozenset([
    'Rubber',
    'Ash',
    'Ca(OH)2',
    'SO2',
    'CaSO4',
    'Char',
])

def create_chemicals():
    data_folder = os.path.join(os.path.dirname(__file__), 'data')
    path = os.path.join(data_folder, 'Data_components_clean.csv')
    df = pd.read_csv(path, encoding='latin1')
    
    chemicals = bst.Chemicals([
        bst.Chemical('Rubber', db='BioSTEAM'),
        bst.Chemical('Ash', db='BioSTEAM'),
        bst.Chemical('Char', db='BioSTEAM'),
        bst.Chemical('Ca(OH)2', search_ID='1305-62-0', phase='s', default=True),
        bst.Chemical('CaSO4', phase='s', default=True),
        bst.Chemical('SO2'),
        bst.Chemical('dodecane'),
    ])
    IDs = df['Component ID']
    names = df['common_names']
    CAS = df['CAS NUM']
    gases = {
        # 'molecular hydrogen', 
        # 'hydrogen sulfide',
        # 'molecular oxygen',
        # 'molecular nitrogen',
        # 'carbon dioxide', 
        # 'carbon monoxide',
        # 'methane',
        # 'ethane',
        # 'propane'
    }
    for i, available in enumerate(df['Available DataBase']): # Search compounds in database
        if not available: continue 
        ID = IDs.iloc[i]
        if ID == 'carbon': phase = 's'
        elif ID in gases: phase = 'g'
        else: phase = None
        component = bst.Chemical(
            ID, search_ID=CAS.iloc[i], aliases=[names.iloc[i]],
            search_db=True, default=True, phase=phase,
        )
        chemicals.append(component)
    chemicals.compile()
    chemicals.set_alias('ammonia', 'NH3')
    chemicals['527-53-7'].mu.l.add_method(f=User_mu_model_isodurene, Tmin = 0, Tmax=1000)  # '1,2,3,5-tetramethylbenzene0'  
    chemicals['527-53-7'].mu.l.method = 'USER_METHOD'
    chemicals['527-53-7'].mu.l.method_P = None

    chemicals['hexylbenzene'].mu.l.add_method(f=User_mu_model_hexylbenzene, Tmin = 0 , Tmax= 1000)
    chemicals['hexylbenzene'].mu.l.method = 'USER_METHOD'
    chemicals['hexylbenzene'].mu.l.method_P = None

    chemicals['1-methylnaphthalene'].mu.l.add_method(f=User_mu_model_methylnaphthalene, Tmin = 20 , Tmax=1000)
    chemicals['1-methylnaphthalene'].mu.l.method = 'USER_METHOD'
    chemicals['1-methylnaphthalene'].mu.l.method_P = None

    chemicals['1-ethylnaphthalene'].mu.l.add_method(f=User_mu_model_ethylnaphthalene, Tmin = 259.34 , Tmax=1000)
    chemicals['1-ethylnaphthalene'].mu.l.method = 'USER_METHOD'
    chemicals['1-ethylnaphthalene'].mu.l.method_P = None
    adjust_tire_composition(chemicals=chemicals)
    return chemicals

def adjust_tire_composition(
        H=7, O=2.7, N=0.3, C=75, S=1.5, # Elemental composition
        rubber=85, ash=13.5, moisture=1.5, # Overall composition
        chemicals=None,
    ):
    if chemicals is None: chemicals = bst.settings.chemicals
    chemicals.define_group(
        'Tire',
        ['Rubber', 'Ash', 'Water'], 
        [rubber, ash, moisture],
        wt=True
    )
    formula = dict(
        H=H, O=O, N=N, C=C, S=S
    ) # Elemental composition is on a dry basis
    MW = sum(formula.values())
    for i, j in formula.items(): 
        formula[i] = j / MW / periodic_table[i].MW # Normalize to 1 kg and set to molar basis
    
    chemicals.Rubber.reset_combustion_data(
        method='Specification',
        HHV=35e3, # kJ / kmol or kJ / kg
        formula=formula,
        Hf=None,
        LHV=None,
    )
    chemicals.refresh_constants() # Refreshes atomic data and 

def test_chemicals():
    chemicals = create_chemicals()
    bst.settings.set_thermo(chemicals)
    assert bst.settings.chemicals.size == 114
    adjust_tire_composition()
    stream = bst.Stream(Rubber=85)
    np.testing.assert_allclose(stream.F_mass, 85)
    stream.imass['Tire'] = 100
    np.testing.assert_allclose(
        stream.imass['Rubber', 'Ash', 'Water'],
        [85, 13.5, 1.5],
    )
  
if __name__ == '__main__':
    test_chemicals()
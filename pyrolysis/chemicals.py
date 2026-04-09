from __future__ import annotations
import pandas as pd 
import biosteam as bst
import numpy as np 
from biosteam import settings


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



def load_chemicals():
    df = pd.read_csv('Data_components_clean.csv', encoding='latin1')
    components = df['ComponentID'][2:].to_list()

    chemicals_list = []
    common_names_list  = [] 
    # Add the first dummy components
    chemicals_list.append(bst.Chemical('Tyre', search_db=False, default=True, Hvap = 1, Psat= 1, Tb = 1, MW = 1.1, common_name= 'Tyre'))
    chemicals_list.append(bst.Chemical('Ash', search_db=False, default=True, Hvap = 1, Psat= 1, Tb = 1, MW = 1.1, common_name= 'Ash')) # No carga nada, sigue igual
    for i in range(df['Available DataBase'].shape[0]):     # search compounds in database
        if df['Available DataBase'][i] == True: # Add component
            c = df['ComponentID'][i]
            component = bst.Chemical(c,search_db=True, default=True )
            chemicals_list.append( component )
            common_names_list.append(component.common_name )
        else: 
            common_names_list.append('NaN')
    chemicals_list.append(bst.Chemical('SO2'))
    chemicals_list.append(bst.Chemical('1305-62-0'))
    chemicals_list.append(bst.Chemical('CaSO4'))
    remove_components = 3
    # add other chemicals blank that must be specified for Tyres
    # Note: Ash does not have any property.
    df['common_names'] = common_names_list
   #  print('Number of chemical added', len(chemicals_list)-1)
    settings.set_thermo(chemicals_list, cache=True, ideal=True) 
    MW_list= settings.chemicals.MW
    #df.to_csv('Data_components_clean.csv', index=False)

    settings.chemicals['527-53-7'].mu.l.add_method(f=User_mu_model_isodurene, Tmin = 0, Tmax=1000)  # '1,2,3,5-tetramethylbenzene0'  
    settings.chemicals['527-53-7'].mu.l.method = 'USER_METHOD'
    settings.chemicals['527-53-7'].mu.l.method_P = None

    settings.chemicals['hexylbenzene'].mu.l.add_method(f=User_mu_model_hexylbenzene, Tmin = 0 , Tmax= 1000)
    settings.chemicals['hexylbenzene'].mu.l.method = 'USER_METHOD'
    settings.chemicals['hexylbenzene'].mu.l.method_P = None

    settings.chemicals['1-methylnaphthalene'].mu.l.add_method(f=User_mu_model_methylnaphthalene, Tmin = 20 , Tmax=1000)
    settings.chemicals['1-methylnaphthalene'].mu.l.method = 'USER_METHOD'
    settings.chemicals['1-methylnaphthalene'].mu.l.method_P = None

    settings.chemicals['1-ethylnaphthalene'].mu.l.add_method(f=User_mu_model_ethylnaphthalene, Tmin = 259.34 , Tmax=1000)
    settings.chemicals['1-ethylnaphthalene'].mu.l.method = 'USER_METHOD'
    settings.chemicals['1-ethylnaphthalene'].mu.l.method_P = None
    return settings


import biosteam as bst
import pandas as pd 
import numpy as np 
from math import exp 
from math import log as ln
from scipy.integrate import quad 
from scipy.integrate import solve_ivp

from chemicals.utils import R

import pyomo.environ as pyo
from collections import defaultdict

from biosteam.units import Flash
from biosteam.units import ThermalOxidizer
from biosteam import HXutility

from pyrolysis.tea import  PyrolyisisTEA
import os 

__all__ = ('Descomposer','Reactor', 
           'Naptha_PartialOxidation', 'Hydrotreatment',
           'Mechanical_Activation', 'PyrolysisReactor', 
           'ThermalOxidizer', 'create_system')


class Descomposer(bst.Unit):
    """
    Create a Descomposer reactor
    Parameters
    ----------
    ins :
        Inlet fluid.
    outs :
        * [0] multistream
    F : float   Mass flow of tyres [kg/hr]
    T : float   Operating temperature [C]
    P : float   Operating pressure [bar].

    """
    _N_ins = 1 # Number in inlets
    _N_outs = 1 # Number in outlets #
    _units = {'Area': 'm^2', 'Duty':'kJ/hr'} 

    def _init(self, Ttyres = None, Treac = None, P = None, P_moisture = None, P_ash = None,
                U_carbon = None, U_h = None, U_n = None, U_s = None , U_o = None , P_volatilemateria = None):
        # The _init methods adds input parameters for unit creation
        self.Ttyres = Ttyres #:  Tires temperature [K]
        self.Treac = Treac #:  Operation temperature [K]
        self.P = P #: Operating pressure [bar].
        self.P_moisture = P_moisture #comp['P_moisture']
        self.P_ash = P_ash #comp['P_ash']
        self.U_carbon   = U_carbon #comp['U_carbon']
        self.U_h  = U_h #comp['U_h2']
        self.U_n = U_n #comp['U_n2']
        self.U_s = U_s #comp['U_s']
        self.U_o =  U_o #comp['U_o2']
        self.P_volatilemateria = P_volatilemateria #comp['P_volatilemateria']
        self._load_CP_data()

    def _load_CP_data(self):
        output_folder = 'pyrolysis'
        os.makedirs(output_folder, exist_ok=True)

        output_path = os.path.join(output_folder, 'Cp_Data.csv')
        self.df_Cp = pd.read_csv(output_path)      # File to calcualte enthalpies

    def _run(self):
        # Equivalent to self.ins[0] when the number of inlets is one
        feed = self.feed
        
        # Get information of Proxanal and Ultanal values 
        Tyre_flow = feed.F_mass  # self.F # [kg/hr]
        H2OF, ASHF = self.P_moisture , self.P_ash
        CF , H2F  = self.U_carbon , self.U_h
        N2F , SF = self.U_n , self.U_s
        O2F =  self.U_o
        # print('llantas comp', self.P_moisture, self.P_ash, self.U_carbon, self.U_h2, self.U_n2, self.U_s, self.U_o2 )
        self.H2OOUT= Tyre_flow*(H2OF/100)
        self.ASHOUT= Tyre_flow*(ASHF/100*(1- H2OF/100))  
        self.COUT = Tyre_flow*(CF/100*(1-H2OF/100))
        self.H2OUT = Tyre_flow*(H2F/100*(1-H2OF/100))
        self.N2OUT = Tyre_flow*(N2F/100*(1-H2OF/100))
        self.SOUT = Tyre_flow*(SF/100*(1-H2OF/100))
        self.O2OUT = Tyre_flow*(O2F/100*(1-H2OF/100))
        
        flows_gas = [('water', self.H2OOUT), ('hydrogen',self.H2OUT), ('nitrogen', self.N2OUT), 
                        ('sulfur',self.SOUT), ('molecular oxygen',self.O2OUT)]
        flows_solid = [('Ash', self.ASHOUT), ('carbon', self.COUT)]
        self.out_flow_gas  = flows_gas
        self.out_flow_solid = flows_solid
        tyre_1 = bst.MultiStream(ID='s1',phases = ('g', 's'), s = flows_solid, g = flows_gas, T=self.Treac, P=self.P, units='kg/hr',)
        product = self.outs[0]
        product.phases = ('g', 'l', 's')
        product['g'].copy_like(tyre_1['g'])
        product['s'].copy_like(tyre_1['s'])

    def _heatcombustion_Boie(self, a1 = 151.2, a2 = 499.77, a3 = 45.0, 
                                a4 = -47.7, a5 = 27.0, a6 = -189.0 ):
        #  Btu/lb 
        Tyre_flow = self.F # [kg/hr] to [lb/hr]
        wCF   =  self.CF/100
        wH2F  =  self.H2OF /100
        wN2F = self.N2F /100
        wSF = self.SF/100
        wO2F =  self.O2F/100
        heat_combustion_dm = (a1*wCF + a2*wH2F  + a3*wSF  + a4*wO2F  + a5*wN2F)*10**2 +a6 
        return heat_combustion_dm

    def Cp_ij(self,T, j, param):
        return param[j,0] + param[j,1]*T + param[j,2]*T**2 + param[j,3]*T**3

    def Cp_dl(self, T, w, param):
        return sum(w[j] * self.Cp_ij(T, j, param ) for j in range(len(w)))

    def _Heat_Capacity_Kirov_Correlations(self, Flow, T_ref , Tint, moisture, fixed_carbon, primary_volatile_matter,secondary_volatile_matter, ash):

        param = np.array([[1.0, 0 , 0 , 0 ],
                        [0.165, 6.8e-4, -4.2e-7, 0 ],
                        [0.395, 8.1e-4, 0 , 0 ],
                        [0.71, 6.1e-4, 0 , 0 ],
                        [0.18, 1.4e-4, 0 , 0] ])
        # create a numpy array with the mass fractions of jth constituents 
        # 1 Moisture, 2 Fixed carbon    , 3 Primary volatile matter
        # 4 Secondary volatile matter   , 5 Ash
        if primary_volatile_matter > 10:
            weights = np.array([moisture,fixed_carbon, primary_volatile_matter, secondary_volatile_matter, ash])
            weights  = weights/(1 - moisture/100) #total_dry
            val = 10.3
            weights[3]  =  weights[2] -val #
            weights[2]  =  val # 10.3 # Standar value to extract 

        else:
            weights = np.array([moisture,fixed_carbon, primary_volatile_matter, secondary_volatile_matter, ash])

        integral, error = quad(self.Cp_dl, T_ref , Tint , args=(weights,param))  # cal/gram-C
        Enthalpy_flow = integral*Flow*1000/3600  # cal/gram * kg/hr * 1000 gram/kg * hr/sec
        return Enthalpy_flow   # [cal/sec]

    def Cp_model(self, T, C):
        term1 = C[0]
        term2 = C[1] * ( (C[2]/T) / np.sinh(C[2]/T))**2
        term3 = C[3] * ( (C[4]/T) / np.cosh(C[4]/T))**2
        return term1 + term2 + term3

    def integrar_Cp_vapour_fase(self,Tref, Tint, C):
        resultado, _ = quad(self.Cp_model,Tref, Tint,  args=(C))
        return resultado

    def _H_out_vapour_fase(self):
        Tref , Tint = 25+273.15 , self.outs[0].T
        flows_gas = self.out_flow_gas  
        species = [specie[0] for specie in flows_gas] 
        flows = np.array([specie[1] for specie in flows_gas]) # kg/hr
        integral = 0
        for i, specie in enumerate(species):
            C = self.df_Cp.loc[self.df_Cp['common _names'] == specie,['1','2','3','4','5']].values[0]
            mw = self.df_Cp.loc[self.df_Cp['common _names'] == specie,'mw'].values[0]
            integral += self.integrar_Cp_vapour_fase( Tref, Tint ,C)/mw * (flows[i]) *1000/3600  # kmol/hr
        return integral

    def _Cp_carbon(self, T):
        # Cp_intC	[cal/mol-K]	
        param = np.array([1.827126898,	0.00835692034	,-7.870513E-06,	-4.95653E-08,	0, -246.86,	27.42])
        if T > param[6] : # Is outside bound 
            # Slope
            T_lim = param[6]
            # Evaluate polinomiun at limit
            Cp_zlim = (param[0])*T + (param[1]/2)*T**2 + (param[2]/3)*T**3 + (param[3]/4)*T**4 + (param[4]/5)*T**5 
            m =  param[0]  + (param[1])*T_lim + (param[2])*T_lim**2 + (param[3])*T_lim**3 + (param[4])*T_lim**4
            Cp_int = Cp_zlim + m *(T - T_lim )
        else:
            Cp_int = (param[0])*T + (param[1]/2)*T**2 + (param[2]/3)*T**3 + (param[3]/4)*T**4 + (param[4]/5)*T**5
        return Cp_int

    def _H_out_carbon(self, mw = 12):
        flows_solid = self.out_flow_solid  
        flows = np.array([specie[1] for specie in flows_solid]) # kg/hr
        Tref , Tint = 25 , self.outs[0].T - 273.15  # Temperatures must be in Kelvin
        enthalpy = (self._Cp_carbon(Tint) - self._Cp_carbon(Tref))/mw *flows[1] *1000/3600
        return enthalpy

    def _H_in(self):
        feed = self.feed
        # Calculate enthalpy of inlet stream (tyres at 20 C)
        Tyre_flow = feed.F_mass # [kg/hr]
        Ttyres = 20 # K a C 
        moisture   = self.P_moisture
        fixed_carbon  =  self.U_carbon
        primary_volatile_matter = self.P_volatilemateria
        secondary_volatile_matter = 0
        ash = self.P_ash
        T_ref = 25
        # Enthalpy in 
        H_in = self._Heat_Capacity_Kirov_Correlations(Tyre_flow, T_ref , Ttyres, moisture, fixed_carbon, 
                        primary_volatile_matter,secondary_volatile_matter, ash )
        return H_in

    def _H_out_solid(self):
        out = self.outs[0].copy()       # Stream
        # ANALIZAR SOLO LAS CENIZAS (ASH)
        Ash_flow, Carbon_flow = out.imol['s', ('Ash', 'carbon')]
        T_operation = 20 
        T_ref= 25
        H_out = self._Heat_Capacity_Kirov_Correlations(Ash_flow, T_ref,  T_operation, 0, 0, 0, 0, 100 )
        return H_out

    def _design(self):
        # Calculate duty
        H_in = self._H_in()    #[cal/sec]
        H_out_ash = self._H_out_solid() 
        H_out_carbon = self._H_out_carbon()
        H_out_vapour = self._H_out_vapour_fase()
        H_out_solid =  H_out_ash + H_out_carbon
        H_out = H_out_solid + H_out_vapour
        # Enthalpy out
        duty = (H_out - H_in)*15.072479976493  # cal/seg a KJ/hr
        self.duty = duty
        if duty < 0: raise RuntimeError(f'{repr(self)} is cooling.')
        T_operation = self.Treac 
        heat_utility = self.add_heat_utility(duty, T_in = self.Ttyres, T_out = T_operation) # New utility is also in self.heat_utilities
      
    def compute_furnace_purchase_cost(self, Q, CE): 
        # Q - duty Btu/hr
        return exp(-0.15241 + 0.785*ln(Q))*CE/567

    def _cost(self):
        # Itemized purchase costs are stored here
        Q = self.duty * 0.947817 
        self.baseline_purchase_costs['Descomposer'] =  self.compute_furnace_purchase_cost(Q, bst.CE) #purchase_cost # Not accounting for material factor
        # Assume design, pressure, and material factors are 1.
        self.F_D['Descomposer'] = self.F_P['Descomposer'] = self.F_M['Descomposer'] = 1.

class Reactor(bst.Unit):
    """
    Create the kinetic reactor        
    Parameters
    ins :    Inlet fluid.
    outs :  * [0] vapor product     * [1] solid product
    F : float   Mass flow of tyres [kg/hr]
    T : float   Operating temperature [K]
    P : float   Operating pressure [bar].
    """
    _N_ins , _N_outs = 1 , 3 # Number in inlets,  Number in outlets 
    _units = {'Area': 'm^2', 'Duty':'cal/sec'}

    def _init(self, Treac = None, Preac = None, velocity = None, carbon_conversion = None,
               processing_capacity = None, settings = None): # diameter, length):
        # The _init methods adds input parameters for unit creation
        self.Treac = Treac #:  Operation temperature [K]
        self.Preac = Preac #: Operating pressure [bar].
        # self.diameter = diameter
        # self.length =  length
        self.processing_capacity = processing_capacity
        self.velocity = velocity
        self.carbon_conversion = carbon_conversion
        self.Baseline_volumen = 0.04047958530116809
        # Load setting
        self.settings = settings
        # Functions
        self._get_MW()
        self._load_kinetik_data()
        self._get_kinetic_data()

    def _get_MW(self):
        self.MW_list= self.settings.chemicals.MW
    
    def _get_chemicals(self):
        self.chemicals = self.settings.chemicals

    def _load_kinetik_data(self):
        output_folder = 'pyrolysis'
        os.makedirs(output_folder, exist_ok=True)

        output_path = os.path.join(output_folder, 'kinetic_main_original.csv')

        self.df_kinetics  = pd.read_csv(output_path)

    def _get_kinetic_data(self):
        reactions_list, k_constants, n_constants , energy_constants , type_of_reaction = [], [], [] , [] ,  []
        number_reactions = 0
        df_kinetics = self.df_kinetics
        for i in range(df_kinetics.shape[0]):
            if df_kinetics.loc[i,'Available'] == 'Yes':
                k_constants.append(  df_kinetics.loc[i,'k_constant']  )
                n_constants.append(  df_kinetics.loc[i,'n' ]  )
                energy_constants.append( df_kinetics.loc[i,'E[kj/mol]']  )
                type_of_reaction.append( df_kinetics.loc[i, 'Dependant_index'  ])

                Product_name = df_kinetics.loc[i,'Common_name']
                C_coefficient, H_coefficient = df_kinetics.loc[i,'C'], df_kinetics.loc[i,'H']
                O_coefficient, S_coefficient = df_kinetics.loc[i,'O'], df_kinetics.loc[i,'S']
                N_coefficient, P_coefficiente = df_kinetics.loc[i,'N'], df_kinetics.loc[i,'Product']
                reactions_list.append( bst.Reaction({'C': -C_coefficient, 'molecular hydrogen': -H_coefficient,
                                    'nitrogen': -N_coefficient ,  'sulfur' : -S_coefficient,  'molecular oxygen': -O_coefficient ,    Product_name: P_coefficiente},  
                                    correct_atomic_balance=False, reactant= df_kinetics.loc[i,'Reactant']) )
                number_reactions += 1
        # Parámetros de reacción
        self.number_reactions = number_reactions
        self.k_constants = np.array(k_constants)  # Constantes cinéticas 
        self.E_ = np.array(energy_constants) *1000   # Energía de activación en kJ/mol a J/mol
        self.n_ = np.array(n_constants)  # Exponentes de temperatura
        self.type_of_reaction = type_of_reaction
        # Índices de las especies
        self.H2_index, self.S_index ,  self.O2_index= 3 , 5 , 6
        # Cargar reacciones pero hay que modificar esto 
        #reactions = bst.ParallelReaction(reactions_list)
        #stoichiometry = reactions.stoichiometry
        #np.savetxt("foo.csv", stoichiometry, delimiter=",")
        #reactions.show()

        output_folder = 'pyrolysis'
        os.makedirs(output_folder, exist_ok=True)

        output_path = os.path.join(output_folder,'stoichometry_main.csv')

        stoichiometry  = pd.read_csv(output_path, header=None)
        stoichiometry = stoichiometry.to_numpy()
        stoichiometry[:,0:8] = stoichiometry[:,0:8] *-1   # Reactants
        self.stoichiometry = stoichiometry

    def _reaction_rate(self, x, type_of_reaction, k, n, E, mw, T , To, P, Po ):
        """ 
        MW is a list containing all molecular weights of the components ha
        Flujo_volumetrico_init se considera constante para 6250 kg/hr 
        """
        gas_flow_mass = sum(x[i]*self.MW_list[i] for i in range(3,len(x)) ) #Flujo masico total
        if type_of_reaction == 1:  # Limitant reactive is hydrogen
            X_H2 = (x[self.H2_index]*2.01568) / gas_flow_mass  
            return (k * (T**n) * np.exp(-E / (R * T)) *( X_H2 ))  # mol/seg
        elif type_of_reaction == 2:  # Limitant reactive is oxygen
            X_O2 = (x[self.O2_index]*31.999) / gas_flow_mass
            return (k * (T**n) * np.exp(-E / (R * T)) *( X_O2  ) )  # mol/seg
        elif type_of_reaction == 3:  # Limitant reactive is oxygen
            X_S = (x[self.S_index]*32.065) / gas_flow_mass
            return (k * (T**n) * np.exp(-E / (R * T)) *( X_S  )) # mol/seg
        elif type_of_reaction == 4:  # The component is not changing (inert)
            return  0 # Water molar flux is constant
   
    def ode(self, t, x):
        rates = []
        Treac = self.Treac
        Preac = self.Preac
        To = self.Tstream
        Po = self.Pstream
        for i in range(self.number_reactions):
            rates.append(self._reaction_rate(x ,  self.type_of_reaction[i], 
                        self.k_constants[i], self.n_[i], self.E_[i], self.MW_list, T=Treac, To= To , 
                        P = Preac, Po = Po ) )  
        rates = np.array(rates).reshape([-1, 1])
        dy = rates * self.stoichiometry
        result = dy.sum(axis=0)
        return result
        
    def _get_molar_flows(self):
        stream_ex = self.ins[0]
        self.carbon_init  = stream_ex['s'].imol['C'] /3600  # Flujo molar [kmol/sec]
        self.hydrogen_init = stream_ex.imol['H2'] /3600 
        self.molecular_oxygen_init = stream_ex.imol['molecular oxygen'] /3600
        self.sulfur_init = stream_ex.imol['sulfur'] /3600 #          
        self.nitrogen_init = stream_ex.imol['nitrogen']  /3600 #     
        self.water_init = stream_ex.imol['oxidane']  /3600  #        
        self.ash_init = 0
        self.tyre_init = 0 
        self.total_molar_flow_gas = stream_ex['g'].F_mol/3600   # Flujo molar en la fase gaseosa total [kmol/seg]

    def _design(self):
        """ 
        Revisar entalpias de la corriente, solo se considera la entalpia de los vapores pero no de los solidos ! 
        """
        feed = self.ins[0] 
        product1 , product2, product3 = self.outs
        # Calculate only the enthalpy of vapours
        H_out_vapour_in = feed['g'].Hnet  
        H_out_vapor_out = product1['g'].Hnet #
        # Enthalpy out
        duty = H_out_vapor_out - H_out_vapour_in   #   KJ/hr
        # Recalcular la parte de las entalipa, la estimada con los gases esta muy abajo de lo esperado... O hacerlo mas riguroso. 
        self.duty =  duty # *1.3 # Correcting factor because carbon and ash enthalpies are not considered. 
        # Temperatures
        T_in = feed.T  # Temperature of the inlet stream 
        T_out = product1.T # Temperature of outlet stream
        iscooling = duty < 0.
        if iscooling: 
            if T_out > T_in:
                T_in = T_out
        else:
            if T_out < T_in:
                T_out = T_in
        self.add_heat_utility(self.duty, T_in, T_out, hxn_ok=True)
        
    def _cost(self, Fm = 1, Fp = 0, Ntubes = 1, Index = 716.2):
        """ 
        Ntube : Number of tubes
        Ltube : Length of tubes (m)
        Fm : 1 (carbon steel)
        Fp : 0 (less than 20 bar)
        Ntubes : is considered as 1 
        Index(2019) = M&S (1716.2) source (Devaraja and Kiss, 2022)
        """
        # Fixed investment costs      
        if self.processing_capacity == 6250: 
            Pyrolysis_unit = 24544000.00 
            Nitrogen_unit =  495000.00 
            Cooling_unit =   48000.00 
        elif self.processing_capacity == 4166: 
            Pyrolysis_unit =  17230000.00  
            Nitrogen_unit =  495000.00 
            Cooling_unit =   48000.00 
        elif self.processing_capacity == 2500:
            Pyrolysis_unit = 11870000.00 
            Nitrogen_unit =  495000.00 
            Cooling_unit =   48000.00 
        # Calculated capital costs
        # A = np.pi*self.diameter*(Ntubes)*self.length  
        # CReactor = (Index/280)*(474.7*A**0.65)*(2.29+Fm*(0.8+Fp))
        # Calculate operating costs
        self.baseline_purchase_costs['MainReactor'] =   (Pyrolysis_unit)+  Nitrogen_unit  #*(self.Volumen/self.Baseline_volumen)**(0.6) +  Nitrogen_unit
        # Assume design, pressure, and material factors are 1.
        self.F_D['MainReactor'] = self.F_P['MainReactor'] = self.F_M['MainReactor'] = 1.

    def yield_error(self, L ):
        self.length = round(L,2)
        Volumen_Reac = (np.pi*self.length*(self.diameter/2)**2)
        time_points = np.linspace(0, Volumen_Reac, 5)
        sol = solve_ivp(self.ode, [0, Volumen_Reac], self.initial_conditions, t_eval=time_points, method='RK45')
        results = sol.y.T  #  (11, ncomps)
        self.final_flows = (results*self.MW_list)*3600  # kg/hr
        # Yield de carbono
        current_yield = self.final_flows[-1,2]/self.processing_capacity  # Base de calculo
        # Error de carbono
        error = abs(current_yield - self.carbon_yield) 
        return error

    def _run(self):
        feed = self.ins[0] 
        product1 , product2, product3 = self.outs
        self.Tstream = feed.T 
        self.Pstream = feed.P
        # Condición inicial de entrada
        self._get_molar_flows()
        # Nota: Si agrego algun componente adicional, 1 .- tengo que modificar 
        self.initial_conditions =  [self.tyre_init, self.ash_init, self.carbon_init, self.hydrogen_init, self.nitrogen_init,
                              self.sulfur_init, self.molecular_oxygen_init, self.water_init ]
        self.initial_conditions.extend([0] * (len(self.settings.chemicals)-8))  # 106
        
        # Calcular diametro necesario para lograr esa velocidad
        Q_Ls  = ((self.total_molar_flow_gas*1000*self.Tstream*8.314)/(self.Pstream))   #  m3/hr
        self.flujo_volumetrico_init  = Q_Ls   # m³/hr  #okey da lo mismo [m3/hr]

        # Calcular ahora la longitud necesaria para lograr cierta conversion 
        def rendimiento_event(t,y):
            flujo_producto = y[2] * self.MW_list[2] * 3600 
            # Flujo de carbono/ condicion inicial
            consumido = (self.initial_conditions[2]* self.MW_list[2] * 3600) -  flujo_producto
            rendimiento = consumido / (self.initial_conditions[2]* self.MW_list[2] * 3600) 
            # print('rendimiento', rendimiento, 'self.carbon_conversion/100', self.carbon_conversion/100)
            return rendimiento - self.carbon_conversion/100

        # Debe ser un proceso iterativo:
        rendimiento_event.terminal= True
        self.diameter = 0.5 # [m]
        self.max_length  = 8 # [m]
        Volumen_Reac = (np.pi*self.max_length*(self.diameter/2)**2)  # (np.pi*1.5*(0.5/2)**2)
        # time_points = np.linspace(0, Volumen_Reac, 11)
        sol = solve_ivp(self.ode, [0, Volumen_Reac], self.initial_conditions, 
                        events=rendimiento_event, #t_eval=time_points
                        method='RK45', max_step = 0.001 )
        # Sacar resultados 
        results = sol.y.T  # 
        self.final_flows = (results*self.MW_list)*3600  # kg/h
        carbon_consumido = (self.initial_conditions[2]* self.MW_list[2] * 3600) - self.final_flows[-1,2]
        self.real_Carbonconversion =  carbon_consumido/( (self.initial_conditions[2]* self.MW_list[2] * 3600) )


        if self.real_Carbonconversion > self.carbon_conversion/100: 
            specification = self.carbon_conversion
            try:
                self.carbon_conversion = (self.real_Carbonconversion*1.1)*100
                #     # Entonces vuelve a resolver pero ahora actualiza la condicion de conversion para encontrar una solucion 
                sol = solve_ivp(self.ode, [0, Volumen_Reac], self.initial_conditions, 
                            events=rendimiento_event, #t_eval=time_points
                            method='RK45', max_step = 0.001 )
                results = sol.y.T  # 
            finally:
                self.carbon_conversion = specification
            self.final_flows = (results*self.MW_list)*3600  # kg/hr
        
        
        # Calcular rendimiento de carbono por llanta
        self.carbon_yield = self.final_flows[-1,2]/self.processing_capacity  # 

        # Update information of each component in the gas phase
        flows_gas , flows_solid =  [] , []
        for i ,c in enumerate(self.chemicals): 
            if i > 2 and i < len(self.chemicals):
                flows_gas.append(( c.common_name , self.final_flows[-1,i]))
        flows_solid1 = [('Ash', feed['s'].imass['Ash'])]
        flows_solid2 = [('carbon', self.final_flows[-1,2])]
        self.out_flow_gas  = flows_gas
        self.out_flow_solid = flows_solid
        vapor = bst.Stream(None, phase = 'g',  T=self.Treac, P=self.Preac, units='kg/hr', **dict(flows_gas))
        solid1 = bst.Stream(None, phase = 's',  T=self.Treac, P=self.Preac, units='kg/hr', **dict(flows_solid1))
        solid2 = bst.Stream(None, phase = 's',  T=self.Treac, P=self.Preac, units='kg/hr', **dict(flows_solid2))
        product1.phases = ('g', 'l', 's')
        product1['g'].copy_like(vapor['g'])

        product2.phases = ('g', 'l', 's')
        product2['s'].copy_like(solid1['s'])
        # Agregar el costo de los metales

        product3.phases = ('g', 'l', 's')
        product3['s'].copy_like(solid2['s'])

class Naptha_PartialOxidation(bst.Unit):
    """
    Create the Gibbs reactor        
    Parameters
    ins :    Inlet fluid.
    outs :  * [0] vapor product     * [1] solid product
    F : float   Mass flow of tyres [kg/hr]
    T : float   Operating temperature [K]
    P : float   Operating pressure [bar].
    """
    _N_ins , _N_outs = 3 , 3 # Number in inlets,  Number in outlets 
    _units = {'Area': 'm^2', 'Duty':'cal/sec'}

    def __init__(self, ID='', ins=None, outs=(), T= None, P=None,  processing_capacity = None, 
                 settings = None , epsilon = 1e-8):
        super().__init__(ID, ins, outs)
        self.Treac = T  # Temperatura en K
        self.Preac = P  # Presión en pascales
        self.epsilon = epsilon # Minimum amount of flowrate for each chemical
        self.settings = settings
        self.processing_capacity = processing_capacity
        self.remove_components = 3

    def gibbs_energy_rule(self, m):
        # calculo la x masica para tenerla como una constraint implicita en la Obj y calcular 
        # la energia de gibbs en funcion de eso...
        x_vec = sum([ (m.x[i]*self.mw_list[i])/self.mass_flow for i in m.I])
        flow_dict = {str(self.components[i]) : pyo.value(m.x[i])  for i in m.I} # self.mass_flow debe ser constante
        s_current = bst.Stream(None, phase='g', T=self.Treac, P=self.Preac, units='kmol/hr', **flow_dict)
        Gibbs_energy =  s_current.Hnet - (s_current.S) * s_current.T
        return x_vec*Gibbs_energy

    def atom_balance_rule(self, m, a):
        sum_atom = sum(self.atoms_matrix[a, i] * m.x[i]  for i in m.I)
        return sum_atom == self.atoms_in[a]

    def flow_limit_rule(self, m, i):
        return m.x[i] == self.targetflows[i]

    def solve_pyomo(self, initial_guess ):
        # Pyomo model
        self.model = pyo.ConcreteModel()
        self.model.I = pyo.RangeSet(0, len(self.components)-1)  # Índices para especies
        self.model.x = pyo.Var(self.model.I, domain=pyo.NonNegativeReals, bounds=(0, None), initialize=lambda m, i: initial_guess[i])

        self.model.restricted_components = pyo.Set(initialize=self.restricted_indices) 
        self.model.flow_limit = pyo.Constraint(self.model.restricted_components, rule=self.flow_limit_rule)

        self.model.obj = pyo.Objective(rule=self.gibbs_energy_rule, sense=pyo.maximize)
        # Restricciones de conservación de átomos
        self.model.A = pyo.Set(initialize=range(self.atoms_matrix.shape[0])) 
        self.model.atom_balance = pyo.Constraint(self.model.A, rule=self.atom_balance_rule)

        solver = pyo.SolverFactory('ipopt')
        solver.options['tol'] = 1e-8
        solver.options['constr_viol_tol'] = 1e-16
        solver.options['acceptable_tol'] = 1e-6
        solver.options['max_iter'] = 2
        solver.options['linear_solver'] = 'mumps'
        results = solver.solve(self.model, tee=False)

        # Obtener valores de x
        x_val = [pyo.value(self.model.x[i]) for i in self.model.I]
        y_obj = pyo.value(self.model.obj)
        return x_val, y_obj , results.solver.status, results.solver.termination_condition , results.solver.termination_condition 

    def _run(self):
        #Armar el directorio de todos los flujos y chemicals 
        chems_mass  = defaultdict(float)
        chems_mol  = defaultdict(float)
        for feed in self.ins:
            for i , chem in enumerate(self.settings.chemicals):
                if i > 2 and i < len(self.chemicals)- self.remove_components:
                    chems_mass[str(chem)] += feed.imass[str(chem)]
                    chems_mol[str(chem)] += feed.imol[str(chem)]
        # Get chemicals available ...
        self.components = list(chems_mol.keys() )
        self.F_mol = list(chems_mol.values())
        self.mass_flow = sum(chems_mass.values()  )
        # Componentes que tienen azufre : Benzothiazole , vamos a eliminar todos compuestos que contienen azufre
        restricted_components =  ['sulfur', 'hydrogen sulfide' , 'benzothiazole']

        self.restricted_indices = [i for i, c in enumerate(self.components) if c in restricted_components]

        self.targetflows = self.F_mol.copy()
        for idx_local, idx_global in enumerate(self.restricted_indices):
            self.targetflows[idx_global] = 0  # Forzar a que sean 0 , no hay azufre en la entrada

        # Atoms matrix and atoms in 
        self.atoms_matrix = self.settings.chemicals.formula_array
        self.atoms_matrix = self.atoms_matrix[~np.all(self.atoms_matrix == 0, axis=1)]  # elimina atomos que no aparezcan
        self.atoms_matrix = self.atoms_matrix[:-1, 3:-self.remove_components]
        self.atoms_in =  self.atoms_matrix @ self.F_mol
        self.atoms_in[-1] = 0
        self.mw_list = self.settings.chemicals.MW
        self.mw_list = self.mw_list[3:]
        
        prev_x  =   np.full(len(self.components), sum(self.F_mol)/len(self.components) ) 
        tol=1e-2
        for i in range(1000):  # 2 iteraciones por cada corrida para volver a evaluar la OBJ (Razon = Pyomo la toma simbolica y despues fija los valores)
            new_x , obj_pyomo , status, termination_condition , termination_condition  = self.solve_pyomo(prev_x )
            # Criterio de convergencia
            diff = np.linalg.norm(np.array(new_x) - np.array(prev_x))
            if diff < tol:
                break
            prev_x = new_x
        self.final_flows = new_x
        flows_gas  =  [] 
        contador = 0
        for i ,c in enumerate(self.chemicals): 
            if i > 2 and i < len(self.chemicals)-self.remove_components:
                flows_gas.append(( c.common_name , self.final_flows[contador]))
                contador += 1
        product1 , product2, product3 = self.outs
        # Productos globales 
        outlet10 = bst.Stream(None, phase = 'g',  T=self.Treac, P=self.Preac, units='kmol/hr', **dict(flows_gas))
        
        # Primero separar hydrogeno y agua en dos corrientes
        flows_gas = [('molecular hydrogen', outlet10.imol['molecular hydrogen']*.90 )]
        outlet2 = bst.Stream(None, phase = 'g',  T=self.Treac, P=self.Preac, units='kmol/hr', **dict(flows_gas))
        product2.phases = ('g', 'l', 's')
        product2['g'].copy_like( outlet2['g'] )

        flows_gas = [('oxidane', outlet10.imol['oxidane']*.90 )]
        outlet3 = bst.Stream(None, phase = 'g',  T=self.Treac, P=self.Preac, units='kmol/hr', **dict(flows_gas))
        product3.phases = ('g', 'l', 's')
        product3['g'].copy_like( outlet3['g'] )

        # Actualiza la mezcla Off_gas
        outlet10.imol['molecular hydrogen'] = outlet10.imol['molecular hydrogen']* (1 -0.90)
        outlet10.imol['oxidane'] = outlet10.imol['oxidane']*(1-0.90)
        outlet10.imol[('sulfur', 'hydrogen sulfide' , 'benzothiazole')] = 0
        product1.phases = ('g', 'l', 's')
        product1['g'].copy_like( outlet10['g'] )

    def _design(self):
        """ 
        Servicios auxiliares
        """
        feed1 , feed2, feed3 = self.ins 
        product1 , product2, product3 = self.outs
        duty = product1.Hnet + product2.Hnet + product3.Hnet - feed1.Hnet - feed2.Hnet - feed3.Hnet  # KJ/hr
        self.duty = duty
        T_in = (feed1.T + feed2.T + feed3.T)/3  # Temperature of the inlet stream 
        T_out = self.Treac                      # Temperature of outlet stream
        iscooling = duty < 0.
        if iscooling: 
            if T_out > T_in:
                T_in = T_out
        else:
            if T_out < T_in:
                T_out = T_in
        self.add_heat_utility(duty, T_in = T_in , T_out = T_out) # New utility is also in self.heat_utilities
        
    def _cost(self):
        """ 
        Cost from Bernardo
        """
        # Calculated capital costs
        # Fixed investment costs      
        if self.processing_capacity == 6250: 
            Cost = 3870000.00
        elif self.processing_capacity == 4166: 
            Cost = 2690000.00 
        elif self.processing_capacity == 2500:
            Cost = 2235000.00 
        self.baseline_purchase_costs['NapthaPox'] = Cost
        # Assume design, pressure, and material factors are 1.
        self.F_D['NapthaPox'] = self.F_P['NapthaPox'] = self.F_M['NapthaPox'] = 1.

class Hydrotreatment(bst.Unit):
    """
    Create the Gibbs reactor        
    Parameters
    ins :    Inlet fluid.
    outs :  * [0] vapor product     * [1] solid product
    F : float   Mass flow of tyres [kg/hr]
    T : float   Operating temperature [K]
    P : float   Operating pressure [bar].
    """
    _N_ins , _N_outs = 2 , 3 # Number in inlets,  Number in outlets 
    _units = {'Area': 'm^2', 'Duty':'cal/sec'}

    def __init__(self, ID='', ins=None, outs=(), T= None, P=None, 
                    processing_capacity = None, settings = None ,epsilon = 1e-8):
        super().__init__(ID, ins, outs)
        self.Treac = T  # Temperatura en K
        self.Preac = P  # Presión en pascales
        self.epsilon = epsilon # Minimum amount of flowrate for each chemical
        self.processing_capacity = processing_capacity
        self.settings = settings
        self.remove_components = 3
   
    def gibbs_energy_rule(self, m):
        # calculo la x masica para tenerla como una constraint implicita en la Obj y calcular 
        # la energia de gibbs en funcion de eso...
        x_vec = sum([ (m.x[i]*self.mw_list[i])/self.mass_flow for i in m.I])
        flow_dict = {str(self.components[i]) : pyo.value(m.x[i])  for i in m.I} # self.mass_flow debe ser constante
        s_current = bst.Stream(None, phase='g', T=self.Treac, P=self.Preac, units='kmol/hr', **flow_dict)
        Gibbs_energy =  s_current.Hnet - (s_current.S) * s_current.T
        return x_vec*Gibbs_energy

    def atom_balance_rule(self, m, a):
        sum_atom = sum(self.atoms_matrix[a, i] * m.x[i]  for i in m.I)
        return sum_atom == self.atoms_in[a]

    def flow_limit_rule(self, m, i):
        return m.x[i] == self.targetflows[i]

    def solve_pyomo(self, initial_guess ):
        # Pyomo model
        self.model = pyo.ConcreteModel()
        self.model.I = pyo.RangeSet(0, len(self.components)-1)  # Índices para especies
        self.model.x = pyo.Var(self.model.I, domain=pyo.NonNegativeReals, bounds=(0, None), initialize=lambda m, i: initial_guess[i])
        
        self.model.restricted_components = pyo.Set(initialize=self.restricted_indices) 
        self.model.flow_limit = pyo.Constraint(self.model.restricted_components, rule=self.flow_limit_rule)

        self.model.obj = pyo.Objective(rule=self.gibbs_energy_rule, sense=pyo.maximize)
        # Restricciones de conservación de átomos
        self.model.A = pyo.Set(initialize=range(self.atoms_matrix.shape[0])) 
        self.model.atom_balance = pyo.Constraint(self.model.A, rule=self.atom_balance_rule)

        solver = pyo.SolverFactory('ipopt')  # z 
        solver.options['tol'] = 1e-8
        solver.options['constr_viol_tol'] = 1e-16
        solver.options['acceptable_tol'] = 1e-6
        solver.options['max_iter'] = 2
        solver.options['linear_solver'] = 'mumps'
        results = solver.solve(self.model, tee=False)

        # Obtener valores de x
        x_val = [pyo.value(self.model.x[i]) for i in self.model.I]
        y_obj = pyo.value(self.model.obj)
        return x_val, y_obj , results.solver.status, results.solver.termination_condition , results.solver.termination_condition 

    def _run(self):
        #Armar el directorio de todos los flujos y chemicals en cada corriente que entra al sistema
        chems_mass  = defaultdict(float)
        chems_mol  = defaultdict(float)
        for feed in self.ins:
            for i , chem in enumerate(self.settings.chemicals):
                if i > 2 and i < len(self.chemicals)-self.remove_components:
                    chems_mass[str(chem)] += feed.imass[str(chem)]
                    chems_mol[str(chem)] += feed.imol[str(chem)]
        # Get chemicals available ...
        self.components = list(chems_mol.keys() )
        self.F_mol = list(chems_mol.values())
        self.mass_flow = sum(chems_mass.values()  )
        # Componentes que tienen azufre : Benzothiazole , vamos a eliminar todos compuestos que contienen azufre
        restricted_components =  [ 'sulfur', 'molecular oxygen' , 'phenol',
                                    'benzonitrile', '2,3-dimethylphenol','benzoic acid', 
                                    '4-propan-2-ylphenol','Benzothiazole','2-methylquinoline',
                                    'pentadecanoic acid',]
        self.conversion = [100, 100, 2, 90, 2, 2, 2, 100, 90, 2]  # %
        
        self.restricted_indices = [i for i, c in enumerate(self.components) if c in restricted_components]

        self.targetflows = self.F_mol.copy()
        for idx_local, idx_global in enumerate(self.restricted_indices):
            self.targetflows[idx_global] = self.F_mol[idx_global] * (1 - self.conversion[idx_local]/100)


        # Atoms matrix and atoms in 
        self.atoms_matrix = self.settings.chemicals.formula_array
        self.atoms_matrix = self.atoms_matrix[~np.all(self.atoms_matrix == 0, axis=1)]  # elimina atomos que no aparezcan
        self.atoms_matrix = self.atoms_matrix[:, 3:-self.remove_components]
        self.atoms_in =  self.atoms_matrix @ self.F_mol
        self.mw_list = self.settings.chemicals.MW
        self.mw_list = self.mw_list[3:]
        
        prev_x  =   np.full(len(self.components), sum(self.F_mol)/len(self.components) ) 
        tol=1e-2
        for i in range(1000):  # 2 iteraciones por cada corrida para volver a evaluar la OBJ (Pyomo la toma simbolica y despues fija los valores)
            new_x , obj_pyomo , status, termination_condition , termination_condition  = self.solve_pyomo(prev_x )
            # Criterio de convergencia
            diff = np.linalg.norm(np.array(new_x) - np.array(prev_x))
            if diff < tol:
                break
            prev_x = new_x
        self.final_flows = new_x

        sulfur_components = {'sulfur','sulfane'}

        purge_components = {'molecular hydrogen', 'molecular nitrogen', 'molecular oxygen',
                            'carbon monoxide', 'carbon dioxide', 'methane', 'ammonia'}

        flows_gas_1 , flows_gas_2, flows_gas_3 =  [] , [] , []
        contador = 0
        for i ,c in enumerate(self.chemicals): 
            if i > 2 and i < len(self.chemicals)-self.remove_components:
                common_name = c.iupac_name[0]
                flow_total = self.final_flows[contador]
                if common_name in purge_components:
                    flows_gas_2.append((common_name, flow_total))  # todo a la purga
                elif common_name in sulfur_components:
                    flows_gas_3.append((common_name, flow_total))  # todo a esta corriente
                else:
                    flows_gas_1.append((common_name, flow_total))  # todo al Off_gas
                contador += 1
        product1, product2, product3 = self.outs
        # Flujos de hidrocarburos
        outlet1 = bst.Stream(None, phase = 'g',  T=self.Treac, P=self.Preac, units='kmol/hr', **dict(flows_gas_1))
        product1.phases = ('g', 'l', 's')
        product1['g'].copy_like(outlet1['g'])

        # Componentes purgados
        outlet2 = bst.Stream(None, phase = 'g',  T=self.Treac, P=self.Preac, units='kmol/hr', **dict(flows_gas_2))
        product2.phases = ('g', 'l', 's')
        product2['g'].copy_like(outlet2['g'])
        
        # Eliminacion de azufre
        outlet3 = bst.Stream(None, phase = 'g',  T=self.Treac, P=self.Preac, units='kmol/hr', **dict(flows_gas_3))
        product3.phases = ('g', 'l', 's')
        product3['g'].copy_like(outlet3['g'])
        
    def _design(self):
        """ 
        Servicios auxiliares
        """
        feed1 , feed2 = self.ins 
        product1 , product2, product3 = self.outs
        duty = product1.Hnet + product2.Hnet + product3.Hnet - feed1.Hnet - feed2.Hnet  # KJ/hr
        self.duty = duty
        T_in = (feed1.T + feed2.T)/2  # Temperature of the inlet stream 
        T_out = self.Treac # Temperature of outlet stream
        iscooling = duty < 0.
        if iscooling: 
            if T_out > T_in:
                T_in = T_out
        else:
            if T_out < T_in:
                T_out = T_in
        self.add_heat_utility(duty, T_in = T_in , T_out = T_out) # New utility is also in self.heat_utilities
        
    def _cost(self):
        """ 
        Cost are assumed to be for the Hydrogenation Equipment and Sulfur Recovery. 
        """
        # Calculated capital costs
        if self.processing_capacity == 6250: 
            Cost =  7970000.00 +  4020000.00 
        elif self.processing_capacity == 4166: 
            Cost = 5450000.00 +  3070000.00 
        elif self.processing_capacity == 2500:
            Cost = 3780000.00 + 1970000.00 
        self.baseline_purchase_costs['Hydrotreater'] =   Cost
        # Assume design, pressure, and material factors are 1.
        self.F_D['Hydrotreater'] = self.F_P['Hydrotreater'] = self.F_M['Hydrotreater'] = 1.

class ThermalOxidizer(bst.Unit):
    """
    Create a ThermalOxidizer that burns any remaining combustibles.
    
    Parameters
    ----------
    ins : 
        [0] Feed gas
        [1] Air
        [2] Natural gas
    outs : 
        Emissions.
    tau : float, optional
        Residence time [hr]. Defaults to 0.00014 (0.5 seconds).
    duty_per_kg : float, optional
        Duty per kg of feed. Defaults to 105858 kJ / kg.
    V_wf : float, optional
        Fraction of working volume. Defaults to 0.95.
    
    Notes
    -----
    Adiabatic operation is assumed. Simulation and cost is based on [1]_.

    """
    _N_ins = 3
    _N_outs = 1
    max_volume = 20. # m3
    _F_BM_default = {'Vessels': 2.06} # Assume same as dryer
    
    @property
    def natural_gas(self):
        """[Stream] Natural gas to satisfy steam and electricity requirements."""
        return self.ins[2]
    
    def _init(self, tau=0.00014, duty_per_kg=61., V_wf=0.95):
        self.define_utility('Natural gas', self.natural_gas)
        self.tau = tau
        self.duty_per_kg = duty_per_kg
        self.V_wf = V_wf

    def _run(self):
        feed, air, ng = self.ins
        ng.imol['CH4'] = self.duty_per_kg * feed.F_mass / self.chemicals.CH4.LHV
        ng.phase = 'g'
        emissions, = self.outs
        ng_burned = ng.copy()
        combustion_rxns = self.chemicals.get_combustion_reactions()
        # Enough oxygen must be present in air to burn natural gas
        combustion_rxns.force_reaction(ng_burned)
        O2 = max(-ng_burned.imol['O2'], 0.)
        air.imol['N2', 'O2'] = [0.79/0.21 * O2, O2]
        # Enough oxygen must be present in air to burn feed as well
        emissions.mix_from(self.ins)
        dummy_emissions = emissions.copy()
        combustion_rxns.force_reaction(dummy_emissions)
        O2 = max(-dummy_emissions.imol['O2'], 0.) # Missing oxygen
        air.imol['N2', 'O2'] += [0.79/0.21 * O2, O2]
        emissions.mix_from(self.ins)
        # Account for temperature raise
        combustion_rxns.adiabatic_reaction(emissions)
        
    def _design(self):
        design_results = self.design_results
        volume = self.tau * self.outs[0].F_vol / self.V_wf
        V_max = self.max_volume
        design_results['Number of vessels'] = N = np.ceil(volume / V_max)
        design_results['Vessel volume'] = volume / N
        design_results['Total volume'] = volume
        
    def _cost(self):
        design_results = self.design_results
        N = design_results['Number of vessels']
        vessel_volume = design_results['Vessel volume']
        C = self.baseline_purchase_costs
        C['Vessels'] = N * 918300. * (vessel_volume / 13.18)**0.6

class Mechanical_Activation(bst.Unit):
    _N_ins = 1     # tyres, syngas_recycle, air
    _N_outs = 1    # raw_gas, metals, carbon, emissions
    _units = {'Duty': 'kJ/hr'}

    def _init(self, processing_capacity=None):
        self.processing_capacity = processing_capacity
        
    def _run(self):
        carbon = self.ins[0]
        carbon_activated = self.outs[0]

        carbon_activated.empty()
        carbon_activated.phase = 's'
        carbon_activated.copy_like(carbon)

    def _design(self):
        """ 
        Servicios auxiliares
        """
        self.duty = 1


    def _cost(self):

        if self.processing_capacity == 6250:
            Refinamiento = 7280000
            Instalacion  =   180000
        elif self.processing_capacity == 4166:
            Refinamiento = 7280000
            Instalacion  =   180000
        elif self.processing_capacity == 2500:
            Refinamiento = 4270000
            Instalacion  =   98000
        else:
            raise ValueError(f'processing_capacity={self.processing_capacity} no soportado en _cost (solo 6250, 4166, 2500).')

        self.baseline_purchase_costs['Unit'] = Refinamiento + Instalacion
        self.F_D['Unit'] = self.F_P['Unit'] = self.F_M['Unit'] = 1.0
        
class PyrolysisReactor(bst.Unit):
    _N_ins = 3     # tyres, syngas_recycle, air
    _N_outs = 4    # raw_gas, metals, carbon, emissions
    _units = {'Duty': 'kJ/hr'}

    def _init(self,
              # --- parámetros de R1 ---
              Ttyres=None, P=None,
              P_moisture=None, P_ash=None,
              U_carbon=None, U_h=None, U_n=None, U_s=None, U_o=None,
              P_volatilemateria=None,
              # --- parámetros de R2 ---
              Treac=None, Preac=None,
              processing_capacity=None,
              carbon_conversion=None,
              settings=None):

        # R1
        self.Ttyres = Ttyres
        self.P = P
        self.P_moisture = P_moisture
        self.P_ash = P_ash
        self.U_carbon = U_carbon
        self.U_h = U_h
        self.U_n = U_n
        self.U_s = U_s
        self.U_o = U_o
        self.P_volatilemateria = P_volatilemateria

        # R2
        self.Treac = Treac
        self.Preac = Preac
        self.processing_capacity = processing_capacity
        self.carbon_conversion = carbon_conversion
        self.settings = settings

        # duties
        self.duty1 = 0.0
        self.duty2 = 0.0
        self.duty  = 0.0

        # heating bookkeeping
        self.Q_deficit = 0.0
        self.Q_from_syngas = 0.0
        self.Q_from_natural_gas = 0.0

        # Data
        self._get_MW()
        self._load_kinetik_data()
        self._get_kinetic_data()
        self._load_CP_data()

    # -------------------- R1 helpers --------------------
    def _load_CP_data(self):
        output_path = os.path.join('pyrolysis', 'Cp_Data.csv')
        self.df_Cp = pd.read_csv(output_path)

    def Cp_ij(self, T, j, param):
        return param[j,0] + param[j,1]*T + param[j,2]*T**2 + param[j,3]*T**3

    def Cp_dl(self, T, w, param):
        return sum(w[j] * self.Cp_ij(T, j, param) for j in range(len(w)))

    def _Heat_Capacity_Kirov_Correlations(self, Flow, T_ref, Tint,
                                          moisture, fixed_carbon,
                                          primary_volatile_matter, secondary_volatile_matter, ash):
        param = np.array([[1.0, 0 , 0 , 0 ],
                          [0.165, 6.8e-4, -4.2e-7, 0 ],
                          [0.395, 8.1e-4, 0 , 0 ],
                          [0.71, 6.1e-4, 0 , 0 ],
                          [0.18, 1.4e-4, 0 , 0] ])
        if primary_volatile_matter > 10:
            weights = np.array([moisture, fixed_carbon, primary_volatile_matter, secondary_volatile_matter, ash])
            weights = weights/(1 - moisture/100)
            val = 10.3
            weights[3] = weights[2] - val
            weights[2] = val
        else:
            weights = np.array([moisture, fixed_carbon, primary_volatile_matter, secondary_volatile_matter, ash])

        integral, _ = quad(self.Cp_dl, T_ref, Tint, args=(weights, param))
        Enthalpy_flow = integral * Flow * 1000 / 3600
        return Enthalpy_flow  # cal/sec

    def Cp_model(self, T, C):
        term1 = C[0]
        term2 = C[1] * (((C[2]/T) / np.sinh(C[2]/T))**2)
        term3 = C[3] * (((C[4]/T) / np.cosh(C[4]/T))**2)
        return term1 + term2 + term3

    def integrar_Cp_vapour_fase(self, Tref, Tint, C):
        resultado, _ = quad(self.Cp_model, Tref, Tint, args=(C))
        return resultado

    def _H_out_vapour_fase_from_flows(self, flows_gas, T_out_K):
        Tref, Tint = 25+273.15, T_out_K
        species = [sp[0] for sp in flows_gas]
        flows = np.array([sp[1] for sp in flows_gas])  # kg/hr
        integral = 0.0
        for i, specie in enumerate(species):
            C = self.df_Cp.loc[self.df_Cp['common _names'] == specie, ['1','2','3','4','5']].values[0]
            mw = self.df_Cp.loc[self.df_Cp['common _names'] == specie, 'mw'].values[0]
            integral += self.integrar_Cp_vapour_fase(Tref, Tint, C)/mw * flows[i] * 1000/3600
        return integral

    def _H_in_tyres(self, Tyre_flow_kgph):
        Ttyres_C = 20
        T_ref = 25
        return self._Heat_Capacity_Kirov_Correlations(
            Tyre_flow_kgph, T_ref, Ttyres_C,
            self.P_moisture, self.U_carbon,
            self.P_volatilemateria, 0, self.P_ash
        )

    # -------------------- R2 helpers --------------------
    def _get_MW(self):
        self.MW_list = self.settings.chemicals.MW

    def _load_kinetik_data(self):
        self.df_kinetics = pd.read_csv(os.path.join('pyrolysis', 'kinetic_main_original.csv'))

    def _get_kinetic_data(self):
        k_constants, n_constants, energy_constants, type_of_reaction = [], [], [], []
        dfk = self.df_kinetics
        number_reactions = 0
        for i in range(dfk.shape[0]):
            if dfk.loc[i, 'Available'] == 'Yes':
                k_constants.append(dfk.loc[i,'k_constant'])
                n_constants.append(dfk.loc[i,'n'])
                energy_constants.append(dfk.loc[i,'E[kj/mol]'])
                type_of_reaction.append(dfk.loc[i,'Dependant_index'])
                number_reactions += 1

        self.number_reactions = number_reactions
        self.k_constants = np.array(k_constants)
        self.E_ = np.array(energy_constants) * 1000
        self.n_ = np.array(n_constants)
        self.type_of_reaction = type_of_reaction

        self.H2_index, self.S_index, self.O2_index = 3, 5, 6

        stoich = pd.read_csv(os.path.join('pyrolysis','stoichometry_main.csv'), header=None).to_numpy()
        stoich[:,0:8] *= -1
        self.stoichiometry = stoich

    def _reaction_rate(self, x, type_of_reaction, k, n, E, T):
        gas_flow_mass = sum(x[i]*self.MW_list[i] for i in range(3, len(x)))
        if gas_flow_mass <= 0:
            return 0.0

        if type_of_reaction == 1:
            X_H2 = (x[self.H2_index]*2.01568) / gas_flow_mass
            return k*(T**n)*np.exp(-E/(R*T))*X_H2
        elif type_of_reaction == 2:
            X_O2 = (x[self.O2_index]*31.999) / gas_flow_mass
            return k*(T**n)*np.exp(-E/(R*T))*X_O2
        elif type_of_reaction == 3:
            X_S = (x[self.S_index]*32.065) / gas_flow_mass
            return k*(T**n)*np.exp(-E/(R*T))*X_S
        else:
            return 0.0

    def ode(self, t, x):
        rates = []
        for i in range(self.number_reactions):
            rates.append(self._reaction_rate(
                x, self.type_of_reaction[i],
                self.k_constants[i], self.n_[i], self.E_[i],
                T=self.Treac
            ))
        rates = np.array(rates).reshape([-1, 1])
        dy = rates * self.stoichiometry
        return dy.sum(axis=0)

    # -------------------- combustión de syngas para utilidad --------------------
    def _combust_stream(self, fuel_stream, air_stream):
        """
        Quema COMPLETO fuel_stream con aire estequiométrico.
        Retorna: (Q_release [kJ/hr], flue_stream)
        """
        if fuel_stream.F_mass <= 1e-12:
            air_stream.empty()
            flue = bst.Stream(None, phase='g'); flue.empty()
            return 0.0, flue

        flue = bst.Stream(None, phase='g')
        flue.copy_flow(fuel_stream)

        combustion = self.settings.chemicals.get_combustion_reactions()  # <- FIX

        air_stream.empty()
        combustion.force_reaction(flue)

        O2_req = -flue.imass['O2']
        if O2_req < 0: O2_req = 0.0

        air_stream.imass['O2'] = O2_req
        air_stream.imass['N2'] = air_stream.imass['O2'] * 79 / 21

        flue.mol += air_stream.mol

        H_before = fuel_stream.Hnet
        combustion.force_reaction(flue)
        H_after = flue.Hnet

        H_rxn = H_after - H_before
        Q_release = max(0.0, -H_rxn)
        return Q_release, flue

    def _natural_gas_for_heat(self, Q_needed):
        ng = bst.Stream(None, phase='g')
        if Q_needed <= 0:
            ng.empty()
            return ng

        # intenta encontrar CH4/methane
        fuel_key = None
        for key in ('methane', 'CH4'):
            try:
                ng.imass[key] = 1.0
                ng.empty()
                fuel_key = key
                break
            except Exception:
                ng.empty()
                fuel_key = None

        if fuel_key is None:
            raise ValueError("No encuentro 'CH4' o 'methane' en tu thermo. Agrega CH4/methane a settings.chemicals.")

        LHV_kJ_per_kg = 50000.0
        m_kgph = Q_needed / LHV_kJ_per_kg
        ng.imass[fuel_key] = m_kgph
        return ng

    # -------------------- RUN --------------------
    def _run(self):
        tyres, syngas_recycle, air = self.ins
        raw_gas, metals, carbon, emissions = self.outs  # <- raw_gas

        Tyre_flow = tyres.F_mass

        # ====== (1) DESCOMPOSER ======
        H2OF, ASHF = self.P_moisture, self.P_ash
        CF, H2F = self.U_carbon, self.U_h
        N2F, SF = self.U_n, self.U_s
        O2F = self.U_o

        self.H2OOUT = Tyre_flow*(H2OF/100)
        self.ASHOUT = Tyre_flow*(ASHF/100*(1- H2OF/100))
        self.COUT   = Tyre_flow*(CF/100*(1- H2OF/100))
        self.H2OUT  = Tyre_flow*(H2F/100*(1- H2OF/100))
        self.N2OUT  = Tyre_flow*(N2F/100*(1- H2OF/100))
        self.SOUT   = Tyre_flow*(SF/100*(1- H2OF/100))
        self.O2OUT  = Tyre_flow*(O2F/100*(1- H2OF/100))

        flows_gas_R1 = [('water', self.H2OOUT),
                        ('hydrogen', self.H2OUT),
                        ('nitrogen', self.N2OUT),
                        ('sulfur', self.SOUT),
                        ('molecular oxygen', self.O2OUT)]
        self.out_flow_gas_R1 = flows_gas_R1

        # Duty 1
        H_in = self._H_in_tyres(Tyre_flow)  # cal/s
        H_out_vap = self._H_out_vapour_fase_from_flows(flows_gas_R1, self.Treac)
        self.duty1 = max(0.0, (H_out_vap - H_in) * 15.072479976493)  # kJ/hr

        # ====== (2) MAIN REACTOR ======
        tyre_init = 0.0
        ash_init  = 0.0
        carbon_init   = (self.COUT/12.0) / 3600.0
        hydrogen_init = (self.H2OUT/2.01568) / 3600.0
        nitrogen_init = (self.N2OUT/28.0134) / 3600.0
        sulfur_init   = (self.SOUT/32.065) / 3600.0
        oxygen_init   = (self.O2OUT/31.999) / 3600.0
        water_init    = (self.H2OOUT/18.01528) / 3600.0

        self.initial_conditions = [tyre_init, ash_init, carbon_init, hydrogen_init,
                                   nitrogen_init, sulfur_init, oxygen_init, water_init]
        self.initial_conditions.extend([0.0] * (len(self.settings.chemicals) - 8))

        def rendimiento_event(t, y):
            flujo_producto = y[2] * self.MW_list[2] * 3600.0
            denom = (self.initial_conditions[2] * self.MW_list[2] * 3600.0)
            consumido = denom - flujo_producto
            rendimiento = consumido/denom if denom > 0 else 0.0
            return rendimiento - self.carbon_conversion/100.0

        rendimiento_event.terminal = True

        self.diameter = 0.5
        self.max_length = 8.0
        V = (np.pi * self.max_length * (self.diameter/2)**2)

        sol = solve_ivp(self.ode, [0, V], self.initial_conditions,
                        events=rendimiento_event, method='RK45', max_step=0.001)

        results = sol.y.T
        self.final_flows = (results * self.MW_list) * 3600.0  # kg/hr
        carbon_consumido = (self.initial_conditions[2]* self.MW_list[2] * 3600) - self.final_flows[-1,2]
        self.real_Carbonconversion =  carbon_consumido/( (self.initial_conditions[2]* self.MW_list[2] * 3600) )

        # carbon_yield
        self.carbon_yield = self.final_flows[-1, 2] / self.processing_capacity

        # ====== (3) Construye salidas BioSTEAM ======
        flows_gas = []
        for i, c in enumerate(self.settings.chemicals):
            if i > 2 and i < len(self.settings.chemicals):
                flows_gas.append((c.common_name, self.final_flows[-1, i]))

        flows_solid1 = [('Ash', self.ASHOUT)]
        flows_solid2 = [('carbon', self.final_flows[-1, 2])]

        vapor = bst.Stream(None, phase='g', T=self.Treac, P=self.Preac, units='kg/hr', **dict(flows_gas))
        solid1 = bst.Stream(None, phase='s', T=self.Treac, P=self.Preac, units='kg/hr', **dict(flows_solid1))
        solid2 = bst.Stream(None, phase='s', T=self.Treac, P=self.Preac, units='kg/hr', **dict(flows_solid2))

        raw_gas.empty()
        raw_gas.phase = 'g'
        raw_gas.copy_like(vapor)

        metals.empty()
        metals.phase = 's'
        metals.copy_like(solid1)

        carbon.empty()
        carbon.phase = 's'
        carbon.copy_like(solid2)

        # ====== (4) Duty 2 ======
        vapor_in = bst.Stream(None, phase='g', T=self.Treac, P=self.Preac, units='kg/hr', **dict(flows_gas_R1))
        H_out_vapour_in = vapor_in.Hnet
        H_out_vapor_out = vapor.Hnet
        duty2_raw = H_out_vapor_out - H_out_vapour_in
        self.duty2 = duty2_raw * 1.3

        self.duty = self.duty1 + self.duty2

        # ====== (5) Heat from recycled syngas + NG deficit ======
        emissions.empty()
        emissions.phase = 'g'

        Q_syngas, flue_syngas = self._combust_stream(syngas_recycle, air)
        self.Q_from_syngas = Q_syngas
        emissions.copy_like(flue_syngas)

        self.Q_deficit = max(0.0, self.duty - self.Q_from_syngas)

        if self.Q_deficit > 0:
            ng_fuel = self._natural_gas_for_heat(self.Q_deficit)
            air_ng = bst.Stream(None, phase='g')
            Q_ng, flue_ng = self._combust_stream(ng_fuel, air_ng)
            self.Q_from_natural_gas = Q_ng
            emissions.mol += flue_ng.mol
        else:
            self.Q_from_natural_gas = 0.0

    # ---- costos (sin cambios) ----
    def compute_furnace_purchase_cost(self, Q, CE):
        return exp(-0.15241 + 0.785*ln(Q)) * CE / 567

    def _cost(self):
        Q = self.duty1 * 0.947817
        if Q <= 0:
            self.baseline_purchase_costs['Descomposer'] = 0.0
        else:
            self.baseline_purchase_costs['Descomposer'] = self.compute_furnace_purchase_cost(Q, bst.CE)
        self.F_D['Descomposer'] = self.F_P['Descomposer'] = self.F_M['Descomposer'] = 1.0

        if self.processing_capacity == 6250:
            Pyrolysis_unit = 24544000.00
            Nitrogen_unit  =   495000.00
        elif self.processing_capacity == 4166:
            Pyrolysis_unit = 17230000.00
            Nitrogen_unit  =   495000.00
        elif self.processing_capacity == 2500:
            Pyrolysis_unit = 11870000.00
            Nitrogen_unit  =   495000.00
        else:
            raise ValueError(f'processing_capacity={self.processing_capacity} no soportado en _cost (solo 6250, 4166, 2500).')

        self.baseline_purchase_costs['MainReactor'] = Pyrolysis_unit + Nitrogen_unit
        self.F_D['MainReactor'] = self.F_P['MainReactor'] = self.F_M['MainReactor'] = 1.0

def create_system(processing_capacity,x,y, settings, type = None):
    
    processing_capacity = processing_capacity

    # Put all results on a dry basis
    dry_processing_capacity = processing_capacity   # kg/hr dry basis
    baseline_moisture = 4.0                        # wt% wet basis

    wet_feed_capacity = dry_processing_capacity / (1 - baseline_moisture/100)
    # print(processing_capacity)
    with bst.System('example') as context_sys:
        if type == None:
            target_conversion = x
            target_price_carbon = y 
            target_price_diesel = 1137.50/1000
            target_temperature_reactor = 550 +273.15
        elif type == 'Titer1':  # Change only carbon conversion and Carbon price
            target_conversion = x
            target_price_carbon = y 
            target_price_diesel = 1137.50/1000
            target_temperature_reactor = 550 +273.15
        elif type == 'Titer2':
            target_conversion = x
            target_price_carbon = 3.5 
            target_price_diesel = y  
            target_temperature_reactor = 550 +273.15
        elif type == 'Titer3':
            target_conversion = x
            target_price_carbon = 3.5 
            target_price_diesel = 1137.50/1000
            target_temperature_reactor = y
        # Corrientes reciclos
        syngas_recycle = bst.Stream('syngas_recycle')
        air = bst.Stream('air')
        Napthas_recycle = bst.Stream('Napthas_recycle')
        Hydrogen_generation = bst.Stream('Hydrogen_generation')

        # Feed stream
        Tyre_stream = bst.Stream('Tyre_Stream', phase='s',Tyre = wet_feed_capacity,  T= 20 + 273.15, units='kg/hr', price=  0.14) 
        
        # Reactor principal 
        Metal = bst.Stream('Metals', price = 237/1000)
        # El precio del carbono puede variar dependiendo de la region
        Carbon =  bst.Stream('CarbonActivated', price = target_price_carbon) 


        R2 = PyrolysisReactor('MainReactor',
                        ins=(Tyre_stream, syngas_recycle, air),
                        outs=('pyrolitic_gas', Metal, 'carbon_post', 'emissiones'),
                        # R1 params
                        Ttyres=20+273.15, P=101325,
                        P_moisture= baseline_moisture, P_ash=2.5, U_carbon=83.3,
                        U_h=7.5, U_n=0.6, U_s=1.6, U_o=4.5, P_volatilemateria=64.6,
                        # R2 params
                        Treac=target_temperature_reactor, Preac=101325,
                        processing_capacity=dry_processing_capacity, carbon_conversion=target_conversion,
                        settings=settings)
        MA = Mechanical_Activation('CarbonTreatment', ins = (R2-2), outs = (Carbon), processing_capacity=processing_capacity)

        # Cooler1 
        cooler1 = HXutility('Cooler1',  outs= 'cooled_stream1', 
                            T= 30+273.150, ins=(R2-0))
        # Flashes
        F1 = Flash('Flash1', outs=('syngas', 'liquid1'),
                    P=101325, T=30 +273.15 , 
                    ins= (cooler1-0) ) 
        

        S1 = bst.Splitter('S1', ins=(F1-0,),
                        outs=(syngas_recycle, bst.Stream('syngas_excess')),
                        split=0.5)


        @S1.add_specification(run=True)
        def adjust_syngas_recycle_split():
            duty = getattr(R2, 'duty', 0.0)
            if duty <= 0:
                S1.split[:] = 0.0
                return

            syngas_vapor, = S1.ins
            tmp_air = bst.Stream(None, phase='g')

            Q_full, _ = R2._combust_stream(syngas_vapor, tmp_air)
            f = 0.0 if Q_full <= 1e-9 else duty / Q_full
            S1.split[:] = min(1.0, max(0.0, f))

        # PowerGeneration -  BoilerTurbogenerators  # S1-1
        BT = bst.BoilerTurbogenerator('BT',
                    (S1-1, '', 'boiler_makeup_water', 'natural_gas', 'waste1', 'waste2'),
                    boiler_efficiency=0.95, turbogenerator_efficiency=0.95)

        
        F2 = Flash('Flash2', outs=('naptha', 'diesel_hfo'), P=101325, T=160+273.15, ins =(F1-1) )
        # Agregar una restriccion para que los compuestos que contengan azufre se vayan todos a los fondos
        @F2.add_specification
        def eliminate_sulfur():
            F2.run()
            IDs = ('sulfur', 'hydrogen sulfide' , 'benzothiazole') 
            vapor, liquid = F2.outs
            # Limpiar lo que esta en el vapor 
            flows = vapor.imass[IDs]       
            # Todo el azufre se va a liquidos
            liquid.imass[IDs] += flows
            vapor.imass[IDs] = 0

        # Add cooler 
        cooler2 = HXutility('Cooler2',  outs= 'cooled_stream2', T= 30+273.150, ins=(F2-0)) # Naptha
        # Add cooler 
        cooler3 = HXutility('Cooler3',  outs= 'cooled_stream3', T= 30+273.150, ins=(F2-1)) # Diesel and HFO
        # Naptha partial oxidation 
        # Faltas precios de oxigeno y revisar el del STEAM
        # https://businessanalytiq.com/procurementanalytics/index/oxygen-price-index/
        Oxygen_stream = bst.Stream('Oxygen', phase = 'g', oxygen = 497, T=25+273.15, P=101325, units= 'kg/hr', price = 0.09) 
        Steam_Stream = bst.Stream('Steam', phase='g', water = 583 , T = 215+273.15,  P=101325, units= 'kg/hr', price= 3.34/1000)
        
        Mix1 = bst.Mixer('Mixer1', ins=(cooler2-0, Napthas_recycle), outs='mixed_stream') 
        
        # Naptha_PartialOxidation
        R3 = Naptha_PartialOxidation('NapthaPox', outs=  ('Off_gas', Hydrogen_generation,'Steam') , 
                                        ins= ( Mix1-0, Oxygen_stream, Steam_Stream ), 
                                        T = 500+273.15, P = 400000 ,
                                         processing_capacity = processing_capacity,
                                           settings = settings )

        # Hydrotreater
        R4 = Hydrotreatment('Hydrotreater', outs = ('hydrotreater_out','purge','sulfur_recovery'), ins = ( R3-1, cooler3-0 ) , 
                            T = 310+273.15, P = 90*100000 ,
                            processing_capacity = processing_capacity,
                              settings = settings )

        # # Fractionador/  Naptha de diesel y LFO
        D1 = bst.BinaryDistillation('D1', ins=(R4-0) ,outs=(Napthas_recycle, 'dieselLFO'), LHK = ('hexane','toluene'), 
                                    Lr = 0.99 , Hr = 0.99, k=1.2, is_divided=False, P= 101325.0, partial_condenser=False)
        D1.check_LHK = False   

        # Se separa el diesel de LFO
        Diesel = bst.Stream('Diesel', price= target_price_diesel)
        LFO = bst.Stream('LFO', price= round(3000*0.14/1000,2) )

        D2 = bst.BinaryDistillation('D2', ins=(D1-1) ,outs=(Diesel, LFO), LHK=('pentadecane', 'Nonadecane'),
                                    Lr=0.99, Hr=0.99, k=1.2, is_divided=False, P= 101325.0, partial_condenser=False)
        D2.check_LHK = False   

        # Mixer 2
        Mix2 = bst.Mixer('Mixer2', ins=(R3-0,R4-1,R2-3), outs='mixed_stream') 

        # Thermaloxidazer ( QUEMA TODO )
        OxRedox = ThermalOxidizer('OxRedox', ins = (Mix2-0)  )

    context_sys.set_tolerance(mol=1e-2, rmol=1e-2)
    context_sys.simulate()
    osbl_units = bst.get_OSBL(context_sys.cost_units)
    tea = PyrolyisisTEA(system=context_sys, 
        IRR=0.10, duration=(2025, 2055), depreciation='MACRS20',
        income_tax=0.21, operating_days=330, lang_factor=None,
        construction_schedule=(0.4, 0.6),
        WC_over_FCI=0.05,
        labor_cost=2.5e6,
        fringe_benefits=0.4,  # mapea a labor_burden
        property_tax=0.001,   # (no lo usan las tablas, pero puedes guardarlo)
        property_insurance=0.005,
        supplies=0.20,  #OKEY
        maintenance=0.01,  # OKEY
        administration=0.005,  # OKEY
        OSBL_units=osbl_units,
        boiler_turbogenerator=BT,  # si quieres depre distinta para BT podrías extender
    )
    NPV =  tea.NPV/1e6 
    IRR_val =  round(tea.solve_IRR()*100,2)
    
    #LCA 
    GWP = 'GWP 100yr'
    bst.F.Tyre_Stream.set_CF(GWP, 0)
    bst.F.CarbonActivated.set_CF(GWP,0)
    bst.F.Metals.set_CF(GWP,0)
    bst.F.Diesel.set_CF(GWP,0)
    bst.F.LFO.set_CF(GWP,0)
    bst.F.Oxygen.set_CF(GWP,0.18)
    bst.F.natural_gas.set_CF(GWP,0.33) # Antes 1.5 Correcto 0.33

    # context_sys.process_impact_items[GWP].clear()
    context_sys.define_process_impact(
        key = GWP ,
        name= 'direct_emissions1', 
        basis = 'kg',
        inventory =  lambda: bst.F.emissions.imass['CO2'] * context_sys.operating_hours,
        CF =  1.0 
    )
    context_sys.define_process_impact(
        key = GWP ,
        name= 'direct_emissions2', 
        basis = 'kg',
        inventory =  lambda: bst.F.OxRedox.outs[0].imass['CO2'] * context_sys.operating_hours,
        CF =  1.0 
    )
    

    # --- Productos a reportar ---
    products = {
        "activatedcarbon": bst.F.CarbonActivated,
        "metals": bst.F.Metals,
        "diesel": bst.F.Diesel,
        "LFO": bst.F.LFO,
    }

    # --- Energy allocation ---
    gwp_energy = {}
    GWP_per_GGE = context_sys.get_property_allocated_impact(key=GWP, name="energy", basis="GGE")
    for k, s in products.items():
        try:
            if s.F_mass <= 0:
                gwp_energy[k] = float("nan")
            else:
                gge_per_hr = s.get_property("LHV", "GGE/hr")  # 0 para corrientes sin LHV
                gwp_energy[k] = float(GWP_per_GGE * gge_per_hr / s.F_mass)
        except Exception:
            # Por si algún stream no soporta la propiedad, o LHV no aplica
            gwp_energy[k] = float("nan")

    # --- Revenue allocation = kgCO2e/USD * precio(USD/kg) => kgCO2e/kg 
    gwp_revenue = {}
    GWP_per_USD = context_sys.get_property_allocated_impact(key=GWP, name="revenue", basis="USD")
    for k, s in products.items():
        gwp_revenue[k] = float(GWP_per_USD * (s.price or 0.0))

    # --- Mass allocation
    gwp_mass = {}
    total_impact = (
        context_sys.get_total_feeds_impact(GWP)
        #+ context_sys.get_net_electricity_impact(GWP)
    )
    total_prod_mass = sum(context_sys.get_mass_flow(s) for s in products.values())
    for k, s in products.items():
        total_prod_mass = context_sys.get_mass_flow( s ) #for s in products.values())
        m = context_sys.get_mass_flow(s)
        gwp_mass[k] = float(total_impact / total_prod_mass) if total_prod_mass > 0 else float("nan")

    GWP_results = {"energy_kgCO2e_per_kg": gwp_energy,
                    "revenue_kgCO2e_per_kg": gwp_revenue,
                    "mass_kgCO2e_per_kg": gwp_mass}
    
    # FEDI calculations
    output_folder = 'pyrolysis'
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder,'Data_components_clean.csv')
    df_data = pd.read_csv(output_path, encoding='latin1')
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

    Mass =  bst.F.MainReactor.outs[0].F_mass/3600  #  [kg/hr] to   [Kg/sec] de vapores
    Pequipment = bst.F.MainReactor.Preac/1000  #  [Pa] to [kPa]
    Vol =   (4*4*4)*22  # [m3]
    VapPress  =    bst.F.MainReactor.Preac/1000  #  752180.689431543   # [kPa] Average Value taken from Aspen
    Factor1 = 0.01*(Mass*(Enthalpy_comb))/3.148
    Factor2 = (6/3.148)*Pequipment*Vol
    Tope = bst.F.MainReactor.Treac
    Factor3 = (1e-3)*(1/Tope)*((Pequipment-VapPress)**2)*Vol   # La temperatura es de la operación
    if Tope > Flash_point_mixture and Tope < 0.75*Ignition_mixture: 
        pn1 = (1.45 +  1.75)/2
    elif Tope > 0.75*Ignition_mixture:
        pn1 = 1.95
    else:
        pn1 = 1.1
    # Penalty 2
    if VapPress > 101.325 and Pequipment > VapPress :
        pn2 = 1 + (Pequipment-VapPress)*0.6/Pequipment
        FactorGen = Factor2 + Factor3
    else:
        pn2 = 1 + (Pequipment-VapPress)*0.4/Pequipment
        FactorGen = Factor2
    if VapPress < 101.325 and 101.325 < Pequipment:
        pn2 = 1 + (Pequipment-VapPress)*0.2/Pequipment
        FactorGen = Factor3
    else:
        pn2 = 1.1
        FactorGen = Factor3          
    Factor4  =   (Mass*abs(bst.F.MainReactor.duty))/3.148

    pn4 = 1 + 0.25*(3 + 0)
    pn3,pn5,pn6,pn7 = 1,1,1, 1.45

    Damage_Potential = (Factor1*pn1 + FactorGen*pn2 + Factor4*pn7)*(pn3*pn4*pn5*pn6)
    # print('Damage_Potential', Damage_Potential)
    Fedi_val = 4.76*(Damage_Potential**(1/3))
    return context_sys, tea, NPV , IRR_val, Fedi_val, GWP_results




import biosteam as bst
from biorefineries.tea.cellulosic_ethanol_tea import capex_table as _capex_table_helper, foc_table as _foc_table_helper

class PyrolyisisTEA(bst.TEA):
    def __init__(self, system, IRR, duration, depreciation, income_tax,
                 operating_days, lang_factor, construction_schedule,
                 WC_over_FCI, labor_cost, fringe_benefits,
                 property_tax, property_insurance, supplies,
                 maintenance, administration,
                 OSBL_units=None, boiler_turbogenerator=None,  # opcionales
                 warehouse=0.04, site_development=0.09, additional_piping=0.045,
                 proratable_costs=0.10, field_expenses=0.10, construction=0.20,
                 contingency=0.40, other_indirect_costs=0.10):
        super().__init__(system, IRR, duration, depreciation, income_tax,
                         operating_days, lang_factor, construction_schedule,
                         # startup: valores simples por ahora
                         startup_months=3, startup_FOCfrac=1.0,
                         startup_VOCfrac=0.75, startup_salesfrac=0.5,
                         WC_over_FCI=WC_over_FCI,
                         finance_interest=None, finance_years=None, finance_fraction=None)

        # ---- parámetros que usan las tablas ----
        self.warehouse = warehouse
        self.site_development = site_development
        self.additional_piping = additional_piping
        self.proratable_costs = proratable_costs
        self.field_expenses = field_expenses
        self.construction = construction
        self.contingency = contingency
        self.other_indirect_costs = other_indirect_costs

        self.WC_over_FCI = WC_over_FCI
        self.labor_cost = labor_cost
        self.labor_burden = fringe_benefits  # mapeo directo
        self.maintenance = maintenance
        self.property_insurance = property_insurance

        # OSBL units (calderas, enfriamiento, etc.)
        if OSBL_units is None:
            OSBL_units = bst.get_OSBL(system.cost_units)
        self.OSBL_units = OSBL_units
        self.boiler_turbogenerator = boiler_turbogenerator

        # cachés
        self._ISBL_DPI_cached = 0.0
        self._DPI_cached = 0.0
        self._FCI_cached = 0.0

    # ------------ costos ISBL / OSBL ------------
    @property
    def ISBL_installed_equipment_cost(self):
        # DPI dentro de batería de planta
        return self._ISBL_DPI(self.installed_equipment_cost)

    @property
    def OSBL_installed_equipment_cost(self):
        sys = self.system
        if isinstance(sys, bst.AgileSystem):
            ucc = sys.unit_capital_costs
            return sum(ucc[u].installed_cost for u in self.OSBL_units)
        else:
            return sum(u.installed_cost for u in self.OSBL_units)

    # ------------ DPI / TDC / FCI / FOC ------------
    def _ISBL_DPI(self, installed_equipment_cost):
        # inversión directa de equipos ISBL
        self._ISBL_DPI_cached = installed_equipment_cost - self.OSBL_installed_equipment_cost
        return self._ISBL_DPI_cached

    def _DPI(self, installed_equipment_cost):
        factors = self.warehouse + self.site_development + self.additional_piping
        self._DPI_cached = installed_equipment_cost + self._ISBL_DPI(installed_equipment_cost) * factors
        return self._DPI_cached

    def _depreciable_indirect_costs(self, DPI):
        return DPI * (self.proratable_costs + self.field_expenses + self.construction + self.contingency)

    def _nondepreciable_indirect_costs(self, DPI):
        return DPI * self.other_indirect_costs

    def _TDC(self, DPI):
        return DPI + self._depreciable_indirect_costs(DPI)

    def _FCI(self, TDC):
        self._FCI_cached = TDC + self._nondepreciable_indirect_costs(self._DPI_cached)
        return self._FCI_cached

    def _FOC(self, FCI):
        # costos fijos anuales sin depreciación
        return (FCI * self.property_insurance
                + self._ISBL_DPI_cached * self.maintenance
                + self.labor_cost * (1 + self.labor_burden))

    # ------------ tablas listas para usar ------------
    def CAPEX_table(self):
        return _capex_table_helper(self)  # MM$ por fila/columna

    def FOC_table(self):
        return _foc_table_helper(self)    # MM$ / yr

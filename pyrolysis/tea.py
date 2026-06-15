from biorefineries.tea import create_cellulosic_ethanol_tea

__all__ = ('create_tea',)

def create_tea(sys):
    tea = create_cellulosic_ethanol_tea(sys)
    tea.duration = (2026, 2046)
    tea.income_tax = 0.21
    tea.operating_days = 330
    tea.contingency = 0.4
    tea.depreciation = 'MACRS7' # Standard for waste reducing processes
    return tea
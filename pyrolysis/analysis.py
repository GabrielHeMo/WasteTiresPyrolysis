import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import biosteam as bst
import pyrolysis
import os 
from warnings import filterwarnings
from biosteam.utils import GG_colors

results_folder = os.path.join(os.path.dirname(__file__), 'results')

def run_montecarlo(): 
    filterwarnings('ignore')
    pm = pyrolysis.WasteTirePyrolysisProcess(processing_capacity=None)
    pm.model.exception_hook = 'raise'
    N_samples = 2000
    np.random.seed(1)
    samples = pm.model.sample(N_samples, 'L')
    pm.model.load_samples(samples)
    for n, processing_capacity in enumerate(pyrolysis.key_processing_capacities):
        autoload_file = os.path.join(results_folder, f'monte_carlo_backup_{n}')
        spearman_file = os.path.join(results_folder, f'spearman_{n}.xlsx')
        monte_carlo_file = os.path.join(results_folder, f'monte_carlo_{n}.xlsx')
        pm.processing_capacity = processing_capacity
        pm.model.evaluate(
            notify=100,
            autosave=100,
            autoload=True,
            file=autoload_file,
        )
        pm.model.table.to_excel(monte_carlo_file)
        rho, p = pm.model.spearman_r(filter='omit nan')
        rho.to_excel(spearman_file)

def plot_monte_carlo():
    fig, axes = plot_scatter_1d(
        ('O7',),
        metrics=[biodiesel_yield, MBSP, GWP_biodiesel_allocation],
        xlabel="",
        ylabel=f"MBSP [{format_units('USD/L')}]",
        fs=12,
        aspect_ratio=1,
        width=4,
        colors=['managua']*1,
        zlabel=f"Carbon intensity [{GWP_units_L}]",
        y_center=[0.45, 1.45],
        x_center=1,
        zticks=None,
    )
    bst.plots.set_font(size=9)
    bst.plots.set_figure_size(width='full', aspect_ratio=0.65)
    pm = pyrolysis.WasteTirePyrolysisProcess(simulate=False)
    metrics = [pm.IRR, pm.GWP, pm.FEDI]
    Xi, Yi, Zi = [i.index for i in metrics]
    dfs = [get_monte_carlo(i, metrics) for i in name]
    xs = np.array([df[Xi].values for df in dfs]) * xscale
    ys = np.array([df[Yi].values for df in dfs]) * yscale
    zs = np.array([df[Zi].values for df in dfs]) * zscale
    fig, axes = bst.plots.plot_scatter_1d(
        xs=xs, ys=ys, zs=zs,
        xticks=xticks,
        yticks=yticks,
        zticks=zticks,
        xticklabels=ticklabels, 
        yticklabels=ticklabels,
        xlabel=xlabel,
        ylabel=ylabel,
        autobox=False,
    )
    
    
    plt.subplots_adjust(
        wspace=0,
        top=0.85,
        right=0.8,
        left=0.15,
    )
    fig.text(
        0.45, 0.03,
        f"Biodiesel yield [500 {format_units('L/FF')}]",
        ha='center', 
        rotation='horizontal'
    )
    for i in ('svg', 'png'):
        file = os.path.join(images_folder, f'oilcane_microbial_oil_combined_kde.{i}')
        plt.savefig(file, dpi=900, transparent=True)

def plot_spearman():
    bst.plots.set_font(size=9)
    bst.plots.set_figure_size(aspect_ratio=0.8)
    pm = pyrolysis.WasteTirePyrolysisProcess(simulate=False)
    metrics = [pm.IRR, pm.GWP, pm.FEDI]
    spearman_file = os.path.join(results_folder, 'spearman_0.xlsx')
    df = pd.read_excel(spearman_file, header=[0, 1], index_col=[0, 1])
    rhos = df[[i.index for i in metrics]]
    color_wheel = [
        GG_colors.blue, GG_colors.red, GG_colors.yellow,
    ]
    fig, ax = bst.plots.plot_spearman_2d(
        rhos, index=[],
        color_wheel=color_wheel,
        xlabel="Spearman's rank correlation coefficient",
        w=1.0,
        cutoff=0.01
    )
    plt.legend(['IRR', 'GWP', 'FEDI'])
    return fig, ax

if __name__ == '__main__':
    run_montecarlo()
    plot_spearman()
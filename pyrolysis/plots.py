import numpy as np 
import biosteam as bst

from pyrolysis.system import create_system
from pyrolysis.chemicals import load_chemicals 

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import seaborn as sns
import pandas as pd

from scipy.stats.kde import gaussian_kde
from matplotlib import colormaps
import os 

__all__ = ('plot_kde_', 'plot_spearman','plot_kde_cbar',
           'plot_scatter_gwp','plot_hexbin_gwp','plot_scatter_gwp_all_in_one')

def plot_spearman(file_df, file = 'default.eps' ):
    
    output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, file_df)
    df_test = pd.read_csv(output_path,header=[0, 1, 2], index_col=[0,1] )
    corr_FEDI = df_test['-']['FEDI [damage*meter]']
    corr_GWP_energy = df_test['-']['GWP energy [kg-CO2e/kg]']
    corr_GWP_price = df_test['-']['GWP revenue [kg-CO2e/kg]']
    corr_IRR = df_test['-']['IRR [%]']

    corr_matrix = pd.concat([corr_FEDI, corr_GWP_energy, corr_GWP_price , corr_IRR], axis=1)
    corr_matrix.columns = ['FEDI', 'GWP_E' , 'GWP_P', 'IRR']
    labels = [  'Moisture content (wt$\%$)',
                'Ash content (wt$\%$)',
                'Carbon content (wt$\%$)',
                'Hydrogen content (wt$\%$)',
                'Oxygen content (wt$\%$)',
                'Nitrogen content (wt$\%$)',
                'Sulfur content (wt$\%$)' ,
                'Temperature Reactor (K)',
                'Carbon conversion ($\%$)',
                'Tire price',
                'Diesel price',
                'Low fuel oil price ', 
                'Activated Carbon price',
                'Metals price',]
    corr_matrix.index = labels
    # Si tu índice es MultiIndex, toma el primer nivel como etiqueta legible
    if isinstance(corr_matrix.index, pd.MultiIndex):
        corr_matrix = corr_matrix.copy()
        corr_matrix.index = corr_matrix.index.get_level_values(0)
    # 1) (opcional) reordenar columnas para que IRR sea la primera en la leyenda
    cols = ['IRR', 'FEDI', 'GWP_E', 'GWP_P']  # ajusta a tus nombres exactos
    corr_matrix = corr_matrix[cols]
    mask = (corr_matrix[cols].abs() >= 0.10).any(axis=1)
    corr_filt = corr_matrix  #corr_matrix.loc[mask].copy()
    # ---- 2) ordenar por importancia (máximo |rho| entre métricas) ----
    order = corr_filt['IRR'].abs().sort_values(ascending=True).index
    corr_ord = corr_filt.loc[order]  # filas de menor a mayor (arriba las menos influyentes)

    # ---- 3) preparar posiciones y offsets para barras agrupadas ----
    vars_ = corr_ord.index.to_list()
    y = np.arange(len(vars_)) * 1.8  # posición base por variable
    bar_h = 0.25               # “alto” de cada barra
    offsets = [-bar_h, 0.0, bar_h,bar_h*2] 

    # ---- 4) dibujar en una sola figura ----
    fig, ax = plt.subplots(figsize=(6,6))

    ax.barh(y + offsets[0], corr_ord['IRR'].values, height=bar_h,
            label='IRR', color='lightblue', edgecolor='black', linewidth=0.7)
    ax.barh(y + offsets[1], corr_ord['FEDI'].values,  height=bar_h,
            label='FEDI',  color='salmon', edgecolor='black', linewidth=0.7)
    # ax.barh(y + offsets[2], corr_ord['GWP_E'].values,  height=bar_h,
    #         label='GWP$_\mathrm{Energy}$',  color='lightgreen', edgecolor='black', linewidth=0.7)
    ax.barh(y + offsets[2], corr_ord['GWP_P'].values,  height=bar_h,
            label='GWP$_\mathrm{Revenue}$',  color='lightseagreen', edgecolor='black', linewidth=0.7)

    # Estilo del gráfico
    ax.set_xlabel("Spearman's correlation")
    ax.set_yticks(y)
    ax.set_yticklabels(vars_)
    ax.set_xlim(-1, 1)
    ax.axvline(0, color='k', linewidth=1)        # línea vertical en 0
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.legend(loc='best', frameon=True)

    fig.tight_layout()
    plt.show()

    output_folder = 'results'
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Construct the full relative path
    output_path = os.path.join(output_folder, file)

    fig.savefig(output_path, format='eps', dpi=1200, bbox_inches='tight', pad_inches=0.02)

    # fig.savefig('correlation_matrix.eps', format = 'eps', dpi= 1200)


def plot_kde_(files, ylabel, xlabel, file_figure = 'default.eps', 
              scenarios = None, zorders=None, **kwargs):
    dfs = []
    for file in files:
        output_folder = 'results'
        os.makedirs(output_folder, exist_ok=True)

        output_path = os.path.join(output_folder, file)
        df = pd.read_csv(output_path, header=[0, 1], index_col=0)
        dfs.append(df)
    x_s, y_s = [] , []
    x_index = ('-', 'FEDI [damage*meter]') # FEDI.index
    y_index = ('-', 'IRR [%]' ) #'IRR [%]')
    c_index = ('-', 'GWP revenue [kg-CO2e/kg]' ) 
    for df in dfs:
        x_s.append(df[x_index])
        y_s.append(df[y_index])
    iter = 0 
    fig, ax = plt.subplots(figsize=(6,4))
    colors_list = ['#a0184a' ,  '#fc0a08', '#049fa4'] 
    for x, y  in zip(x_s, y_s):
        xs = x if isinstance(x, (tuple, list)) or x.ndim == 2 else (x,)
        ys = y if isinstance(y, (tuple, list)) or y.ndim == 2 else (y,)
        # print(xs)
        
        colors = [colormaps['viridis']]
        if zorders is None: zorders = len(xs) * [5]

        for x, y, color, zorder in zip(xs, ys,colors, zorders):
            # print(x,y)
            scatter_kwargs = kwargs.copy()
            
            k = gaussian_kde([x, y])
            z = k(np.vstack([x, y]))
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            # scatter_kwargs['cmap'] = color
            # scatter_kwargs['c'] = z

            plt.scatter(x, y, zorder=zorder, label = scenarios[iter], 
                        c= colors_list[iter] , s = 20, alpha= 0.5) # **scatter_kwargs)
        iter += 1
    ax.tick_params(axis='y', labelsize=14)  # cambia el tamaño de los números en eje Y
    ax.tick_params(axis='x', labelsize=14)  # cambia el tamaño de los números en eje X (si hubiera)
    ax.set_xlabel(xlabel, fontsize = 14)
    ax.set_ylabel(ylabel, fontsize = 14)
    ax.set_ylim([0,60])
    ax.legend(loc="upper right",  fontsize= 12, ncol = 1 )
    # ax.tight
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    output_folder = 'results'
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Construct the full relative path
    output_path = os.path.join(output_folder, file_figure)

    fig.savefig(output_path, format='eps', dpi=1200, bbox_inches='tight', pad_inches=0.02)


def plot_kde_cbar(files, ylabel, xlabel, file = 'default.eps', 
              scenarios = None, zorders=None, **kwargs):
    dfs = []
    for file in files:
        output_folder = 'results'
        os.makedirs(output_folder, exist_ok=True)

        output_path = os.path.join(output_folder, file)
        df = pd.read_csv(output_path, header=[0, 1], index_col=0)
        dfs.append(df)
    x_s, y_s = [] , []
    x_index = ('-', 'FEDI [damage*meter]') # FEDI.index
    y_index = ('-', 'IRR [%]' ) #'IRR [%]')
    c_index = ('-', 'GWP revenue [kg-CO2e/kg]' ) 
    for df in dfs:
        x_s.append(df[x_index])
        y_s.append(df[y_index])
    iter = 0 
    fig, ax = plt.subplots(figsize=(6,4))
    colors_list = ['#a0184a' ,  '#fc0a08', '#049fa4'] 
    mark_list =  [".", "o", "*"] 
    # Color map
    cm = plt.cm.get_cmap('RdYlBu')

    for x, y  in zip(x_s, y_s):
        xs = x if isinstance(x, (tuple, list)) or x.ndim == 2 else (x,)
        ys = y if isinstance(y, (tuple, list)) or y.ndim == 2 else (y,)
        # print(xs)
        
        colors = [colormaps['viridis']]
        if zorders is None: zorders = len(xs) * [5]

        for x, y, color, zorder in zip(xs, ys,colors, zorders):
            # print(x,y)
            scatter_kwargs = kwargs.copy()
            
            k = gaussian_kde([x, y])
            z = k(np.vstack([x, y]))
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            scatter_kwargs['cmap'] = color
            scatter_kwargs['c'] = z

            plt.scatter(x, y, zorder=zorder, label = scenarios[iter] , 
                     s = 20, alpha= 0.5, marker= mark_list[iter]  ,  **scatter_kwargs)
        iter += 1
    ax.tick_params(axis='y', labelsize=14)  # cambia el tamaño de los números en eje Y
    ax.tick_params(axis='x', labelsize=14)  # cambia el tamaño de los números en eje X (si hubiera)
    ax.set_xlabel(xlabel, fontsize = 14)
    ax.set_ylabel(ylabel, fontsize = 14)
    ax.set_ylim([0,60])
    ax.legend(loc="upper right",  fontsize= 12, ncol = 1 )
    # ax.tight
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    output_folder = 'results'
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Construct the full relative path
    output_path = os.path.join(output_folder, file)

    fig.savefig(output_path, format='eps', dpi=1200, bbox_inches='tight', pad_inches=0.02)

def plot_scatter_gwp(files, ylabel, xlabel, file_figure='default.eps',
                     scenarios=None, **kwargs):
    """
    Genera un scatter FEDI vs IRR para cada archivo (escenario),
    con color por GWP revenue y su propio colorbar.
    """

    output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)

    # Leer dataframes
    dfs = []
    for fname in files:
        output_path = os.path.join(output_folder, fname)
        df = pd.read_csv(output_path, header=[0, 1], index_col=0)
        dfs.append(df)

    # Índices de columnas
    x_index = ('-', 'FEDI [damage*meter]')
    y_index = ('-', 'IRR [%]')
    c_index = ('-', 'GWP revenue [kg-CO2e/kg]')

    # Para que todos los subplots tengan la misma escala de color
    all_c = np.concatenate([df[c_index].values for df in dfs])
    vmin, vmax = np.min(all_c), np.max(all_c)

    n = len(dfs)
    if scenarios is None:
        scenarios = [f'Scenario {i+1}' for i in range(n)]

    # Crear figure y ejes
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharex=True, sharey=True)

    # Si solo hay un escenario, axes no es lista
    if n == 1:
        axes = [axes]

    cmap = colormaps['viridis']

    for ax, df, scen in zip(axes, dfs, scenarios):
        x = df[x_index].values
        y = df[y_index].values
        c = df[c_index].values

        scatter_kwargs = dict(s=20, alpha=0.6, edgecolors='none')
        scatter_kwargs.update(kwargs)

        sc = ax.scatter(x, y, c=c, cmap=cmap, vmin=vmin, vmax=vmax,
                        **scatter_kwargs)

        ax.set_title(scen, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)

        # Colorbar individual para cada subplot
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"GWP revenue [kg CO$_2$e $\cdot$ kg$^{-1}$]",
                       fontsize=12)

    # Etiquetas globales (como solo hay una fila de subplots)
    axes[0].set_ylabel(ylabel, fontsize=14)
    for ax in axes:
        ax.set_xlabel(xlabel, fontsize=14)

    # Límites de IRR si quieres fijarlos como antes
    axes[0].set_ylim(0, 60)

    plt.tight_layout()

    # Guardar figura
    output_path_fig = os.path.join(output_folder, file_figure)
    fig.savefig(output_path_fig, format='eps', dpi=1200,
                bbox_inches='tight', pad_inches=0.02)
    plt.show()

def plot_hexbin_gwp(files, ylabel, xlabel, file_figure='default.eps',
                    scenarios=None, **kwargs):
    """
    Genera un hexbin FEDI vs IRR para cada archivo (escenario),
    con color por GWP revenue y su propio colorbar.
    Los límites de cada subplot se ajustan como:
        x_min = 0.9 * min(FEDI), x_max = 1.1 * max(FEDI)
        y_min = 0.9 * min(IRR),  y_max = 1.1 * max(IRR)
    """

    output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)

    # Leer dataframes
    dfs = []
    for fname in files:
        output_path = os.path.join(output_folder, fname)
        df = pd.read_csv(output_path, header=[0, 1], index_col=0)
        dfs.append(df)

    # Índices de columnas
    x_index = ('-', 'FEDI [damage*meter]')
    y_index = ('-', 'IRR [%]')
    c_index = ('-', 'GWP revenue [kg-CO2e/kg]')

    # Escala global de color (para que los subplots sean comparables)
    all_c = np.concatenate([df[c_index].values for df in dfs])
    vmin, vmax = np.min(all_c), np.max(all_c)

    n = len(dfs)
    if scenarios is None:
        scenarios = [f'Scenario {i+1}' for i in range(n)]

    # Crear figura y ejes (sin sharex/sharey para poder ajustar cada uno)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    if n == 1:
        axes = [axes]

    cmap = colormaps['viridis']

    for ax, df, scen in zip(axes, dfs, scenarios):
        x = df[x_index].values
        y = df[y_index].values
        c = df[c_index].values

        # Hexbin: color = GWP promedio en cada celda
        hex_kwargs = dict(gridsize=35, reduce_C_function=np.mean, mincnt=1)
        hex_kwargs.update(kwargs)

        hb = ax.hexbin(
            x, y, C=c,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            **hex_kwargs
        )

        # Límites individuales por subplot
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        ax.set_xlim(0.9 * x_min, 1.1 * x_max)
        ax.set_ylim(0.9 * y_min, 1.1 * y_max)

        ax.set_title(scen, fontsize=24)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=13)

        # Colorbar individual por subplot
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label(r"GWP revenue [kg CO$_2$e $\cdot$ kg$^{-1}$]",
                       fontsize=12)

    # Etiquetas globales
    axes[0].set_ylabel(ylabel, fontsize=24)
    for ax in axes:
        ax.set_xlabel(xlabel, fontsize=24)

    plt.tight_layout()

    # Guardar figura
    output_path_fig = os.path.join(output_folder, file_figure)
    fig.savefig(output_path_fig, format='eps', dpi=1200,
                bbox_inches='tight', pad_inches=0.02)
    plt.show()

def plot_scatter_gwp_all_in_one(files, ylabel, xlabel, file_figure='default.eps',
                                scenarios=None, **kwargs):
    """
    Genera un scatter FEDI vs IRR donde todos los escenarios se grafican
    en una sola figura, con:
    - distinto marker por escenario
    - color por GWP revenue
    - un solo colorbar
    """

    output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)

    dfs = []
    for fname in files:
        output_path = os.path.join(output_folder, fname)
        df = pd.read_csv(output_path, header=[0, 1], index_col=0)
        dfs.append(df)

    # Columnas
    x_index = ('-', 'FEDI [damage*meter]')
    y_index = ('-', 'IRR [%]')
    c_index = ('-', 'GWP revenue [kg-CO2e/kg]')

    # Determinar min y max global para colorbar
    all_c = np.concatenate([df[c_index].values for df in dfs])
    vmin, vmax = np.min(all_c), np.max(all_c)

    if scenarios is None:
        scenarios = [f"Scenario {i+1}" for i in range(len(files))]

    # Markers distintos por escenario
    markers = ["o", "s", "D", "^", "v", "*"]  # puedes agregar más si necesitas
    colors_edge = ["black", "black", "black"]

    cmap = colormaps["viridis"]

    # Figura única
    fig, ax = plt.subplots(figsize=(7, 5))

    for idx, (df, scen) in enumerate(zip(dfs, scenarios)):
        x = df[x_index].values
        y = df[y_index].values
        c = df[c_index].values

        scatter_kwargs = dict(s=35, alpha=0.65, edgecolors='none')
        scatter_kwargs.update(kwargs)

        sc = ax.scatter(
            x, y,
            c=c,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            marker=markers[idx % len(markers)],
            label=scen,
            **scatter_kwargs
        )

    # Etiquetas
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 60])

    # Leyenda (solo los markers, sin color)
    ax.legend(title="Scenarios", fontsize=12)

    # Un solo colorbar global
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"GWP revenue [kg CO$_2$e $\cdot$ kg$^{-1}$]", fontsize=12)

    plt.tight_layout()

    # Guardar
    fig_path = os.path.join(output_folder, file_figure)
    fig.savefig(fig_path, format='eps', dpi=1200, bbox_inches='tight', pad_inches=0.02)
    plt.show()
import numpy as np 
import biosteam as bst

from pyrolysis.system import create_system
from pyrolysis.chemicals import load_chemicals 

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
from matplotlib import colormaps
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd

from scipy.stats.kde import gaussian_kde
from matplotlib import colormaps
import os 
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "STIXGeneral",
    "text.latex.preamble": r"\usepackage[utf8]{inputenc}\usepackage{amsmath}",
    # "figure.figsize": [12, 4],  # ancho, Largo  
    # "xtick.labelsize": 13,  # tamaño ticks en eje x
    # "ytick.labelsize": 13   # tamaño ticks en eje y
})
plt.rcParams["text.latex.preamble"] = r"\usepackage{xcolor}"


__all__ = ('plot_kde_', 'plot_spearman', 'plot_kde_cbar',
            'plot_scatter_gwp', 'plot_hexbin_gwp', 'plot_scatter_gwp_all_in_one',
            'plot_gwp', 'plot_one_dot_explainer')

def plot_spearman(file_df, file = 'default.eps' ):
    
    output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, file_df)
    df_test = pd.read_csv(output_path,header=[0, 1, 2], index_col=[0,1] )
    corr_FEDI = df_test['-']['FEDI [damage*meter]']
    corr_GWP_energy = df_test['-']['GWP energy activatedcarbon [kg-CO2e/kg]']
    corr_GWP_price = df_test['-']['GWP revenue activatedcarbon [kg-CO2e/kg]']
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
                'Tire price  ($\$ \,kg^{-1}$)',
                'Diesel price ($\$ \,kg^{-1}$)',
                'Low fuel oil price  ( $\$ \,kg^{-1}$)', 
                'Activated Carbon price  ( $\$ \,kg^{-1}$)',
                'Metals price  ( $\$ \,kg^{-1}$)',]
    # labels   = [ 'Contenido de humedad ($\%$ en peso)',
    #             'Contenido de cenizas ($\%$ en peso)',
    #             'Contenido de carbono ($\%$ en peso)',
    #             'Contenido de hidrógeno ($\%$ en peso)',
    #             'Contenido de oxígeno ($\%$ en peso)',
    #             'Contenido de nitrógeno ($\%$ en peso)',
    #             'Contenido de azufre ($\%$ en peso)' ,
    #             'Temperatura del reactor (K)',
    #             'Conversión de carbono ($\%$)',
    #             'Precio de llantas ($\$ \,kg^{-1}$)',
    #             'Precio de diésel ($\$ \,kg^{-1}$)',
    #             'Precio de LFO ($\$ \,kg^{-1}$)',
    #             'Precio de carbón activado ($\$ \,kg^{-1}$)',
    #             'Precio de los metales ($\$ \,kg^{-1}$)',]
    bounds = [
        (0.4, 4.0, 12.13),
        (0.0, 2.5, 9.89),
        (75.0, 83.3, 89.9),
        (6.56, 7.5, 7.99),
        (1.29, 4.5, 10.79),
        (0.3, 0.6, 1.0),
        (0.87, 1.6, 2.46),
        (773.15, 823.15, 1073.15),
        (30.0, 55, 80.0),
        (0.11, 0.14, 0.17),
        (0.91, 1.13, 1.365),
        (0.34, 0.42, 0.50),
        (1.80, 3.50, 4.20),
        (0.18, 0.23, 0.27),
    ]
    # Crear labels con bounds en segunda línea

    labels = [
    rf"{lab}" + "\n" +
    rf"[$\color{{blue}}{{{bmin:g}}},\ \color{{black}}{{{bmean:g}}},\ \color{{red}}{{{bmax:g}}}$]"
    for lab, (bmin, bmean, bmax) in zip(labels, bounds)]

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
    y = np.arange(len(vars_)) *2.5  # posición base por variable
    bar_h = 0.25               # “alto” de cada barra
    offsets = [-bar_h, 0.0, bar_h,bar_h*2] 

    # ---- 4) dibujar en una sola figura ----
    fig, ax = plt.subplots(figsize=(6,6))
    fontsize = 11.5
    ax.barh(y + offsets[0], corr_ord['IRR'].values, height=bar_h,
            label='IRR', color='lightblue', edgecolor='black', linewidth=0.7)
    ax.barh(y + offsets[1], corr_ord['FEDI'].values,  height=bar_h,
            label='FEDI',  color='salmon', edgecolor='black', linewidth=0.7)
    ax.barh(y + offsets[2], corr_ord['GWP_P'].values,  height=bar_h,
            label='GWP$_\mathrm{Revenue}$',  color='lightseagreen', edgecolor='black', linewidth=0.7)

    # Estilo del gráfico
    ax.set_xlabel("Spearman's correlation", fontsize=fontsize)
    ax.set_yticks(y)
    ax.set_yticklabels(vars_, fontsize = fontsize)
    ax.set_xlim(-1, 1)
    ax.axvline(0, color='k', linewidth=1)        # línea vertical en 0
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.legend(loc='lower left', frameon=True, fontsize = fontsize)

    fig.tight_layout()
    plt.show()

    output_folder = 'figures'
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
    ax.legend(loc="lower left",  fontsize= 12, ncol = 1 )
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
                    scenarios=None, benchmarks=None,
                    extend_limits_for_benchmarks=True,
                    show_benchmark_text=False,
                    **kwargs):

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
    c_index = ('-', 'GWP revenue activatedcarbon [kg-CO2e/kg]')

    # Escala global de color
    all_c = np.concatenate([df[c_index].values for df in dfs])
    vmin, vmax = np.min(all_c), np.max(all_c)

    n = len(dfs)
    if scenarios is None:
        scenarios = [f'Scenario {i+1}' for i in range(n)]

    # ---------- FIGURA CON COLUMNA EXTRA PARA COLORBAR (tipo tu ejemplo) ----------
    # 3 subplots + 1 columna angosta para el colorbar
    fig = plt.figure(figsize=(5 * n + 0.6, 6))
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1]*n + [0.045], wspace=0.12)
    fontsize = 24
    axes = []
    for i in range(n):
        if i == 0:
            ax = fig.add_subplot(gs[0, i])
        else:
            ax = fig.add_subplot(gs[0, i], sharey=axes[0])
        axes.append(ax)

    cax = fig.add_subplot(gs[0, -1])  # eje dedicado al colorbar

    cmap = colormaps['viridis']
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # ---- helpers ----
    def fmt_pc(v):
        if v is None:
            return "?"
        return f"{v:,.0f}"

    def bench_label(b):
        author = b.get("author", b.get("label", "Literature"))
        year = b.get("year", "")
        pc = b.get("pc", b.get("capacity", None))
        return f"{author} {year} — {fmt_pc(pc)} kg/h".strip()

    # ---------- LIMITES GLOBALES PARA SHAREY ----------
    all_y = np.concatenate([df[y_index].values for df in dfs])
    y_min_global, y_max_global = np.min(all_y), np.max(all_y)

    if extend_limits_for_benchmarks and benchmarks:
        bench_y_vals = []
        for b in benchmarks:
            if "irr" in b:
                bench_y_vals.append(float(b["irr"]))
            elif "irr_range" in b:
                lo, hi = b["irr_range"]
                bench_y_vals += [float(lo), float(hi)]
        if bench_y_vals:
            y_min_global = min(y_min_global, min(bench_y_vals))
            y_max_global = max(y_max_global, max(bench_y_vals))

    y_min_global *= 0.9
    y_max_global *= 1.1

    # Leyenda global
    legend_map = {}

    # ---------------------- LOOP ----------------------
    for ax, df, scen in zip(axes, dfs, scenarios):
        x = df[x_index].values
        y = df[y_index].values
        c = df[c_index].values

        hex_kwargs = dict(gridsize=35, reduce_C_function=np.mean, mincnt=1)
        hex_kwargs.update(kwargs)

        ax.hexbin(x, y, C=c, cmap=cmap, norm=norm, **hex_kwargs)

        # x-limits individuales
        x_min, x_max = np.min(x), np.max(x)
        ax.set_xlim(0.9 * x_min, 1.1 * x_max)

        # y-limits globales
        ax.set_ylim(y_min_global, y_max_global)

        ax.set_title(scen, fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)

        # Benchmarks (solo líneas; si llega irr_range -> promedio)
        if benchmarks:
            for i, b in enumerate(benchmarks):
                color = b.get("color", f"C{i}")
                lw = b.get("lw", 2.0)
                ls = b.get("linestyle", "-")
                alpha = b.get("alpha", 0.95)

                lab = bench_label(b)

                if "irr" in b:
                    irr = float(b["irr"])
                else:
                    lo, hi = b["irr_range"]
                    irr = 0.5 * (float(lo) + float(hi))

                ax.axhline(irr, color=color, lw=lw, ls=ls, alpha=alpha)

                if lab not in legend_map:
                    legend_map[lab] = Line2D([0], [0], color=color, lw=lw, ls=ls, alpha=alpha)

    # Etiquetas
    axes[0].set_ylabel(ylabel, fontsize=fontsize)
    for ax in axes:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)  # limpia y mantiene sharey

    # Colorbar global en eje dedicado
    cbar = fig.colorbar(sm, cax=cax)
    # cbar.set_label(r"GWP$_{revenue}$ [kg CO$_2$e $\cdot$ kg$^{-1}$]", fontsize=fontsize)
    cbar.set_label(r"GWP$_{revenue}$ [kg CO$_2$e $\cdot$ kg$^{-1}$]", fontsize=fontsize)
    cax.tick_params(labelsize=fontsize)

    # Leyenda global arriba
    if legend_map:
        labels = list(legend_map.keys())
        handles = [legend_map[k] for k in labels]
        fig.legend(handles, labels,
                   loc="upper center",
                   bbox_to_anchor=(0.5, 1.25),
                   ncol=2,
                   frameon=True,
                   fontsize=fontsize)

    # En GridSpec, esto funciona mejor que tight_layout:
    fig.subplots_adjust(top=0.80)   # deja espacio para la leyenda
    # (el wspace ya está controlado por gridspec)
    output_folder = 'figures'
    # Guardar
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


def plot_one_dot_explainer(df_final, means_by_class, std_by_class,
                                         feature_col, class_order,
                                         cmap="coolwarm", vmin=-2, vmax=2,
                                         size_min=30, size_max=300,
                                         y_label=None,
                                         jitter=0.10, points_size=6, points_alpha=0.25):
    """
    One-dot explainer using:
    - strip/jitter plot (raw points distribution on Y)
    - central point = class mean
    - whiskers = ± 1 standard deviation
    - colored/size-encoded central dot: color=z-score (across classes), size=CV
    """

    # --- extract distributions per class (raw samples) ---
    data_per_class = [
        df_final.loc[df_final["class"] == c, feature_col].dropna().values.astype(float)
        for c in class_order
    ]

    # --- mean/std per class (consistent with dot-heatmap inputs) ---
    mu = np.array([means_by_class.loc[c, feature_col] for c in class_order], dtype=float)
    sd = np.array([std_by_class.loc[c, feature_col]   for c in class_order], dtype=float)

    # --- z-score across class means ---
    denom = np.std(mu, ddof=0)
    z = np.zeros_like(mu) if np.isclose(denom, 0.0) else (mu - mu.mean()) / denom
    z = np.clip(z, vmin, vmax)

    # --- CV for marker size ---
    eps = 1e-12
    cv = sd / (np.abs(mu) + eps)
    cv_min, cv_max = float(np.nanmin(cv)), float(np.nanmax(cv))
    if np.isclose(cv_max - cv_min, 0.0):
        s = np.full_like(cv, (size_min + size_max) / 2.0, dtype=float)
    else:
        t = (cv - cv_min) / (cv_max - cv_min)
        s = size_min + t * (size_max - size_min)

    # --- plot ---
    # fig, ax = plt.subplots(figsize=(9, 4.8))
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi = 250)
    x = np.arange(1, len(class_order) + 1)
    fontsize = 13.5

    # raw points with jitter
    rng = np.random.default_rng(123)
    for i, d in enumerate(data_per_class):
        xj = x[i] + rng.uniform(-jitter, jitter, size=len(d))
        ax.scatter(xj, d, s=points_size, alpha=points_alpha, edgecolors="none", zorder=1)

    # mean ± std (whiskers)
    ax.errorbar(x, mu, yerr=sd, fmt="none", capsize=4, linewidth=1.2, color="k", zorder=2)

    # central dot: mean (color=z, size=CV)
    sc = ax.scatter(x, mu, c=z, s=s, cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolors="k", linewidths=0.8, zorder=3)

    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Within-parameter standardized deviation (z-score)", fontsize=fontsize)
    # cbar.s
    cbar.ax.tick_params(labelsize=fontsize)
    scenarios = [
        r"High$_{\mathrm{IRR}}$" +  "\n & \n"    + r" Low$_{\mathrm{FEDI}}$",
        r"High$_{\mathrm{IRR}}$" +  "\n & \n"    + r" High$_{\mathrm{FEDI}}$",
        r"Low$_{\mathrm{IRR}}$" +  "\n & \n"    + r" Low$_{\mathrm{FEDI}}$",
        r"Low$_{\mathrm{IRR}}$" +  "\n & \n"    + r" High$_{\mathrm{FEDI}}$",
    ]
    
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=0, fontsize=fontsize)
    
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_ylabel('Carbon conversion [%wt]', fontsize=fontsize)
    ax.grid(True, alpha=0.2)
    
    # plt.tight_layout()

    # save
    output_folder = "figures"
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, "Figure13a.eps")
    fig.savefig(out_path, format="eps", dpi=1200, bbox_inches="tight", pad_inches=0.02)
    plt.show()



def plot_gwp(file_df='default.csv', file='FigureS1.eps'):

    output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)

    candidate = os.path.join(output_folder, file_df)
    csv_path = candidate if os.path.exists(candidate) else file_df
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No encontré el archivo '{file_df}' ni en '{output_folder}/' ni como ruta directa."
        )

    df = pd.read_csv(csv_path, header=[0, 1], index_col=0)

    allocs = ['energy', 'revenue']
    products = [
        ('activatedcarbon', 'Activated carbon'),
        ('LFO', 'LFO'),
        ('diesel', 'Diesel'),
        ('metals', 'Metals'),
    ]

    def _col(method, prod_key):
        return ('-', f'GWP {method} {prod_key} [kg-CO2e/kg]')

    def _bxp_stats(x, label):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return None
        mean = float(np.mean(x))
        q1, q3 = np.percentile(x, [25, 75])
        return dict(
            label=label,
            whislo=float(np.min(x)),
            q1=float(q1),
            med=float(mean),   # línea central = media
            q3=float(q3),
            whishi=float(np.max(x)),
            fliers=[]
        )


    # =========================
    # Figura
    # =========================
    fig = plt.figure(figsize=(10, 11))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.08)

    ax_energy = fig.add_subplot(gs[0])
    ax_revenue = fig.add_subplot(gs[1], sharex=ax_energy)

    axes = {
        'energy': ax_energy,
        'revenue': ax_revenue
    }

    def _style_bxp(bp):
        for box in bp['boxes']:
            box.set_alpha(0.35)
        for whisk in bp['whiskers']:
            whisk.set_linewidth(1.2)
        for cap in bp['caps']:
            cap.set_linewidth(1.2)
        for med in bp['medians']:
            med.set_linewidth(2.0)

    # Guardar resumen estadístico
    summary_results = {}

    # =========================
    # LOOP methods
    # =========================
    for method in allocs:
        ax = axes[method]

        stats = []
        missing_cols = []
        summary_results[method] = {}

        for prod_key, prod_label in products:
            col = _col(method, prod_key)
            if col not in df.columns:
                missing_cols.append(col)
                stats.append(None)
                continue

            values = df[col].values

            # stats para boxplot
            s = _bxp_stats(values, prod_label)
            stats.append(s)


        if any(s is None for s in stats):
            ax.text(
                0.5, 0.5,
                "Faltan columnas:\n" + "\n".join([str(c) for c in missing_cols]),
                ha='center', va='center', transform=ax.transAxes, fontsize=10
            )
            ax.set_axis_off()
            continue

        bp = ax.bxp(stats, showfliers=False, patch_artist=True)
        _style_bxp(bp)

        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylabel(f"{method.capitalize()} allocation\nGWP [kg-CO$_2$e/kg]", fontsize=12)

        if method == 'energy':
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    ax_revenue.set_xlabel("Products", fontsize=12)

    plt.tight_layout()

    out_path = os.path.join(output_folder, file)
    fig.savefig(out_path, format='eps', dpi=1200, bbox_inches='tight', pad_inches=0.02)
    plt.show()
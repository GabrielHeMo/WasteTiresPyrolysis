import numpy as np 
import biosteam as bst

from pyrolysis.system import create_system
from pyrolysis.chemicals import load_chemicals 

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import seaborn as sns

import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "STIXGeneral",
    "text.latex.preamble": r"\usepackage[utf8]{inputenc}\usepackage{amsmath}",
    "figure.figsize": [7.0, 4.0],  # ancho, Largo  
    # "xtick.labelsize": 11,  # tamaño ticks en eje x
    # "ytick.labelsize": 11,   # tamaño ticks en eje y
    # "legend.fontsize": 11,
    # "figure.titlesize": 11,
})

# plt.rcParams.update({
#     "font.size": 9,
#     "axes.titlesize": 9,
#     "axes.labelsize": 9,
#     "xtick.labelsize": 9,
#     "ytick.labelsize": 9,
#     "legend.fontsize": 9,
#     "figure.titlesize": 9,
# })


__all__ = ('carbonyield_at_carbonprice_titer',
           '_contour_subplots', 
            '_plot_contour_single_metric',
             'titer_carbonyield_price', 
             'plot_titer_carbonyield_price',
             'titer_carbonyield_dieselprice',
            'plot_titer_carbonyield_dieselprice',
            'titer_carbonyield_temperature_FEDI',
            'plot_titer_carbonyield_temperature_FEDI')

def _contour_subplots(nrows, ncols, single_colorbar=True):
    wbar = 0.3
    widths = np.ones(ncols + 1)
    widths[-1] *= wbar / 4
    gs_kw = dict(width_ratios=widths)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols + 1 , sharex=True, sharey=True, gridspec_kw=gs_kw)
    axes = axes.reshape([nrows, ncols + 1])
    if single_colorbar:
        gs = axes[0, 0].get_gridspec()
        for ax in axes[:, -1]: ax.remove()
        ax_colorbar = fig.add_subplot(gs[:, -1])
        return fig, axes, ax_colorbar
    else:
        return fig, axes


def _plot_contour_single_metric(
        X, Y, Z, xlabel, ylabel, xticks, yticks, metric_bar, line_lvls ,cbar_ax_title,  file, 
        titles=None, fillcolor=None, styleaxiskw=None, label=False,
        contour_label_interval=2):
    *_, nrows, ncols = Z.shape

    fontsize =  18.5 # 13.75
    assert Z.shape == (*X.shape, nrows, ncols), (
        "Z.shape must be (X, Y, M, N), where (X, Y) is the shape of both X and Y"
    )
    fig, axes, ax_colorbar = _contour_subplots( nrows, ncols, single_colorbar=True)
    if styleaxiskw is None: styleaxiskw = dict(xtick0=False, ytick0=False)
    cps = np.zeros([nrows, ncols], dtype=object)
    # linecolor = np.array([*c.neutral_shade.RGBn, 0.1])
    other_axes = []
    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row, col]
            plt.sca(ax)

            # style_plot_limits(xticks, yticks)
            yticklabels = col == 0
            xticklabels = row == nrows - 1
            
            metric_data = Z[:, :, row, col]
            cp = plt.contourf(X, Y, metric_data,
                              levels=metric_bar.levels,
                              cmap=metric_bar.cmap,
                              norm=metric_bar.norm)
            cp.set_edgecolors('face') # For svg background
            

            # Esto es una buena opcion para arriba: line_lvls = metric_bar.levels[::6]  # ajusta 5 -> 4, 6, etc.
            cs = plt.contour(
                X, Y, metric_data,
                levels=line_lvls, colors=[[0,0,0,0.25]], linewidths=0.8, zorder=1)
            
            # --- Etiquetas sólo en esas líneas (redondeadas) ---
            label_lvls = line_lvls
            clabels = ax.clabel(cs, levels=label_lvls, inline=True, fmt='%.0f',  # cambia a '%.1f' si quieres
                                colors=['k'], fontsize= fontsize )
            # Evita que se corten en los bordes:
            for t in clabels:
                t.set_clip_on(False)
                t.set_bbox(dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.25'))



            cps[row, col] = cp
            if row == nrows - 1 and not styleaxiskw.get('ytick0', True):
                sak = styleaxiskw.copy()
                sak['ytick0'] = True
            else:
                sak = styleaxiskw
            if col == 0 and not styleaxiskw.get('xtick0', True):
                sak = sak.copy()
                sak['xtick0'] = True
            dct = bst.utils.style_axis(ax, xticks, yticks, xticklabels, yticklabels, **sak)
            other_axes.append(dct)
    #  COLORBAR

    cbarkwargs = {}
    cbarkwargs['fraction'] = 0.5
    cbarkwargs['pad'] = 0.15

    ticks = np.round(metric_bar.levels, 0)
    cbar = fig.colorbar(cp, cax=ax_colorbar, ticks=ticks, boundaries=metric_bar.levels,   **cbarkwargs)

    cbar.locator = mticker.MaxNLocator(nbins=6) #mticker.FixedLocator(metric_bar.levels)
    cbar.formatter = mticker.FormatStrFormatter('%.0f')  # o '%.1f' o '%.0f'
    cbar.update_ticks()
    cbar_ax = cbar.ax
    cbar_ax.set_title( cbar_ax_title ,fontsize=fontsize)
    ub = False
    lb = False
    try:
        ylabels = [y.get_text() for y in cbar_ax.get_yticklabels()]
        ylabels = [(i if i[0].isdigit() else '-'+i[1:]) for i in ylabels]

        if ub:
            ylabels[-1] = '>' + ylabels[-1]
        if lb:
            ylabels[0] = '<' + ylabels[0]
        cbar_ax.set_yticklabels(ylabels, fontsize= fontsize)
    except:
        pass
    for axrow in axes:
        for ax in axrow[:-1]:  # evita la columna de colorbar
            ax.tick_params(axis='y', labelsize=fontsize)  # which='major'
            # ax.set_yticklabels([f"{yt:.1f}" for yt in yticks])  # ejemplo
    if titles:
        for col, title in enumerate(titles):
            ax = axes[0, col]
            ax.set_title(title, color='black', fontsize=fontsize, fontweight='bold')
    
    for ax in axes[:, :-1].ravel():           # evita la columna del colorbar
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))     # 5–6 ticks máx
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))  # sin decimales
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))     # 5–6 ticks máx
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.01f'))  # sin decimales
        # ax.tick_params(axis='x', pad=1) 
        ax.tick_params(axis='x', labelsize=fontsize, pad=1)
        
   
    plt.subplots_adjust(
    left=0.12,   # mueve más cerca el ylabel
    right=0.9,   # deja espacio para la cbar
    bottom=0.15, # mueve más cerca el xlabel
    top=0.9, 
    wspace=0.05, 
    hspace=0.1)

    fig = plt.gcf()
    fig.supxlabel(xlabel, fontsize = fontsize)
    fig.supylabel(ylabel, fontsize = fontsize, x=0.06, ha='center')
    # fig.tight_layout()
    # fig.savefig(file, format='eps', dpi=1200, bbox_inches='tight', pad_inches=0.02)
    
    
    return fig, axes, cps, cbar, other_axes

# TITER FUNCTIONS PARA IRR

def carbonyield_at_carbonprice_titer(carbon_yield, carbon_price , titer):
    settings = load_chemicals()
    x = carbon_yield #bst.F.MainReactor.carbon_conversion  = carbon_yield  # buscar variables que voy a modificar
    y = carbon_price #bst.F.CarbonActivated.price = carbon_price
    IRR_array = np.zeros_like(titer)  # Variable to get
    for i, value in enumerate(titer):
        context_sys, tea, NPV , IRR_value , Fedi_val, GWP_revenue  = create_system(processing_capacity=value, x = x, y = y,  
                                                                      settings = settings, type = 'Titer1'  ) 
        context_sys.simulate()

        IRR_array[i] = IRR_value   # microbial_lipids_tea.solve_price(lipid_product)  tea.IRR = tea.solve_IRR()
    return IRR_array

def carbonyield_at_dieselprice_titer(carbon_yield, diesel_price , titer):
    settings = load_chemicals()
    x = carbon_yield #bst.F.MainReactor.carbon_conversion  = carbon_yield  # buscar variables que voy a modificar
    y = diesel_price #bst.F.CarbonActivated.price = carbon_price
    IRR_array = np.zeros_like(titer)  # Variable to get
    for i, value in enumerate(titer):
        context_sys, tea, NPV , IRR_value, Fedi_val , GWP_revenue = create_system(processing_capacity=value, x = x, y = y , 
                                                                     settings = settings, type = 'Titer2' ) 
        context_sys.simulate()

        IRR_array[i] = IRR_value   # microbial_lipids_tea.solve_price(lipid_product)  tea.IRR = tea.solve_IRR()
    return IRR_array

# TITER FUNCTIONS PARA FEDI

def FEDI_carbonyield_temperature_FEDI(carbon_yield, temperature_val , titer):
    settings = load_chemicals()
    x = carbon_yield 
    y = temperature_val 
    FEDI_array = np.zeros_like(titer)  # Variable to get
    for i, value in enumerate(titer):
        context_sys, tea, NPV , IRR_value, Fedi_val, GWP_revenue  = create_system(processing_capacity=value, x = x, y = y ,
                                                                     settings = settings, type = 'Titer3' ) 
        context_sys.simulate()
        FEDI_array[i] = Fedi_val   # microbial_lipids_tea.solve_price(lipid_product)  tea.IRR = tea.solve_IRR()
    return FEDI_array




def titer_carbonyield_price(file = 'default.npz', n = 10):

    titer = np.array([2500,4166,6250])  # puntos 
    carbon_conversion = 55
    carbon_price = 3.5
    xlim = np.array([0.75 * carbon_conversion, 1.25 * carbon_conversion])
    ylim = np.array([0.75 * carbon_price, 1.25 * carbon_price])
    X, Y, Z = bst.plots.generate_contour_data(
        carbonyield_at_carbonprice_titer,
        xlim=xlim, ylim=ylim,
        args=(titer,),
        n = n,)  # numero de puntos a evaluar por en x  y y 
    # Guarda vectores y despues cargarlos 
    output_folder = 'results'
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Construct the full relative path
    output_path = os.path.join(output_folder, file)
    np.savez(output_path, array_a=X, array_b=Y, array_c = Z)

def plot_titer_carbonyield_price(file = 'default.npz'):
    output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, file)
    
    # Recupera valores
    npzfile  = np.load( output_path )
    X = npzfile['array_a']
    Y = npzfile['array_b']
    Z = npzfile['array_c']

    # xlabel = r"$\mathrm{Conversion}\,\mathrm{de}\,\mathrm{carbon} \, %[\%]$"
    # xlabel = r"$\mathrm{Carbon}\,\mathrm{conversion} \, [\%]$" #"Conversión de carbono $[\%]$" #r"Conversion de carbono %[\%]$"
    # ylabel = r"Carbon activated price" +  "\n" +  r"$[\mathrm{USD} \cdot \mathrm{kg}^{-1}]$"


    xlabel = r"$\mathrm{Carbon}\,\mathrm{conversion} \, [\%]$" #"Conversión de carbono $[\%]$" #r"Conversion de carbono %[\%]$"
    ylabel = r"$\mathrm{Carbon\ activated\ price}$" + "\n" + r"$[\mathrm{USD}\cdot\mathrm{kg}^{-1}]$"

    units = r'c'
    # titles = [r"$\mathrm{PC}_{\max}$" + "\n" + r"$6250\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$", r"$\mathrm{PC}_{\text{mean}}$" + "\n" + r"$4166\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$", r"$\mathrm{PC}_{\min}$" + "\n" + r"$2500\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$"]
    
    titles = [ "\n" + r"$6250\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$",
            r"$\textrm{Processing Capacities}$" + "\n" + r"$4166\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$", 
            "\n" + r"$2500\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$"]
    

    titles.reverse()
    xticks = X[0,:] #[30, 55, 70 ]
    yticks = Y[:,0] #[0.16, 0.3, 0.40]


    metric_bar = bst.plots.MetricBar(
    r'MSP', r'USD', plt.cm.get_cmap('viridis'),
    bst.plots.rounded_tickmarks_from_data(Z, 5, 1, expand=0, p=0.5), 50, 0  # Levels, nivel de redondeo 
    )
    cbar_ax_title = '$\mathrm{IRR}\,[\%]$'
    line_lvls =  [ 10.449,  12.898, 20.245, 24.327 , 29,  37, 43.102]
    file = 'Figure31_Carbonoprice.svg' #'IRR_CarbonActivatedPrice_CarbonConversion.eps'
    fig, axes, cps, cbar, other_axes = _plot_contour_single_metric(
        X, Y, Z[:, :, None, :], xlabel, ylabel, xticks, yticks, metric_bar, 
        line_lvls , cbar_ax_title, file , 
        titles=titles, fillcolor=None,
        styleaxiskw=dict(xtick0=False), label=True) 
    
    output_folder = 'figures'
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Construct the full relative path
    output_path = os.path.join(output_folder, file)

    fig.savefig(output_path, format='svg', dpi=1200, bbox_inches='tight', pad_inches=0.02)


    # return fig, axes, cps, cbar, other_axes

# Hasta aca voy

def titer_carbonyield_dieselprice(file = 'default.npz', n = 10):
    titer = np.array([2500,4166,6250])  # puntos 
    carbon_conversion = 55
    diesel_price = 1137.50/1000
    xlim = np.array([0.75 * carbon_conversion, 1.25 * carbon_conversion])
    ylim = np.array([0.75 * diesel_price, 1.25 * diesel_price])
    X, Y, Z = bst.plots.generate_contour_data(
        carbonyield_at_dieselprice_titer,
        xlim=xlim, ylim=ylim,
        args=(titer,),
        n = n,)  # numero de puntos a evaluar por en x  y y 
    # Guarda vectores y despues cargarlos 
    output_folder = 'results'
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Construct the full relative path
    output_path = os.path.join(output_folder, file)
    np.savez(output_path, array_a=X, array_b=Y, array_c = Z)


def plot_titer_carbonyield_dieselprice(file = 'default.npz'):
    output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, file)
    
    # Recupera valores
    npzfile  = np.load( output_path )
    X = npzfile['array_a']
    Y = npzfile['array_b']
    Z = npzfile['array_c']

    # xlabel = r"$\mathrm{Carbon}\,\mathrm{conversion} \, [\%]$"
    # ylabel = r"$\mathrm{Diesel}\,\mathrm{price}$" +  "\n" +  r"$[\mathrm{USD} \cdot \mathrm{kg}^{-1}]$"

    xlabel = "Carbon conversion $[\%]$" #r"Conversion de carbono %[\%]$"
    ylabel = r"Diesel price" +  "\n" +  r"$[\mathrm{USD} \cdot \mathrm{kg}^{-1}]$"


    units = r'c'
    # titles = [r"$\mathrm{PC}_{\max}$" + "\n" + r"$6250\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$", r"$\mathrm{PC}_{\text{mean}}$" + "\n" + r"$4166\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$", r"$\mathrm{PC}_{\min}$" + "\n" + r"$2500\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$"]
    titles = [ "\n" + r"$6250\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$",
            r"$\textrm{Processing Capacities}$" + "\n" + r"$4166\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$", 
            "\n" + r"$2500\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$"]
    # titles = [ "\n" + r"$6250\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$",
    #         r"$\textrm{Capacidades de procesamiento}$" + "\n" + r"$4166\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$", 
    #         "\n" + r"$2500\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$"]

    titles.reverse()
    xticks = X[0,:] #[30, 55, 70 ]
    yticks = Y[:,0] #[0.16, 0.3, 0.40]


    metric_bar = bst.plots.MetricBar(
    r'MSP', r'USD', plt.cm.get_cmap('viridis'),
    bst.plots.rounded_tickmarks_from_data(Z, 5, 1, expand=0, p=0.5), 50, 0  # Levels, nivel de redondeo 
    )
    file = 'Figure32_Carbonoprice.svg'#'Figure10.eps' #'IRR_DiselActivatedPrice_CarbonConversion.eps'
    cbar_ax_title = '$\mathrm{IRR}\,[\%]$'
    line_lvls =  [ 10.449,  12.898, 16.163, 20.245, 24.327 , 28,  32,  39.02, 43.102]
    fig, axes, cps, cbar, other_axes = _plot_contour_single_metric(
        X, Y, Z[:, :, None, :], xlabel, ylabel, xticks, yticks, metric_bar, 
        line_lvls , cbar_ax_title, file , 
        titles=titles, fillcolor=None,
        styleaxiskw=dict(xtick0=False), label=True) 
    # return fig, axes, cps, cbar, other_axes
    
    output_folder = 'figures'
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Construct the full relative path
    output_path = os.path.join(output_folder, file)

    fig.savefig(output_path, format='svg', dpi=1200, bbox_inches='tight', pad_inches=0.02)

# Falta modificar estos

def titer_carbonyield_temperature_FEDI(file = 'default.npz', n = 10):

    titer = np.array([2500,4166,6250])  # puntos 
    carbon_conversion = 55
    temperature = 550 +273.15
    xlim = np.array([0.75 * carbon_conversion, 1.25 * carbon_conversion])
    ylim = np.array ( [500 + 273.15  ,  800 + 273.15]) 
    X, Y, Z = bst.plots.generate_contour_data(
        FEDI_carbonyield_temperature_FEDI,
        xlim=xlim, ylim=ylim,
        args=(titer,),
        n = n,)  # numero de puntos a evaluar por en x  y y 
    
    # Guarda vectores y despues cargarlos 
    output_folder = 'results'
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Construct the full relative path
    output_path = os.path.join(output_folder, file)
    np.savez(output_path, array_a=X, array_b=Y, array_c = Z)


def plot_titer_carbonyield_temperature_FEDI(file = 'default.npz'):
    output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, file)
    
    # Recupera valores
    npzfile  = np.load( output_path )
    X = npzfile['array_a']
    Y = npzfile['array_b']
    Z = npzfile['array_c']

    # xlabel = r"$\mathrm{Carbon}\,\mathrm{conversion} \, [\%]$"
    # xlabel = "Conversión de carbono $[\%]$"
    xlabel = r"$\mathrm{Carbon}\,\mathrm{conversion} \, [\%]$"
    ylabel = r"Reaction temperature [K]"
    # ylabel = r"$\mathrm{Carbon\ activated\ price}\ [\mathrm{USD}\,\cdot\,\mathrm{kg}^{-1}]$"

    units = r'c'
    # titles = [r"$\mathrm{PC}_{\max}$" + "\n" + r"$6250\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$", r"$\mathrm{PC}_{\text{mean}}$" + "\n" + r"$4166\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$", r"$\mathrm{PC}_{\min}$" + "\n" + r"$2500\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$"]
    titles = [ "\n" + r"$6250\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$",
            r"$\textrm{Processing Capacities}$" + "\n" + r"$4166\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$", 
            "\n" + r"$2500\,\mathrm{kg}\,\cdot\mathrm{h}^{-1}$"]

    titles.reverse()
    xticks = X[0,:] #[30, 55, 70 ]
    yticks = Y[:,0] #[0.16, 0.3, 0.40]
    metric_bar = bst.plots.MetricBar(
    r'MSP', r'USD', plt.cm.get_cmap('magma_r'),
    bst.plots.rounded_tickmarks_from_data(Z, 5, 1, expand=0, p=0.5), 100, 0  # Levels, nivel de redondeo
    )
    cbar_ax_title = r'$\mathrm{FEDI}$'  # + "\n" + r"$\,[\textrm{damage radius} \cdot \mathrm{m}]$"
    file = 'Figure12.eps'#'Figure11.eps' #'FEDI_Temperature_CarbonConversion.eps'
    line_lvls = [200, 300, 400, 500, 600 , 700 , 800, 900, 1000, 1100]

    fig, axes, cps, cbar, other_axes = _plot_contour_single_metric(
        X, Y, Z[:, :, None, :], xlabel, ylabel, xticks, yticks, metric_bar, 
        line_lvls , cbar_ax_title, file , 
        titles=titles, fillcolor=None,
        styleaxiskw=dict(xtick0=False), label=True) 

    # Cambiar solo axes
    for ax in axes[:, :-1].ravel():           # evita la columna del colorbar
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))     # 5–6 ticks máx
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))  # sin decimales
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))     # 5–6 ticks máx
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))  # sin decimales
            ax.tick_params(axis='x', pad=2) 

    output_folder = 'figures'
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Construct the full relative path
    output_path = os.path.join(output_folder, file)

    fig.savefig(output_path, format='eps', dpi=1200, bbox_inches='tight', pad_inches=0.02)








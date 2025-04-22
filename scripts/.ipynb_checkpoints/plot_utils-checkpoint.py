import matplotlib.pyplot as plt
from cartopy.crs import PlateCarree
from scipy.stats import gaussian_kde
import matplotlib.colors as colors
import pandas as pd
import numpy as np
import cmocean.cm as cm

def set_plot(nrows = 1, ncols = 1, figsize = [7,7], font_size = 12.5):

    """Generates subplots with PlateCarree projection.
    Suitable for either single plots or single-row(column) plot disposition
    Input: 
        nrows (int) : number of rows
        ncols (int) : number of columns
        figsize (list) : figure size (default to [7,7])
        font_size (float) : fontsize of coordinate tick (default to 12.5)
    Output:
        fig, ax : empty figure with subplots ready to be filled"""

    # Generate subplot with PlateCarree projection
    fig, ax = plt.subplots(subplot_kw=dict(projection=PlateCarree()), figsize=figsize, nrows=nrows, ncols=ncols,
                           constrained_layout = True)

    # Checks if a single plot wants to be generated
    if nrows != 1 or ncols != 1:
        # For each plot, gridlines and ticks are disposed
        for a in ax:
            a.coastlines(linewidths=0.5, zorder = 1000)
            a.tick_params(axis="both", labelsize=font_size)
            gl = a.gridlines(linewidth=1, color='gray', alpha=0.5, linestyle='--', zorder = 2, draw_labels = True)
            gl.top_labels = False
            gl.right_labels = False
            if a != ax[0]:
                gl.left_labels = False
    else:
            ax.coastlines(linewidths=0.5, zorder = 1000)
            ax.tick_params(axis="both", labelsize=font_size)
            gl = ax.gridlines(linewidth=1, color='gray', alpha=0.5, linestyle='--', zorder = 2, draw_labels = True)
            gl.top_labels = False
            gl.right_labels = False

    return fig, ax

def write_shelf_line(ax, bathy, ref_depth = -1500):

    try:
        for a in ax:
                a.contour(bathy.longitude, bathy.latitude, bathy, [ref_depth], 
                          colors = "red", linestyles = "solid")
    except:
        ax.contour(bathy.longitude, bathy.latitude, bathy, [ref_depth], 
                   colors = "red", linestyle = "solid")

def map_colors_to_labels(train_data):
    
    colors_list = ["yellow", "pink", "purple", "brown", "grey", "orange", "olive"]

    num_clusters = len(train_data["labels"].unique())
    cols = {k : v for k in range(num_clusters) for v in np.zeros(num_clusters, dtype = str)}       

    filament_cluster = train_data["labels"].iloc[np.argmax(train_data["chl"])]
    cold_cluster = train_data["labels"].iloc[np.argmin(train_data["sst"])]
    warm_cluster = train_data["labels"].iloc[np.argmax(train_data["sst"])]

    
    i = 0
    for c in range(num_clusters):
        if c == filament_cluster:
            cols[c] = "green"
        elif c == cold_cluster:
            cols[c] = "blue"
        elif c == warm_cluster:
            cols[c] = "red"
        else:
            cols[c] = colors_list[i]
            i += 1
    return {v : k for k, v in cols.items()}

def plot_clustered_space(train_data, pipeline, fontsize = 13):
    
    fig, axs = plt.subplot_mosaic("""AAAD
                                     BBBC
                                     BBBC
                                     BBBC""")

    cols = map_colors_to_labels(train_data)

    cmap = colors.ListedColormap(cols)
    
    # CENTER B PLOT
    axs["B"].set_xlabel("$\Delta$Chl [mg m$^{-3}$]", fontsize = fontsize)
    axs["B"].set_ylabel("$\Delta$SST [°C]", fontsize = fontsize)
    
    axs["B"].plot(train_data["chl"], train_data["sst"], ".", 
                  markersize = 3, alpha = .7, zorder = -2,
                 color = "dimgray")
    
    
    yabs_max = abs(max(axs["B"].get_ylim(), key=abs))
    xabs_max = abs(max(axs["B"].get_xlim(), key=abs))
    
    axs["B"].set_ylim(ymin=-yabs_max, ymax=yabs_max)
    axs["B"].set_xlim(xmin=-xabs_max, xmax=xabs_max)
    
    axs["B"].hlines(0, xmin = -xabs_max, xmax = xabs_max, color = "dimgray", linestyle = "dashed", linewidth = .5)
    axs["B"].vlines(0, ymin = -yabs_max, ymax = yabs_max, color = "dimgray", linestyle = "dashed", linewidth = .5)
    h = 0.02  # step size of the mesh
    
    xx, yy = np.meshgrid(np.arange(-xabs_max, xabs_max, h), np.arange(-yabs_max, yabs_max, h))
    Z = pipeline.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    axs["B"].contour(xx, yy, Z, alpha=0.5, colors = "black", linewidths = .5)
    axs["B"].imshow(
            Z,
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=cmap,
            aspect="auto",
            origin="lower",
            zorder = -1, alpha = 0.5
        )
    
    centroids = pipeline["km"].cluster_centers_
    
    for i, c in enumerate(centroids):
        c = pipeline["scaler"].inverse_transform(c.reshape(1,-1)).flatten()
        axs["B"].text(c[0], c[1], str(i), ha = "center", va = "center")
    
    chl_x = np.linspace(train_data["chl"].min(), train_data["chl"].max(), 50)
    sst_x = np.linspace(train_data["sst"].min(), train_data["sst"].max(), 50)
    chl_kde = gaussian_kde(train_data["chl"])
    sst_kde = gaussian_kde(train_data["sst"])
    
    # UPPER CHL KERNEL PLOT
    axs["A"].set_xticks([])
    axs["A"].set_xlim(xmin=-xabs_max, xmax=xabs_max)
    axs["A"].hist(train_data["chl"], alpha = .5, density = True, bins = 50)
    axs["A"].plot(chl_x, chl_kde(chl_x), color = "black")
    axs["A"].vlines(0, ymin = 0, ymax = max(axs["A"].get_ylim(), key=abs), 
                    color = "dimgray", linestyle = "dashed", linewidth = .5)
    
    
    # RIGHT SST KERNEL PLOT
    axs["C"].set_yticks([])
    axs["C"].set_ylim(ymin=-yabs_max, ymax=yabs_max)
    axs["C"].plot(sst_kde(sst_x), sst_x, color = "black")
    axs["C"].hist(train_data["sst"], alpha = .5, density = True, bins = 50, orientation = "horizontal")
    axs["C"].hlines(0, xmin = 0, xmax = max(axs["C"].get_xlim(), key=abs), 
                    color = "dimgray", linestyle = "dashed", linewidth = .5)
    
    
    # UPPER RIGHT POPULATION PLOT
    axs["D"].set_ylim([0,50])
    axs["D"].set_yticks([0, 25, 50])
    counts = pd.DataFrame({"labels" : train_data["labels"]}).value_counts().sort_index()
    perc_counts = counts / len(train_data["labels"]) * 100
    
    axs["D"].bar(range(len(centroids)), perc_counts, color = cols, edgecolor = "black", alpha = 0.7)
    axs["D"].set_xticks(range(0, len(perc_counts)))
    axs["D"].xaxis.set_ticks_position('none') 
    axs["D"].tick_params(axis='x', which='major', pad=-2)
    axs["D"].yaxis.tick_right()
    axs["D"].set_yticklabels(["0%", "25%", "50%"])

    return fig, axs

def xy_rectangle(coords):

    """coords (list) : list of coordinates in the form [lon_min, lon_max, lat_min, lat_max]"""
    return [coords[0], coords[1], coords[1], coords[0], coords[0]], [coords[2]]*2 + [coords[3]]*2 + [coords[2]] 

def plot_bathymetry_profile(bathy, ref_depth, alpha = 0.5):
    fig, ax = set_plot()
    
    c = ax.contourf(bathy.longitude, bathy.latitude, bathy, 
                   cmap = cm.deep_r, vmax = 0, vmin = -5000, 
                   levels = np.arange(-5000, 500, 500), extend = "min", alpha = alpha)
    
    #ax.contour(bathy.longitude, bathy.latitude, bathy, 
    #               colors = "black", linewidths = .3, 
    #               vmax = 0, vmin = -5000, 
    #               levels = np.arange(-5000, 500, 500), extend = "min")
    
    write_shelf_line(ax, bathy, ref_depth = ref_depth)
    
    cb = plt.colorbar(c, ax = ax, shrink = .75)
    cb.set_label(label='Depth [m]', fontsize = 12)

    return ax
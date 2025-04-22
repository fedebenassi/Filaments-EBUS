import xarray as xr
import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.optimize import curve_fit
from scipy.stats import kstest


def find_streamer_cluster(fitted_train_data):
    """Finds the Streamer cluster (associated to greatest chlorophyll anomaly)
    Input:
        fitted_train_data (pandas.DataFrame) : fitted train dataset with "labels" column
    Output:
        cluster_number (int) : streamers cluster number"""
    
    return fitted_train_data["labels"].iloc[np.argmax(fitted_train_data["chl"])]

def compute_mass_content(data, eudepth, binary_masks):
    """Computes the mass content of "data" in regions delimited by binary masks [in Tg].
    Input:
        data (xarray.DataArray) : concentration [mg m^-3] time series (chlorophyll, diatoms, etc.)
        eudepth (xarray.DataArray) : euphotic depth [m] time series
    Output:
        content_values (pandas.DataFrame) : returns time series of mass content [Tg]"""
    time = binary_masks.time

    content = []
    for t in time:
        d, ed = data.sel(time = t).values, eudepth.sel(time = t).values
        m = binary_masks.sel(time = t).values
        
        content.append(np.nansum(d[m] * ed[m])  * (4000)**2 * 1e-15)

    content_values = pd.DataFrame(data = content, index = time, columns = ["content_Tg"])

    return content_values

def weibull(x, beta, eta):
    return (beta / eta) * ((x / eta) ** (beta - 1)) * np.exp( - (x / eta) ** beta) 

def fit_weibull(bin_edges, hist):
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    fit_params, _ = curve_fit(weibull, bin_centers, hist)
    return fit_params

def cum_weibull(x, beta, eta):
    return 1 - np.exp( - (x / eta) ** beta )

def compute_export_values(mass_content):

    # Value of absolute yearly export
    export = mass_content.sum() / (len(np.unique(mass_content.index.year)))

    hist, bin_edges = np.histogram(mass_content, bins = 100, density = True)
    beta, eta = fit_weibull(bin_edges, hist)

    # Error computation
    mu = eta * gamma(1 + 1/beta)
    sigma = eta * np.sqrt(gamma(1 + 2 / beta) - gamma(1 + 1 / beta)**2)
    relative_error = relative_error = kstest(mass_content.values.flatten(),
                    cum_weibull, args = [beta, eta]).statistic

    abs_error = relative_error * export
    return export.values, abs_error.values
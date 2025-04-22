import xarray as xr
import numpy as np

import os
import joblib
from scripts.preprocessing import read_region_input_files, crop_region

from tqdm import tqdm

from omegaconf import OmegaConf

from dask.distributed import Client

from sklearn.utils import resample
import pandas as pd

def compute_export_values_bootstrap(mass_content, n_iterations = 10000):

    num_years = len(np.unique(mass_content.time.dt.year))    
    
    # Container for bootstrap yearly exports
    bootstrap_yearly_exports = []
    
    # Perform bootstrap sampling
    for _ in range(n_iterations):
        # Resample with replacement
        boot_sample = resample(mass_content.values, replace=True)
        # Compute the average yearly export for the bootstrap sample
        boot_yearly_export = boot_sample.sum() / num_years
        
        bootstrap_yearly_exports.append(boot_yearly_export)
    
    # Calculate the mean of the bootstrap yearly exports
    bootstrap_mean = np.mean(bootstrap_yearly_exports)
    
    # Calculate the 95% confidence interval (2 stds)
    bootstrap_std = np.std(bootstrap_yearly_exports) * 2
    
    return bootstrap_mean, bootstrap_std

if __name__ == '__main__':

    # Read coordinate list for boxes
    print('Reading regions')
    regions = read_region_input_files('regions.input')

    if not os.path.exists('outputs/export_timeseries/'):
        print('export_timeseries folder is missing. Creating it.')
        os.makedirs('outputs/export_timeseries/')

    # Read global configurations (number of clusters etc)
    cfg = OmegaConf.load('../glob_config.yaml')

    c_to_chl_ratio = cfg.c_to_chl_ratio

    boxes = [box for boxes in regions.values() for box in boxes] 

    df = pd.DataFrame(columns = ['region', 'export', 'error'], 
                      index = np.arange(1, len(boxes) + 1))

    for i, box in enumerate(boxes):

        i += 1

        time_series = xr.open_dataarray(f'outputs/export_timeseries/box_{i}.nc') * c_to_chl_ratio

        export, error = compute_export_values_bootstrap(time_series)

        df.loc[i] = pd.Series({'region' : i, 'export' : export, 'error' : error})
                  
    df.to_csv('outputs/carbon_exports.csv')

        


        


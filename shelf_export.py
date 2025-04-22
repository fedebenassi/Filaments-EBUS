import xarray as xr
import numpy as np

import os
import joblib
from scripts.preprocessing import read_region_input_files, crop_region

from tqdm import tqdm

from omegaconf import OmegaConf

from dask.distributed import Client


if __name__ == '__main__':

    client = Client()
    print(client.dashboard_link)
    # Read coordinate list for boxes
    print('Reading regions')
    regions = read_region_input_files('regions.input')

    if not os.path.exists('outputs/shelf_timeseries/'):
        print('shelf_timeseries folder is missing. Creating it.')
        os.makedirs('outputs/shelf_timeseries/')

    # Read global configurations (number of clusters etc)
    cfg = OmegaConf.load('../glob_config.yaml')

    boxes = [box for boxes in regions.values() for box in boxes] 

    for i, box in enumerate(boxes):
        i += 1

        content = xr.open_dataarray(f'outputs/shelf_content/box_{i}.nc').chunk({'time' : 1})

        time_series = content.sum(dim=['longitude', 'latitude'])

        time_series.to_netcdf(f'outputs/shelf_timeseries/box_{i}.nc')

    client.close()

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

    if not os.path.exists('outputs/shelf_content/'):
        print('shelf_content folder is missing. Creating it.')
        os.makedirs('outputs/shelf_content/')

    # Read global configurations (number of clusters etc)
    cfg = OmegaConf.load('../glob_config.yaml')

    boxes = [box for boxes in regions.values() for box in boxes] 

    for i, box in enumerate(boxes):
        i += 1

        #masks = xr.open_dataarray(f'outputs/streamers_masks/box_{i}.nc').chunk({'time' : 1})
        lons, lats = slice(box[0], box[1]), slice(box[2], box[3])
        shelf_mask = xr.open_dataarray(cfg.bathy_path).sel(longitude=lons, latitude=lats) >= cfg.ref_depth

        chl = crop_region(cfg.chl_path, box) * shelf_mask
        eudepth = crop_region(cfg.eudepth_path, box) * shelf_mask

        biomass_content = chl * eudepth * (4000)**2 * 1e-15
        #time_series = biomass_content.sum(dim=['longitude', 'latitude']) * (4000)**2 * 1e-15

        biomass_content.to_netcdf(f'outputs/shelf_content/box_{i}.nc')

    client.close()

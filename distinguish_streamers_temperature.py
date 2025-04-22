import dask
from dask import delayed
import dask.array as da
from dask.distributed import Client
import numpy as np
import pandas as pd
import xarray as xr
from scripts.clustering_utils import create_labels_map
import os
import joblib
from omegaconf import OmegaConf
from scripts.preprocessing import read_region_input_files, _compute_off_shelf_anomalies
from scripts.clustering_utils import find_streamer_cluster
from tqdm import tqdm

def generate_mask_for_single_time_step(delta_slice, model, streamers_cluster):
    """
    Generate binary masks for a single time step slice of the delta values of chlorophyll and SST.
    
    Input:
        delta_slice (xarray.Dataset): Dataset slice for a single time step (delta chlorophyll and delta SST)
        model (sklearn model): A pre-trained scikit-learn model for binary classification
        
    Output:
        mask (xarray.DataArray): 2D binary mask for this time step
    """
    
    # Get the shape (latitude, longitude) of the input slice
    shape = (delta_slice.sizes['latitude'], delta_slice.sizes['longitude'])
    
    
    # Create a nan mask to avoid invalid data points
    nan_mask = (~np.isnan(delta_slice['chl'].data)) & (~np.isnan(delta_slice['sst'].data))

    # Initialize an empty array for the mask, filled with NaN
    labels_mask = np.zeros(nan_mask.shape, dtype = bool)

    # Stack the chlorophyll and SST data for input to the model (in Dask)
    features = delta_slice.to_dataframe()[['chl', 'sst']].dropna()
    
    # Check if we have any valid data points
    if nan_mask.any():
        # Compute the predictions
        labels = model.predict(features.astype('double')) == streamers_cluster
        
        # Reshape the labels to match the spatial dimensions (latitude, longitude)        
        labels_mask[nan_mask] = labels
    
    return xr.DataArray(labels_mask.reshape(1, shape[0], shape[1]), dims=['time', 'latitude', 'longitude'], coords={'time' : delta_slice['time'],'latitude': delta_slice['latitude'], 'longitude': delta_slice['longitude']})


def generate_binary_masks_in_parallel(deltas, model, streamers_cluster):
    """
    Generate binary masks in parallel for each time step using Dask and map_blocks.
    
    Input:
        deltas (xarray.Dataset): Merged dataset containing delta chlorophyll and delta SST for all time steps
        model (sklearn model): Pre-trained model for prediction
        
    Output:
        masks (xarray.DataArray): Binary masks for each time step, applied in parallel
    """

    deltas = deltas.chunk({'time' : 1})

    # Define a template for the output masks. The output has the same structure as deltas, but only one variable (the binary mask)
    mask_template = deltas['chl'] * np.nan

    # Use xr.map_blocks to apply the mask generation function in parallel across time steps
    masks = deltas.map_blocks(generate_mask_for_single_time_step, args = [model, streamers_cluster], template = mask_template)
    
    return masks



def prepare_delta_timeseries(box, chl_path, sst_path, bathy_path, ref_depth):
    """
    Prepares delta time series for chlorophyll and SST data, ensuring parallel processing
    by persisting both datasets immediately after loading.
    
    Input:
        box (tuple): Bounding box coordinates (lon_min, lon_max, lat_min, lat_max)
        chl_path (str): Path to the chlorophyll time series
        sst_path (str): Path to the SST time series
        bathy_path (str): Path to the bathymetry data
        ref_depth (float): Reference depth for off-shelf anomaly computation
        
    Output:
        delta_chl (xarray.DataArray): Anomaly data for chlorophyll
        delta_sst (xarray.DataArray): Anomaly data for SST
    """
    
    # Define the geographic slice
    lons, lats = slice(box[0], box[1]), slice(box[2], box[3])
    
    # Load and slice chlorophyll and SST datasets
    chl = _open_time_series(chl_path).sel(longitude=lons, latitude=lats).persist()
    sst = _open_time_series(sst_path).sel(longitude=lons, latitude=lats).persist()

    # Load bathymetry data
    bathy = xr.open_dataarray(bathy_path).sel(longitude=lons, latitude=lats).persist()

    # Compute anomalies based on bathymetry
    delta_chl = _compute_off_shelf_anomalies(chl, bathy, ref_depth)
    delta_sst = _compute_off_shelf_anomalies(sst, bathy, ref_depth)

    return delta_chl, delta_sst

def _open_time_series(path):
    return xr.open_zarr(path, chunks={'longitude': -1, 'latitude': -1})

# def _compute_off_shelf_anomalies(data, bathy, ref_depth):
#     data = data.where(bathy < ref_depth)
#     num_years = len(data.time) // 365
#     data = data.assign_coords({'dayofyear': ('time', np.tile(np.arange(0, 365), num_years))})
#     data = data.chunk({'time': 365, 'longitude': -1, 'latitude': -1})
#     data_clima = data.groupby('dayofyear').mean().chunk({'dayofyear': 365})
#     delta_data = data.groupby('dayofyear') - data_clima
#     return delta_data.drop_vars('dayofyear').transpose('time', ...).chunk({'time' : 1})


if __name__ == '__main__':

    # Reading the input regions with the boxes from the regions.input file
    print('Reading regions')
    regions = read_region_input_files('regions.input')
    print(f'Found regions: {regions.keys()}')

    # Global configurations are stored in the glob_config.yaml file
    cfg = OmegaConf.load('../glob_config.yaml')

    # Ensure output directory exists
    if not os.path.exists('outputs/streamers_masks/low_dsst'):
        print('streamers_masks/low_dsst folder is missing. Creating it.')
        os.makedirs('outputs/streamers_masks/low_dsst')
    # Ensure output directory exists
    if not os.path.exists('outputs/streamers_masks/high_dsst'):
        print('streamers_masks/high_dsst folder is missing. Creating it.')
        os.makedirs('outputs/streamers_masks/high_dsst')
    
    #client = Client()
    #print(client.dashboard_link)
    box_code = 0

    for region_label, boxes in regions.items():
        print(f'Loading training data, labels and pipeline for {region_label}')
        
        # Load trained dataset, labels, and pipeline
        train_data = pd.read_csv(f'outputs/train_data/{region_label}.csv')
        labels = np.load(f'outputs/labels/{region_label}.npy')
        pipeline = joblib.load(f'outputs/models/{region_label}.joblib')

        streamers_cluster = find_streamer_cluster(train_data, labels)
        
        del train_data
        del labels

        for box in tqdm(boxes):
            box_code += 1

            delta_chl, delta_sst = prepare_delta_timeseries(box, cfg.chl_path, cfg.sst_path, cfg.bathy_path, cfg.ref_depth)
            deltas = xr.merge([delta_chl, delta_sst]).persist()
            
            masks = generate_binary_masks_in_parallel(deltas, pipeline, streamers_cluster).compute()

            # Save output to NetCDF
            masks.to_netcdf(f'outputs/streamers_masks/box_{box_code}.nc')

    #client.close()


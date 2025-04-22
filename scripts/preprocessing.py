import xarray as xr
import numpy as np
import pandas as pd
import ast
import os

def read_region_input_files(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file {path} not found")
    with open(path, 'r') as file:
        data = file.read()
    return ast.literal_eval(data)

def _open_time_series(path):
    return xr.open_zarr(path, chunks={'longitude': -1, 'latitude': -1})

def _compute_off_shelf_anomalies(data, bathy, ref_depth):
    data = data.where(bathy < ref_depth)
    num_years = len(data.time) // 365
    data = data.assign_coords({'dayofyear': ('time', np.tile(np.arange(0, 365), num_years))})
    data = data.chunk({'time': 365, 'longitude': -1, 'latitude': -1})
    data_clima = data.groupby('dayofyear').mean().chunk({'dayofyear': 365})
    delta_data = data.groupby('dayofyear') - data_clima
    return delta_data.drop('dayofyear')

def _create_time_mask(start_date, end_date):
    time_series = pd.date_range(start_date, end_date, freq='1D')
    return time_series[~((time_series.day == 29) & (time_series.month == 2))]

def crop_region(data_path, box, engine = 'zarr'):
    lons, lats = slice(box[0], box[1]), slice(box[2], box[3])
    data = xr.open_dataarray(data_path, engine = engine).sel(longitude=lons, latitude=lats)
    return data.chunk({'time' : 1, 'longitude': -1, 'latitude': -1})

def crop_square(data, box, engine = 'zarr'):
    lons, lats = slice(box[0], box[1]), slice(box[2], box[3])
    data = data.sel(longitude=lons, latitude=lats)
    return data.values

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

def prepare_training_data(box, chl_path, sst_path, bathy_path, ref_depth, start_date, end_date):
    
    delta_chl, delta_sst = prepare_delta_timeseries(box, chl_path, sst_path, bathy_path, ref_depth)
    
    mask = _create_time_mask(start_date, end_date)

    delta_chl = delta_chl.chl.sel(time=mask)
    delta_sst = delta_sst.sst.sel(time=mask)
    
    deltas = xr.merge([delta_chl, delta_sst])
    
    deltas_df = deltas[['chl', 'sst']].to_dask_dataframe()

    return deltas_df[['chl', 'sst']].dropna()

def obtain_boxes_grouping(regions):
    list_lengths = [len(group) for group in regions.values()]

    # Step 2: Generate a list of numbers from 1 to the total number of elements
    total_elements = sum(list_lengths)
    number_list = list(range(1, total_elements + 1))

    # Step 3: Group the numbers according to the lengths of the values
    grouped_numbers = []
    start = 0
    for length in list_lengths:
        grouped_numbers.append(number_list[start:start + length])
        start += length
    return grouped_numbers

def compute_deltachl(box, chl_path, bathy_path, ref_depth):
    # Define the geographic slice
    lons, lats = slice(box[0], box[1]), slice(box[2], box[3])

    # Load bathymetry data
    bathy = xr.open_dataarray(bathy_path).sel(longitude=lons, latitude=lats)

    chl = _open_time_series(chl_path).sel(longitude=lons, latitude=lats)

    # Compute anomalies based on bathymetry
    delta_chl = _compute_off_shelf_anomalies(chl, bathy, ref_depth)

    return delta_chl



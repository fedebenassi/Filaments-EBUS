import xarray as xr
import numpy as np
import pandas as pd
import ast

def read_region_input_files(path):
    # Read the dictionary from the file
    with open(path, 'r') as file:
        data = file.read()
    # Convert the string representation of the dictionary to an actual dictionary
    return ast.literal_eval(data)

def _open_time_series(path):
    return xr.open_zarr(path, 
                            chunks = {'longitude' : -1, 'latitude' : -1})

def crop_regions(box, chl_path,
                sst_path,
                bathy_path,
                eudepth_path):
    """Opens global files and crops regions based on input lon and lat.
    Input:
        lon (slice) : boundary longitudes (the smaller goes first)
        lat (slice) : boundary latitudes (the smaller goes first)
        chl_path (str) : path to chlorophyll global file
        sst_path (str) : path to sea surface temperature global file
        bathy_path (str) : path to bathymetry global file
        eudepth_path (str) : path to euphotic depth global file
    Output:
        chl (xarray.DataArray) : cropped chlorophyll time series
        sst (xarray.DataArray) : cropped sea surface temperature time series
        eudepth (xarray.DataArray) : cropped euphotic depth time series
        bathy (xarray.DataArray) : cropped bathymetry"""
    
    lons, lats = slice(box[0], box[1]), slice(box[2], box[3])
    # Open chlorophyll and select lons and lats
    chl = _open_time_series(chl_path)\
        .sel(longitude = lons, latitude = lats)


    # Open sea surface temperature and select lons and lats
    sst = _open_time_series(sst_path)\
        .sel(longitude = lons, latitude = lats)\

    # Open sea surface temperature and select lons and lats
    eudepth = _open_time_series(eudepth_path)\
        .sel(longitude = lons, latitude = lats)
    
    # Open bathymetry and select lons and lats
    bathy = xr.open_dataarray(bathy_path)\
        .sel(longitude = lons, latitude = lats)
    
    return chl, sst, eudepth, bathy

def _mask_shelf(data, bathy, ref_depth):
    """Creates a shelf mask based on reference depth.
    Input: 
        data (np.ndarray) : series of 2D data on which the masking is performed
        bathy (xr.DataArray) : reference bathymetry
        ref_depth (int) : reference depth to define shelf and open ocean
    Output:
        off_shelf_data (np.ndarray) : time series of masked 2D data"""
    # Create shelf mask based on reference depth
    shelf_mask = bathy.values > ref_depth

    # Shelf mask is expanded to be used with the other data
    #expanded_mask = np.expand_dims(shelf_mask, axis=0)
    #boolean_mask = np.tile(expanded_mask, (data.shape[0], 1, 1))
    boolean_mask = np.tile(shelf_mask, (data.shape[0], 1, 1))

    # Create a copy of the time series
    #off_shelf_data = np.copy(data)

    # Set values inside the mask to NaN
    data[boolean_mask] = np.nan
    
    return data


def _generate_off_shelf_data(data, bathy, ref_depth):

    """Generates xarray.DataArray of shelf-masked data. Points with associated
    depth (from bathymetry) greater than a reference value are set to numpy.nan
    
    Input:
        data (xarray.DataArray) : dataset (time series) on which masking is performed
        bathymetry (xarray.DataArray) : reference bathymetry DataArray
        ref_depth (int, default = -1500) : reference depth
    Output:
        off_shelf_data (xr.DataArray) : masked dataset
    """

    
    # off_shelf_data = xr.DataArray(data = _mask_shelf(data.values, bathy, ref_depth),
    #                         dims = ["time", "latitude", "longitude"],
    #                         coords=dict(time=pd.to_datetime(data.time),
    #                                     latitude= data.latitude,
    #                                     longitude= data.longitude))

    off_shelf_data = data.where(bathy.values > ref_depth)

    return off_shelf_data



def _compute_delta(data):

    """Computes time anomaly by subtracting the computed climatolodgy.
    Input:
        data (xarray.DataArray) : dataset on which anomalies are computed
    Output:
        delta_data (xarray.DataArray) : computed anomalies"""
    
    data_clima = data.groupby("time.dayofyear").mean(dim = "time")

    delta_data = data.groupby("time.dayofyear") - data_clima
    
    return delta_data.drop_vars("dayofyear")

def _compute_off_shelf_anomalies(data, bathy, ref_depth):

    data = data.where(bathy > ref_depth)

    num_years = len(data.time) // 365

    data = data.assign_coords({'dayofyear' : ('time', np.tile(np.arange(0, 365), num_years))})   
    
    data = data.chunk({'time' : 365, 'longitude' : -1, 'latitude' : -1})

    data_clima = data.groupby('dayofyear').mean().chunk({'dayofyear' : 365})

    delta_data = data.groupby('dayofyear') - data_clima

    return delta_data.drop('dayofyear')

def prepare_training_data(box, chl_path,
                            sst_path,
                            bathy_path,
                            eudepth_path, ref_depth):

    lons, lats = slice(box[0], box[1]), slice(box[2], box[3])

    chl = _open_time_series(chl_path).sel(longitude = lons, latitude = lats)
    sst = _open_time_series(sst_path).sel(longitude = lons, latitude = lats)

    bathy = xr.open_dataarray(bathy_path).sel(longitude = lons, latitude = lats).persist()

    delta_chl = _compute_off_shelf_anomalies(chl, bathy, ref_depth)

    delta_sst = _compute_off_shelf_anomalies(sst, bathy, ref_depth)

    mask = pd.date_range('01-01-2003', '31-12-2003', freq = '1D')

    #deltas = 
    delta_chl = delta_chl.chl.sel(time = mask).to_dask_dataframe()
    delta_sst = delta_sst.sst.sel(time = mask).to_dask_dataframe()

    deltas = delta_chl.merge(delta_sst)[['chl', 'sst']]

    return deltas



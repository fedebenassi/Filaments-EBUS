import xarray as xr
import numpy as np
import pandas as pd

def crop_regions(lon, lat, chl_path = "./data/processed/averaged/chl_8D.nc",
                sst_path = "./data/processed/averaged/sst_8D.nc",
                bathy_path = "./data/processed/bathymetry.nc",
                eudepth_path = "./data/processed/averaged/eudepth_8D_occci.nc"):
    """Opens global files and crops regions based on input lon and lat.
    Input:
        lon (slice) : boundary longitudes (the smaller goes first)
        lat (slice) : boundary latitudes (the smaller goes first)
        chl_path (str, optional) : path to chlorophyll global file
        sst_path (str, optional) : path to sea surface temperature global file
        bathy_path (str, optional) : path to bathymetry global file
        eudepth_path (str, optional) : path to euphotic depth global file
    Output:
        chl (xarray.DataArray) : cropped chlorophyll time series
        sst (xarray.DataArray) : cropped sea surface temperature time series
        eudepth (xarray.DataArray) : cropped euphotic depth time series
        bathy (xarray.DataArray) : cropped bathymetry"""

    # Open chlorophyll and select lons and lats
    chl = xr.open_dataarray(chl_path)\
         .sel(longitude = lon, latitude = lat)


    # Open sea surface temperature and select lons and lats
    sst = xr.open_dataarray(sst_path)\
             .sel(longitude = lon, latitude = lat)\

    # Open sea surface temperature and select lons and lats
    eudepth = xr.open_dataarray(eudepth_path)\
             .sel(longitude = lon, latitude = lat)
    
    # Open bathymetry and select lons and lats
    bathy = xr.open_dataarray(bathy_path)\
             .sel(longitude = lon, latitude = lat)
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


def generate_off_shelf_data(data, bathy, ref_depth = -1500):

    """Generates xarray.DataArray of shelf-masked data. Points with associated
    depth (from bathymetry) greater than a reference value are set to numpy.nan
    
    Input:
        data (xarray.DataArray) : dataset (time series) on which masking is performed
        bathymetry (xarray.DataArray) : reference bathymetry DataArray
        ref_depth (int, default = -1500) : reference depth
    Output:
        off_shelf_data (xr.DataArray) : masked dataset
    """

    
    off_shelf_data = xr.DataArray(data = _mask_shelf(data.values, bathy, ref_depth),
                            dims = ["time", "latitude", "longitude"],
                            coords=dict(time=pd.to_datetime(data.time),
                                        latitude= data.latitude,
                                        longitude= data.longitude))
    return off_shelf_data



def compute_delta(data):

    """Computes time anomaly by subtracting the computed climatolodgy.
    Input:
        data (xarray.DataArray) : dataset on which anomalies are computed
    Output:
        delta_data (xarray.DataArray) : computed anomalies"""
    
    data_clima = data.groupby("time.dayofyear").mean(dim = "time")

    delta_data = data.groupby("time.dayofyear") - data_clima
    
    return delta_data.drop_vars("dayofyear")


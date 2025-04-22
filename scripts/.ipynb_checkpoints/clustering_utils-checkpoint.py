import numpy as np
import xarray as xr
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from processing import * 

def prepare_training_data(delta_chl, delta_sst, time_range = [2003,2004]):

    """Prepares data for clustering fitting.
    Input:
        delta_chl (xarray.DataArray) : chlorophyll anomaly
        delta_sst (xarray.DataArray) : sea surface temperature anomaly
        time_range (list, default = [2003,2004]) : year range to consider in training dataset
    Output:
        data (pandas.DataFrame) : dataframe with "chl" and "sst" column ready for training
        """

    
    mask =  (delta_chl.time.dt.year.isin([i for i in range(time_range[0], time_range[1])]))
    chl_subset = delta_chl.sel(time = mask).values.flatten()
    sst_subset = delta_sst.sel(time = mask).values.flatten()

    data = pd.DataFrame({"chl" : chl_subset, "sst" : sst_subset}).dropna()

    return data

def fit_clustering(train_data, n_clusters = 4):

    """Fits a StandardScaler and a K-means clustering to the train dataset in input.
    Input:
        train_data (pandas.DataFrame) : fitting dataset with delta chl and delta sst columns
        n_clusters (int, default to 4) : number of clusters for K-means fitting
    Output:
        train_data (pandas.DataFrame) : fitting dataset with an additional "labels" column
        pipeline (sklearn model): fitted model"""

    pipeline = Pipeline([("scaler", StandardScaler()),
                        ("km", KMeans(n_clusters = n_clusters, random_state = 123, n_init = 10))])
    
    labels = pipeline.fit_predict(train_data.astype("double"))

    train_data["labels"] = labels
    
    return train_data, pipeline

def prepare_single_day_data(delta_chl_day, delta_sst_day):

    """Prepares single-day images of chlorophyll and sea surface temperature anomalies for labels mapping.
    Input:
        delta_chl_day (xarray.DataArray) : single delta chlorophyll image from the time series
        delta_sst_day (xarray.DataArray) : single delta sea surface temperature image from the time series
    Output:
        data (pandas.DataFrame) : dataframe ready for being fed to the pipeline
        """
    
    data = pd.DataFrame({"chl" : delta_chl_day.values.flatten(), 
                         "sst" : delta_sst_day.values.flatten()}).dropna()
    return data

def create_labels_map(delta_chl_day, delta_sst_day, model):

    """Geographically maps the labels of each pixel.
    Input:
        delta_chl_day (xarray.DataArray) : single delta chlorophyll image from the time series
        delta_sst_day (xarray.DataArray) : single delta sea surface temperature image from the time series
        model (sklearn.model) : fitted pipeline
    Output:
        labels_map (np.array) : 2D matrix with labels projected on the map
        """
    
    data = prepare_single_day_data(delta_chl_day, delta_sst_day)

    if len(data) == 0:
        labels_map = np.empty(delta_chl_day.values.shape)
        labels_map[:,:] = np.nan
    
    else:

        labels = model.predict(data.astype("double"))
            
        nan_mask = ~(np.isnan(delta_chl_day.values)) & \
                    ~(np.isnan(delta_sst_day.values))
    
        
        labels_map = np.empty(nan_mask.shape)
        labels_map[:,:] = np.nan
        labels_map[nan_mask] = labels
        
    return labels_map


def generate_mask_time_series(delta_chl, delta_sst, 
                              model, start_year = 2004,
                              filter_streamers_cluster = True):

    """Generates the time series of the clustering masks, starting from a specified start_year.
    Input: 
        delta_chl (xarray.DataArray) : chlorophyll anomalies time series
        delta_sst (xarray.DataArray) : sea surface temperature anomalies time series
        model (sklearn.model) : fitted model
        start_year (optional, default to 2004) : starting year from which computing the masks
        filter_streamers_cluster(optional, default to True) : if true, generates a binary mask for streamers delimitation
    Output:
        masks (xarray.DataArray) : time series of clustering masks"""
    
    time = delta_chl.time[delta_chl.time.dt.year >= start_year]

    masks = []
    
    for t in time:

        c, s = delta_chl.sel(time = t), delta_sst.sel(time = t)

        labels_map = create_labels_map(c, s, model)
        masks.append(labels_map)


    coords = {'time': time,
          'latitude': delta_chl.latitude,
          'longitude': delta_chl.longitude}
    dims = ['time', 'latitude', 'longitude']
        
    masks = xr.DataArray(np.array(masks), coords=coords, dims=dims)
    
    return masks

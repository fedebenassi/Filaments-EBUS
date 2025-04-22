import numpy as np
import pandas as pd
import xarray as xr

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import os
import joblib
from scripts.processing import read_region_input_files

from tqdm import tqdm

from omegaconf import OmegaConf
from dask.distributed import Client

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

    return labels, pipeline


if __name__ == '__main__':
    # Read coordinate list for boxes
    print('Reading regions')
    regions = read_region_input_files('regions.input')

    # Read global configurations (number of clusters etc)
    cfg = OmegaConf.load('../glob_config.yaml')

    # The train_data folder is created in case it is not existing
    if not os.path.exists('outputs/models/'):
        print('models folder is missing. Creating it.')
        os.makedirs('outputs/models/')
    
    if not os.path.exists('outputs/labels/'):
        print('labels folder is missing. Creating it.')
        os.makedirs('outputs/labels/')

    for region_label in tqdm(regions.keys()):

        print(f'Loading data for {region_label} from outputs/train_data/{region_label}.csv')
        train_data = pd.read_csv(f'outputs/train_data/{region_label}.csv', index_col = 0)

        print('Fitting clustering algorithm')
        labels, pipeline = fit_clustering(train_data, n_clusters=cfg.n_clusters)
        print('Fitted! Saving model and labels.')
        np.save(f'outputs/labels/{region_label}.npy', labels)
        joblib.dump(pipeline, f'outputs/models/{region_label}.joblib')



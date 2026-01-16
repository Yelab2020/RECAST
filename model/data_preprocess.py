### Data Preprocessing

import os
import numpy as np
import pandas as pd
from collections import Counter

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import re

# Upsampling
def upsampling(X_train, Y_train, seed):
    ros = RandomOverSampler(random_state=seed)
    X_train, Y_train = ros.fit_resample(X_train, Y_train)
    return X_train, Y_train

# Downsampling
def downsampling(X_train, Y_train, seed):
    rds = RandomUnderSampler(random_state=seed)
    X_train, Y_train = rds.fit_resample(X_train, Y_train)
    return X_train, Y_train

# No sampling
def nosampling(X_train, Y_train):
    return X_train, Y_train

# SMOTE
def SMOTEsampling(X_train, Y_train, seed):
    # Check the number of samples in the minority class
    n_minority = np.min(np.bincount(Y_train.values.flatten().astype(int)))
    print(n_minority)
    
    # Dynamically set n_neighbors
    # Ensure it does not exceed the sample count limit
    n_neighbors = min(5, n_minority - 1)
    
    # At least 2 samples are required
    if n_minority > 1:
        sm = SMOTE(
            random_state=seed,
            k_neighbors=n_neighbors,
            sampling_strategy='auto'
        )
        return sm.fit_resample(X_train, Y_train)
    else:
        # Use RandomOverSampler if samples are too few
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=seed)
        return ros.fit_resample(X_train, Y_train)

def normalize_name(name):
    """Standardize column names: convert to uppercase and remove spaces, -, _, /, etc."""
    normalized = name.upper()
    normalized = re.sub(r'[-_()\s/]+', '', normalized)
    return normalized

def bulk_preprocess(exp_path, label_path, select_drug, save_path=None, test_size=0.2, sampling=None, seed=114514):
    data_bulk = pd.read_csv(exp_path, index_col=0)
    label_bulk = pd.read_csv(label_path, index_col=0, na_values=["", "NA"])

    # Unify drug name format
    select_drug = normalize_name(select_drug)
    label_bulk.columns = [normalize_name(col) for col in label_bulk.columns]

    # Select indices of rows with non-null values in label_bulk
    selected_idx = label_bulk.loc[:, select_drug].notnull()

    # Get row names with non-null values
    selected_row_names = label_bulk.index[selected_idx]

    # Extract data from data_bulk and label_bulk based on selected indices and sort by row names
    data_select = data_bulk.loc[selected_row_names]
    label_select = label_bulk.loc[selected_row_names, select_drug]

    print("raw:", data_select.shape)
    
    # Convert label categories to numerical representation (resistant: 0; sensitive: 1)
    le = LabelEncoder()
    label_select_binary = pd.DataFrame(le.fit_transform(label_select.values.reshape(-1, 1)), 
                                       index=label_select.index)

    # Check if saving is required
    if save_path is not None:
        data_path = os.path.join(save_path, 'bulk_exp_reindex.csv')
        label_path = os.path.join(save_path, 'bulk_label_binary.csv')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(data_path):
            data_select.to_csv(data_path, index=True, header=True)
        if not os.path.exists(label_path):
            label_select_binary.to_csv(label_path, index=True, header=True)

    # Split into train and test sets
    X_train, X_valid, Y_train, Y_valid = train_test_split(data_select,
                                                          label_select_binary,
                                                          test_size=0.2,
                                                          random_state=seed,
                                                          stratify=label_select_binary.values.ravel())

    # Sampling
    if sampling == "no":
        X_train, Y_train = nosampling(X_train, Y_train)
        print("nosampling:", X_train.shape)
    elif sampling == "up":
        X_train, Y_train = upsampling(X_train, Y_train, seed)
        print("upsampling:", X_train.shape)
    elif sampling == "down":
        X_train, Y_train = downsampling(X_train, Y_train, seed)
        print("downsampling:", X_train.shape)
    elif sampling == "SMOTE":
        X_train, Y_train = SMOTEsampling(X_train, Y_train, seed)
        print("SMOTE:", X_train.shape)

    if save_path is not None:
        tdata_path = os.path.join(save_path, f'bulk_train_exp_{sampling}.csv')
        tlabel_path = os.path.join(save_path, f'bulk_train_label_binary_{sampling}.csv')
        vdata_path = os.path.join(save_path, f'bulk_valid_exp_{sampling}.csv')
        vlabel_path = os.path.join(save_path, f'bulk_valid_label_binary_{sampling}.csv')
        
        if not os.path.exists(tdata_path):
            X_train.to_csv(tdata_path, index=True, header=True)
        if not os.path.exists(vdata_path):
            X_valid.to_csv(vdata_path, index=True, header=True)
        if not os.path.exists(tlabel_path):
            Y_train.to_csv(tlabel_path, index=True, header=True)
        if not os.path.exists(vlabel_path):
            Y_valid.to_csv(vlabel_path, index=True, header=True)

    return X_train, X_valid, Y_train, Y_valid

def bulk_preprocess_ndrug(exp_path, label, select_drug, save_path=None, test_size=0.2, sampling=None, seed=114514):
    data_bulk = pd.read_csv(exp_path, index_col=0)

    selected_idx = data_bulk.index.argsort().argsort() # label order

    # Extract data from data_bulk and label_bulk based on selected indices
    data_select = data_bulk
    label_select = pd.DataFrame(
        label[selected_idx],
        index=data_bulk.index,
        columns=[select_drug]
        )

    print("raw:", data_select.shape)
    
    # Convert label categories to numerical representation (resistant: 0; sensitive: 1)
    le = LabelEncoder()
    label_select_binary = pd.DataFrame(le.fit_transform(label_select.values.reshape(-1, 1)), 
                                       index=label_select.index)

    # Check if saving is required
    if save_path is not None:
        data_path = os.path.join(save_path, 'bulk_exp_reindex.csv')
        label_path = os.path.join(save_path, 'bulk_label_binary.csv')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(data_path):
            data_select.to_csv(data_path, index=True, header=True)
        if not os.path.exists(label_path):
            label_select_binary.to_csv(label_path, index=True, header=True)

    # Split into train and test sets
    X_train, X_valid, Y_train, Y_valid = train_test_split(data_select,
                                                          label_select_binary,
                                                          test_size=0.2,
                                                          random_state=seed,
                                                          stratify=label_select_binary.values.ravel())

    # Sampling
    if sampling == "no":
        X_train, Y_train = nosampling(X_train, Y_train)
        print("nosampling:", X_train.shape)
    elif sampling == "up":
        X_train, Y_train = upsampling(X_train, Y_train, seed)
        print("upsampling:", X_train.shape)
    elif sampling == "down":
        X_train, Y_train = downsampling(X_train, Y_train, seed)
        print("downsampling:", X_train.shape)
    elif sampling == "SMOTE":
        X_train, Y_train = SMOTEsampling(X_train, Y_train, seed)
        print("SMOTE:", X_train.shape)

    if save_path is not None:
        tdata_path = os.path.join(save_path, f'bulk_train_exp_{sampling}.csv')
        tlabel_path = os.path.join(save_path, f'bulk_train_label_binary_{sampling}.csv')
        vdata_path = os.path.join(save_path, f'bulk_valid_exp_{sampling}.csv')
        vlabel_path = os.path.join(save_path, f'bulk_valid_label_binary_{sampling}.csv')
        
        if not os.path.exists(tdata_path):
            X_train.to_csv(tdata_path, index=True, header=True)
        if not os.path.exists(vdata_path):
            X_valid.to_csv(vdata_path, index=True, header=True)
        if not os.path.exists(tlabel_path):
            Y_train.to_csv(tlabel_path, index=True, header=True)
        if not os.path.exists(vlabel_path):
            Y_valid.to_csv(vlabel_path, index=True, header=True)

    return X_train, X_valid, Y_train, Y_valid
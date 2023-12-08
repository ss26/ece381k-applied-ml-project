import os
import pandas as pd
import torch
import torchvision
from pathlib import Path
from tqdm import tqdm

# Installing necessary library for COCO format handling
# !pip install pycocotools

# Setting the path for the dataset and reading the training events CSV file
data_folder = Path("/kaggle/input/child-mind-institute-detect-sleep-states")
train_events = pd.read_csv(data_folder/"train_events.csv")
print(train_events.head())

# Identifying unique series IDs from the training events
series_ids = train_events['series_id'].unique()
print(f"Total number of unique series IDs: {len(series_ids)}")

# Preparing the path for Faster R-CNN data and loading images
data_prep_folder = Path("/kaggle/input/d-s-s-faster-r-cnn-data-prep")
images_folder = data_prep_folder/"images"
image_names = [file_path.name for file_path in images_folder.glob("*.jpg")]
print(f"Total images found: {len(image_names)}")
print("First five image names:", image_names[:5])

# Reading annotations from the prepared CSV file
annotations_df = pd.read_csv(data_prep_folder/"annotations_8_30pm_utc_cutoff.csv")
print(annotations_df.head())

# Getting the unique counts for series IDs and image names in annotations
print("Number of unique series IDs in annotations:", len(annotations_df['series_id'].unique()))
print("Number of unique image names in annotations:", len(annotations_df['image_name'].unique()))

# Reading window properties from the CSV file
window_properties_df = pd.read_csv(data_prep_folder/"window_properties_8_30pm_utc_cutoff.csv")
print(window_properties_df.head())

# Splitting the dataset into training, validation, and testing sets
num_test_series_ids = round(0.1 * len(series_ids))
num_val_series_ids = round(0.2 * len(series_ids))
num_val_test_series_ids = num_test_series_ids + num_val_series_ids
np.random.seed(42)
series_ids_in_val_test = np.random.choice(series_ids, size=num_val_test_series_ids, replace=False)
series_ids_in_val = np.random.choice(series_ids_in_val_test, size = num_val_series_ids, replace=False)
series_ids_in_test = series_ids_in_val_test[~ np.isin(series_ids_in_val_test, series_ids_in_val)]
print(series_ids_in_val, len(series_ids_in_val))
print(series_ids_in_test, len(series_ids_in_test))
series_ids_in_train = series_ids[~ np.isin(series_ids, series_ids_in_val_test)]
print(series_ids_in_train, len(series_ids_in_train))

# Creating training, validation, and testing datasets from window properties
train = window_properties_df.loc[window_properties_df['series_id'].isin(series_ids_in_train)].reset_index(drop=True)
val = window_properties_df.loc[window_properties_df['series_id'].isin(series_ids_in_val)].reset_index(drop=True)
test = window_properties_df.loc[window_properties_df['series_id'].isin(series_ids_in_test)].reset_index(drop=True)
print(len(train), len(val) , len(test))

# Filtering event data for validation and testing
val_events = train_events.loc[train_events['series_id'].isin(series_ids_in_val)]
val_events = val_events.loc[val_events['timestamp'].notna()].reset_index(drop=True)
print(val_events.head())

test_events = train_events.loc[train_events['series_id'].isin(series_ids_in_test)]
test_events = test_events.loc[test_events['timestamp'].notna()].reset_index(drop=True)
print(test_events.head())

train_events = train_events.loc[train_events['series_id'].isin(series_ids_in_train)]
train_events = train_events.loc[train_events['timestamp'].notna()].reset_index(drop=True)
print(train_events.head())

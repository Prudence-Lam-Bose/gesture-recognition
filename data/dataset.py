import torch
import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from datetime import datetime
from torchtext.utils import extract_archive
from torch.utils.data import TensorDataset, DataLoader, Dataset


class CapsenseDataset(Dataset):   
    '''
    35543 samples
    '''
    def __init__(self, data, sequence_length=15):
        '''
        Args:
            data (pandas.Dataframe): Dataframe containing all sensor data with annotations
            root_dir (string): Directory with all the csv files 
            window_size (int): Duration of window frame to take. Default is 15 samples (15s of data).
        '''
        self.data = data
        self.sequence_length = sequence_length
        # self.feature_cols = list(data.columns.difference(["Gesture", "Label","Time","Series_id"]))
        self.features = torch.from_numpy(data.loc[:, ["Sns0", "Sns1", "Sns2"]].values).float()
        self.labels = torch.from_numpy(data.loc[:,"Label"].values).float() 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= self.sequence_length - 1:
            i_start = idx - self.sequence_length + 1
            X = self.features[i_start:(idx + 1), :]
        else:
            padding = self.features[0].repeat(self.sequence_length - idx - 1, 1)
            X = self.features[0:(idx + 1), :]
            X = torch.cat((padding, X), 0)

        return X, self.labels[idx]

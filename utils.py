import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from glob import glob

# data paths
PATHS = ['smalls', 'goodyear']
EXT = '*.csv'

# we are only interested in the timestamp and the normalized data
sensor_names = ['Sns0', 'Sns1', 'Sns2']
col_names = ['Time', 'Diff_LinearSlider_', 'Status_LinearSlider_']

def main():
    '''Find all the data and parse into dictionary for pickling'''
    # uncomment for smalls
    data_files = [file for path, subdir, files in os.walk(PATHS[0]) for file in glob(os.path.join(path, EXT))]
    
    # uncomment for goodyear
    # data_files = [file for path, subdir, files in os.walk(PATHS[1]) for file in glob(os.path.join(path, EXT))]

    for file in data_files:
        df = load_dataset(file)
        file_name = os.path.splitext(file)[0]
        pkl_path = f'{file_name}'

        # visualization
        # plot_sensor_data(df)

        # pickle for future use 
        with open(pkl_path, 'wb') as ppath: 
            pickle.dump(df, ppath)
    
def load_dataset(file):
    '''Load the dataset in a dictionary.
    From the dataframe, it reads the []
    columns and loads their corresponding arrays into the <dataset> dictionary
    Params:
        dataframe: a pandas dataframe with x columns []. Each column contains the normalized capsense data at each timestamp,
        which are going to be read and appended into a list
    Return: 
            dataset = {
                time: [1,2,3,4]
                'e0': [0,1,2,3]
                'e1': [data1,data2,...]
                'e2': [data1,data2,...]
            }
    '''
    df = pd.read_csv(file, parse_dates=[col_names[0]], date_parser=lambda x : pd.datetime.strptime(x, "%H:%M:%S")) # some date parsing
    dataset = select_cols(df)
    return dataset

def select_cols(df):
    """
    Helper function to select relevant columns (Time and sensors)
    """

    dict_cols = [col_names[0]] + sensor_names
    sensor_dict = {col: None for col in dict_cols}

    for sensor in sensor_names:
        diff_list = df[col_names[1] + sensor].values.tolist()
        status_list = df[col_names[2] + sensor].values.tolist()

        # changed to numpy array
        diff_arr = np.array(diff_list)
        
        # status_arr = np.array(status_list)
        ## check if diff counts and sensor status is high or low at the same time
        # diff_arr_mask = np.where(diff_arr > 0, 1, 0)
        # if not np.array_equal(diff_arr_mask, status_arr): # arrays are not equivalent
        #     idx = [diff_list.index(y) for x, y in zip(status_arr, diff_arr) if y != x]
        #     for i in idx: 
        #         np.delete(diff_arr, i)
        
        if sensor_dict[sensor] is None:
            sensor_dict[sensor] = np.expand_dims(diff_arr, 0)
        else: # update the dictionary
            sensor_dict[sensor] = np.concatenate([sensor_dict[sensor], np.expand_dims(diff_arr, 0)], axis=None)
        
        sensor_dict[sensor] = sensor_dict[sensor].flatten()
        
    sensor_dict[col_names[0]] = np.array(df[col_names[0]].tolist())

    return sensor_dict

def plot_sensor_data(df):
    """
    Helper function to plot rwa sensor data (e0, e1, e2)
    """
    new_df = pd.DataFrame.from_dict(df)
    new_df = new_df.dropna()
    
    fig, axs = plt.subplots(3,1)

    for i in range(len(sensor_names)):
        sns.lineplot(data=new_df, x='Time', y=sensor_names[i], ax=axs[i])
        
    plt.show()

if __name__ == "__main__":
    main()
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from datetime import datetime
from glob import glob
from exceptions import exceptions
from sklearn.impute import SimpleImputer
from seglearn.pipe import Pype
from seglearn.transform import Segment
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from modules.transform import FeatureExtractor
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# data paths
PATHS = ['gesture-recognition/data/smalls', 'gesture-recognition/data/goodyear']
EXT = '*.csv'

# we are only interested in the timestamp and the normalized data
sensor_names = ['Sns0', 'Sns1', 'Sns2']
col_names = ['Time', 'Diff_LinearSlider_', 'Status_LinearSlider_']

def preprocess_data(form, scale=False):
    """Extract all data and parse into dataframe, then combine the dataframes"""

    if form == "smalls":
        data_files = [file for path, subdir, files in os.walk(PATHS[0]) for file in glob(os.path.join(path, EXT))]
    elif form == "goodyear" or form == "gy":
        data_files = [file for path, subdir, files in os.walk(PATHS[1]) for file in glob(os.path.join(path, EXT))]
    
    data_list = [] 
    id = 1

    for file in data_files:
        df = load_dataset(file)
        file_name = os.path.splitext(file)[0]

        # add labels 
        if "tap" in file_name: 
            df["Gesture"] = "tap"
        elif "swipe_up" in file_name:
            df["Gesture"] = "swipe_up" 
        elif "swipe_down" in file_name:
            df["Gesture"] = "swipe_down"
        # elif "squiggle" in file_name:
        #     df["Gesture"] = "squiggle"
        # elif "circle" in file_name:
        #     df["Gesture"] = circle
        # elif "cross" in file_name:
        #     df["Gesture"] = cross
        elif "rest" in file_name:
            df["Gesture"] = "rest"
        else: 
            df["Gesture"] = "outlier" 
        
        df["Series_id"] = id 
        id += 1
        data_list.append(df)
    
    # encode gesture as a categorical variable
    all_data = pd.concat(data_list)
    all_data["Gesture"] = all_data["Gesture"].astype('category')
    all_data["Label"] = all_data["Gesture"].cat.codes

    if scale: # scale data between 0 to 1 while preserving distribution shape
        numerical_cols = [e for e in all_data.columns.values.tolist() if e in sensor_names]
        data_to_transform = all_data.loc[:, numerical_cols]
        all_data.loc[:,numerical_cols] = MinMaxScaler().fit_transform(data_to_transform)
    
    return all_data

def load_dataset(file):
    """
    Load the dataset into a pandas.Dataframe
    with relevant columns.

    Params:
        file: a pandas dataframe with x columns []. Each column contains the normalized capsense data at each timestamp,
        which are going to be read and appended into a list
    
    """
    df = pd.read_csv(file, parse_dates=[col_names[0]], date_parser=lambda x : datetime.strptime(x, "%H:%M:%S")) # some date parsing
    dataset = select_cols(df)
    dataset = dict_to_df(dataset)
    dataset = resample_df(dataset)
    return dataset

def select_cols(df):
    """
    Helper function to select relevant columns (Time and sensors)
    Returns a dictionary of form: 
            dataset = {
                time: [1,2,3,4]
                'e0': [0,1,2,3]
                'e1': [data1,data2,...]
                'e2': [data1,data2,...]
            }
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
        
    sensor_dict[col_names[0]] = np.array(df[col_names[0]].tolist()) # time

    return sensor_dict

def dict_to_df(dic):
    """
    Helper function to go from dictionary to pandas.Dataframe, does not handle NA values
    """
    new_df = pd.DataFrame.from_dict(dic) 
    return new_df

def resample_df(df):
    """
    Helper function to downsample dataset from miliseconds to seconds
    """
    df = df.resample('1S', on='Time').mean()
    return df.fillna(0)

def plot_sensor_data(df):
    """
    Helper function to plot normalized (raw - baseline) electrode data
    """
    sensor_names = ['Sns0', 'Sns1', 'Sns2']
    df_list = [g for _, g in df.groupby('Series_id')]

    fig = plt.figure(figsize=(30,40))
    fig.suptitle('Raw sensor data over time')
    fig.tight_layout()
    subfigs = fig.subfigures(nrows=7, ncols=1)

    df_id = 0 

    for row, subfig in enumerate(subfigs): 
        axs = subfig.subplots(nrows=1, ncols=3)
        curr_df = df_list[df_id]
        curr_df = curr_df[:10]

        subfig.suptitle(curr_df['Gesture'][0])
        sensor_id = 0

        for col, ax in enumerate(axs): 
            palette = sns.color_palette("mako", 3)
            curr_sensor = sensor_names[sensor_id] 
            sns.lineplot(data=curr_df, x='Time',y=curr_sensor, ax=ax, ci=None, palette=palette[sensor_id])
            sensor_id += 1
            ax.tick_params(rotation=45)
            ax.grid(which='major')
            ax.grid(which='minor')
            ax.minorticks_on()

        df_id += 1 

    plt.savefig('visual.jpg')

def crnn_model(width=100, n_vars=6, n_classes=7, conv_kernel_size=5,
               conv_filters=3, lstm_units=3):
    input_shape = (width, n_vars)
    model = Sequential()
    model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size,
                     padding='valid', activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size,
                     padding='valid', activation='relu'))
    model.add(LSTM(units=lstm_units, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(n_classes, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    return model

def classify_with_feature_stats(df, classifier,n):
    X, y, g = [g for _, g in df.groupby('Series_id')], [], []

    for i in range(len(X)):
        labels = X[i]["Label"].values 
        y.append(labels[0])
        series = X[i]["Series_id"].values
        g.append(series[0])
        X[i] = X[i].loc[:,["Sns0","Sns1","Sns2"]].fillna(0).values
    
    if classifier == 'knn':
        if n == None:
            raise exceptions.InvalidNumNeighborsKNN(
                "Selected KNN Classifier, but number of neighbors is None. Please specify an integer"
            )
        # classifier = KNeighborsClassifier(n_neighbors=int(n), metric='l2')
        classifier = KNeighborsClassifier(n_neighbors=n)
    elif classifier == 'rf':
        classifier = RandomForestClassifier()
    elif classifier == 'svc':
        classifier = LinearSVC()
    else:
        raise exceptions.InvalidClassifier(
            "Invalid classifier selected, please choose between knn, rg, or svc"
        )
    
    scaler = MinMaxScaler()

    pipe = Pype([('segment', Segment(15)),
                ('features', FeatureExtractor()),
                ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
                ('scaler', scaler),
                ('cls', classifier)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    print("Accuracy: ", score)

    return score 
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import datetime

from glob import glob

"""TODO: we want to create the classes here and also assign labels"""

# data paths
PATHS = ['smalls', 'goodyear']
EXT = '*.pkl'

def main():
    pkl_files = [file for path, subdir, files in os.walk(PATHS[0]) for file in glob(os.path.join(path, EXT))]

if __name__ == "__main__":
    main()
# import necessary libraries

import numpy as np
import pandas as pd
import os
import glob
# root directory
root_dir_path = r"D:\PythonProjects\MachinelearningProjects\Raw_data\MetaMotion\MetaMotion"

files = glob.glob(os.path.join(root_dir_path ,"*.csv"))
# put eveything in a ffunction
def preprocessed_data(files):
    # creating dataframes
    acc_df = pd.DataFrame()
    gyro_df = pd.DataFrame()

    # creeating unique identifiers
    acc_set = 1
    gyro_set = 1

    # crreating a for loop to loop over the files
    for f in files:
        # extract participant
        participant = f.split("-")[0].replace("D:\\PythonProjects\\MachinelearningProjects\\Raw_data\\MetaMotion\\MetaMotion\\","")
        # extract label
        label = f.split("-")[1]
        # extract category
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
        # read the file
        df = pd.read_csv(f)
        
        df['participant'] =  participant
        df['label'] = label
        df['category'] = category
        # check the type of file
        if "Accelerometer" in f:
            df['set'] = acc_set
            acc_set+=1
            acc_df = pd.concat([acc_df,df])
        if "Gyroscope" in f:
            df['set'] = gyro_set
            gyro_set+=1
            gyro_df = pd.concat([gyro_df,df])

    # set dattime inex foe each
    acc_df.index = pd.to_datetime(acc_df['epoch (ms)'],unit="ms")
    gyro_df.index = pd.to_datetime(gyro_df['epoch (ms)'],unit="ms")

    # delete unnessary columns from the data
    del acc_df['epoch (ms)']
    del acc_df['time (01:00)']
    del acc_df['elapsed (s)']

    # gryo scope data
    del gyro_df['epoch (ms)']
    del gyro_df['time (01:00)']
    del gyro_df['elapsed (s)']

    return acc_df , gyro_df




acc_df , gyro_df = preprocessed_data(files=files)
acc_df['category'].value_counts()

# merging thr two dataframes
data_merged = pd.concat([acc_df.iloc[:,:3],gyro_df],axis=1)

# check the columns of the merged data
data_merged.columns
# rename the columns
data_merged = data_merged.rename(columns={
    'x-axis (g)': 'acc_x',
    'y-axis (g)': 'acc_y',
    'z-axis (g)': 'acc_z',
    'x-axis (deg/s)': 'gyr_x',
    'y-axis (deg/s)': 'gyr_y',
    'z-axis (deg/s)': 'gyr_z'
})

# resample the data
sampling = {col: "mean" for col in ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"]}

# update
sampling.update({
    "label": "last",
    "category": "last",
    "participant": "last",
    "set": "last"
})

# data resmapled to 200ms
data_resampled = pd.concat(
    df.resample("200ms").agg(sampling).dropna()
    for _, df in data_merged.groupby(pd.Grouper(freq="D"))
)

# seting set to int
data_resampled.info()

data_resampled["set"] = data_resampled["set"].astype(int)

# save to pickle
data_resampled.to_pickle(r"D:\PythonProjects\MachinelearningProjects\Preprocessed\preprocessed_data.pkl")







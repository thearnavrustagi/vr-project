import pandas as pd
import math
import numpy as np

FPATH = "./dataset/IMUData.csv"
instrument, direction = ['gyroscope','accelerometer','magnetometer'],'XYZ'
data = {}

def normalize(dataframe):
    data = dataframe.to_numpy()
    for i,row in enumerate(data):
        norm = np.linalg.norm(row)
        if not norm: continue

        data[i] = row / norm;

    return pd.DataFrame(data, columns=[d for d in direction])

if __name__ == "__main__":
    dataframe = pd.read_csv(FPATH)
    print(dataframe.keys())

    for i in instrument:
        data[i] = pd.DataFrame()
        for d in direction:
            data[i][d] = dataframe[' '+i+'.'+d]

    cols = [d for d in direction]
    data['gyroscope'][cols] = np.deg2rad(data['gyroscope'][cols])
    
    for i in instrument[1:]:
        cols = [d for d in direction]
        data[i][cols] = normalize(data[i][cols])


    for i in instrument:
        data[i].to_csv(f'./dataset/{i}.csv')

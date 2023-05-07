import pandas as pd
from qmath import quaternion, euler, quaternion_multiply, unit, euler2quat, euler_between
import numpy as np

def main ():
    gyroscope = pd.read_csv('./dataset/quaternion_gyroscope.csv')
    time = pd.read_csv('./dataset/time.csv')
    df = dead_reckoning(gyroscope, time)
    print("dead reckoning done")
    tilt_correction(df)
    print("tilt correction done")

def tilt_correction(predicted):
    acceleration = pd.read_csv('./dataset/accelerometer.csv')
    alpha = 0.01

    data = []
    for i,row in predicted.iterrows():
        a = acceleration.iloc[[i]].to_numpy()[0][1:]
        phi = (euler_between(a,(0,1,0)))

        data.append(np.array(quaternion_multiply(quaternion(-1*alpha*phi,a),row)))

    df = pd.DataFrame(np.array(data),columns=['X Y Z W'.split()])
    df.to_csv('./dataset/tilt_correction.csv')
    

def dead_reckoning(gyroscope,time):
    print(np.column_stack(time.to_numpy()))
    prev_time = 0.0
    displacement = (1,0,0,0)
    data = [np.array(displacement)]
    for i, row in gyroscope.iterrows():
        curr_time = time.iloc[[i]].to_numpy()[0][1]

        delta = curr_time - prev_time
        net_rotation = np.linalg.norm(row)*delta
        quat = quaternion(net_rotation,unit(row))
        displacement = quaternion_multiply(data[-1],quat)
        prev_time = curr_time

        data.append(np.array(displacement))

    df = pd.DataFrame(np.array(data)[1:],columns=['X Y Z W'.split()])
    df.to_csv('./dataset/dead_reckoning.csv')

    return df
    

if __name__ == "__main__":
    main()

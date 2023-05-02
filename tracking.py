import pandas as pd

FPATH = "IMUData.csv"

if __name__ == "__main__":
    dataframe = pd.read_csv(FPATH)
    print(dataframe)

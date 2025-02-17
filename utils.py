import pandas as pd

import torch

def load_data(is_full=False):
    df = pd.read_csv("./data/boston_house_prices.csv", skiprows=1)
    df.rename(columns={"MEDV" : "TARGET"}, inplace=True)

    if is_full:
        data = torch.from_numpy(df.values).float()
    else:
        cols = ["INDUS", "RM", "LSTAT", "NOX", "DIS", "TARGET"]
        data= torch.from_numpy(df[cols].values).float()

    x = data[:, :-1]
    y = data[:, -1:]

    return x, y
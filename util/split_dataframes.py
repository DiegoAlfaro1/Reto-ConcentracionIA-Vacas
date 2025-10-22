import pandas as pd

def split_dataframe(df: pd.DataFrame, train_percent: int, test_percent: int, val_percent: int):

    train_end = len(df) * train_percent // 100
    test_end = train_end + len(df) * test_percent // 100
    val_end = test_end + len(df) * val_percent // 100

    train = df.iloc[0:train_end]
    test = df.iloc[train_end : test_end]
    val = df.iloc[test_end : val_end]

    return train, test, val
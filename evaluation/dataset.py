import pandas as pd



def load_evaluation_dataset(path):
    df = pd.read_csv(path)
    df = df.to_dict("records")
    return df

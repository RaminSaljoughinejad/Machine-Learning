import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np


def data():
    df = pd.read_csv("student-mat.csv", sep=";")
    myPreprocessing = preprocessing.LabelEncoder()
    df_dict = {}
    for col in df.columns:
        if df[col].dtype=="object":
            exec(f"df_dict['{col}']=myPreprocessing.fit_transform(list(df['{col}']))")
        else:
            exec(f"df_dict['{col}']=df['{col}'].values.tolist()")
    df = pd.DataFrame(df_dict)
    return df
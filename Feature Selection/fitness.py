import sklearn
from sklearn import linear_model
import constants as c
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold

def fitness(population_list, n, m, df):
    names = list(df.columns)
    for i in range(m*2):
        temp = ["G3"]
        if population_list[i][-1]==0:
            for j in range(c.N-1):
                if population_list[i][j]==0:
                    temp.append(names[j])
            X = df.drop(temp,axis="columns")
            y = df[c.PREDICT]
            k_folds = KFold(n_splits=5)
            model = linear_model.LinearRegression()
            score = cross_val_score(model, X,y, cv=k_folds)
            # x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=c.RANDOM_STATE)
            # model.fit(x_train, y_train)
            # score = model.score(x_test, y_test)
            population_list[i][-1]=score.mean()
    return population_list



from os import popen
import numpy as np


def init_population(n, m):
    population_list = np.random.randint(2, size=(m*2,n+1))
    population_list[:,-1]=0
    return population_list.astype('float64')
    


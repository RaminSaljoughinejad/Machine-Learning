import numpy as np

def cross_over(population_list, n, m):
    half = n//2
    for i in range(0,m,2):
        child1 =np.concatenate((population_list[i][:half], population_list[i+1][half:]))
        child2 =np.concatenate((population_list[i+1][:half], population_list[i][half:]))
        child1[-1]=0
        child2[-1]=0
        population_list[m+i]  = child1
        population_list[m+i+1]= child2
    return population_list
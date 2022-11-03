from random import shuffle
from random import randint as rnd

def mutation(population_list, n, m, r):
    lst = list(range(m,m*2))
    shuffle(lst)
    lst = lst[:int(m*r)]
    for i in lst:
        cell = rnd(0,n-1)
        population_list[i][cell]=1 if population_list[i][cell]==0 else 0
    return population_list
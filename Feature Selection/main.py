#Importing Dependencies
import constants as c
from init_population import init_population
from cross_over import cross_over
from mutation import mutation
from fitness import fitness
from data_prepration import data

#main
if __name__=="__main__":
    df = data()
    current_population = init_population(c.N, c.POPULATION_SIZE)
    for i in range(c.EPOCH):
        current_population = cross_over(current_population, c.N, c.POPULATION_SIZE)
        current_population = mutation(current_population, c.N, c.POPULATION_SIZE, c.MUTATION_RATE)
        current_population = fitness(current_population, c.N, c.POPULATION_SIZE, df)
        current_population = sorted(current_population, key=lambda x:-x[-1])
        print(i+1,current_population[0][-1])
    else:
        print("Best:",current_population[0])
    
"""
Simple Genetic algorithm
"""
import numpy as np

def evaluation(eval_func, population, evaluations, x_1, x_2, nr_of_genes):
    """
    Evaluation function - Simple Genetic Algorithm
    """
    for i in range(x_1, x_2, nr_of_genes):
        result = eval_func(population[i:i+nr_of_genes])
        evaluations[int(i/nr_of_genes)] = result

def tournament_selection(population, temp_population, evaluations, pop_size, x_1, x_2, nr_of_genes, size = 3):
    """
    Tournament selection method - Simple Genetic Algorithm
    """
    for i in range(x_1, x_2, nr_of_genes):
        indexes = np.random.randint(0, pop_size, size)

        best = indexes[0]
        for j in range(1, len(indexes)):
            if evaluations[indexes[j]] < evaluations[best]:
                best = indexes[j]

        for j in range(0, nr_of_genes):
            temp_population[i+j] = population[best+j]

def linear_crossing(evaluation_func, population, x_1, x_2, nr_of_genes):
    """
    Linear crossing method - Simple Genetic Algorithm
    """
    CHILDREN_C = 3
    for idx in range(x_1, x_2, nr_of_genes*2):
        first = np.array(population[idx:idx+nr_of_genes])
        second = np.array(population[idx+nr_of_genes:idx+2*nr_of_genes])

        childs = np.zeros(CHILDREN_C * nr_of_genes)
        childs[0:nr_of_genes] = 1.5 * first - 0.5 * second
        childs[nr_of_genes:nr_of_genes*2] = 0.5 * first - 1.5 * second
        childs[nr_of_genes*2:nr_of_genes*3] = 0.5 * first + 0/5 * second

        fitnesses = np.fromiter((evaluation_func(childs[i:i+nr_of_genes]) for i in range(0, CHILDREN_C*nr_of_genes, nr_of_genes)), dtype=float)
        ind = np.argpartition(fitnesses, -2)[:2]
        population[idx:idx+nr_of_genes] = childs[ind[0]*nr_of_genes:ind[0]*nr_of_genes+nr_of_genes]
        population[idx+nr_of_genes:idx+2*nr_of_genes] = childs[ind[1]*nr_of_genes:ind[1]*nr_of_genes+nr_of_genes]

def mutation(p_mute, population, x_1, x_2, nr_of_genes):
    """
    Mutation - Simple Genetic Algorithm
    """
    mu, sigma = 0, 0.1 # mean and standard deviation
    for i in range(x_1, x_2, nr_of_genes):
        factor = np.random.uniform(0.0, 0.1)
        if factor < p_mute:
            genes = population[i:i+nr_of_genes]
            vector = np.random.normal(mu, sigma, nr_of_genes)
            population[i:i+nr_of_genes] = genes+vector

def linear_crossing_with_mutation(evaluation_func, p_mute, population, x_1, x_2, nr_of_genes):
    """
    Linear crossing method - Simple Genetic Algorithm
    """
    CHILDREN_C = 3
    mu, sigma = 0, 0.1 # mean and standard deviation
    for idx in range(x_1, x_2, nr_of_genes*2):
        first = np.array(population[idx:idx+nr_of_genes])
        second = np.array(population[idx+nr_of_genes:idx+2*nr_of_genes])

        childs = np.zeros(CHILDREN_C * nr_of_genes)
        childs[0:nr_of_genes] = 1.5 * first - 0.5 * second
        childs[nr_of_genes:nr_of_genes*2] = 0.5 * first - 1.5 * second
        childs[nr_of_genes*2:nr_of_genes*3] = 0.5 * first + 0/5 * second

        fitnesses = np.fromiter((evaluation_func(childs[i:i+nr_of_genes]) for i in range(0, CHILDREN_C*nr_of_genes, nr_of_genes)), dtype=float)
        ind = np.argpartition(fitnesses, -2)[:2]

        population[idx:idx+nr_of_genes] = childs[ind[0]*nr_of_genes:ind[0]*nr_of_genes+nr_of_genes]
        population[idx+nr_of_genes:idx+2*nr_of_genes] = childs[ind[1]*nr_of_genes:ind[1]*nr_of_genes+nr_of_genes]

        for i in range(0, CHILDREN_C-1):
            factor = np.random.uniform(0.0, 0.1)
            if factor < p_mute:
                if i == 1:
                    genes = population[idx:idx+nr_of_genes]
                    vector = np.random.normal(mu, sigma, nr_of_genes)
                    population[idx:idx+nr_of_genes] = genes+vector
                else:
                    genes = population[idx+nr_of_genes:idx+2*nr_of_genes]
                    vector = np.random.normal(mu, sigma, nr_of_genes)
                    population[idx+nr_of_genes:idx+2*nr_of_genes] = genes+vector

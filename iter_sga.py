"""
Implementation of Simple Genetic Algorithm
"""
#!/usr/bin/python
import sys
import numpy as np
from algorithm import evaluation, tournament_selection, linear_crossing, mutation

def run(func, nr_of_genes, min_x, max_x, size):
    """
    Run iterative Simple Genetic Algorithm with parameters:
    func - evaluation function
    nr_of_genes - dimension of func (genes per individual)
    min_x - minimum value of randomly generated gene information
    max_x - maximum value of randomly generated gene information
    size - size of population, amount of individuals
    """
    print(func, min_x, max_x, nr_of_genes, size)

    pop = np.random.uniform(min_x, max_x, (size*nr_of_genes))
    n_pop = np.zeros(size*nr_of_genes)
    evals = np.zeros(size)

    value = sys.float_info.max
    delta = 1e-2
    generation = 0
    while generation < 100:
        # Evaluation
        evaluation(func, pop, evals, 0, size*nr_of_genes, nr_of_genes)

        value = min(evals)
        if value < delta:
            break

        # Keep fittest element in population
        fittest_id = np.argmin(evals)
        for gene in range(nr_of_genes):
            n_pop[gene] = pop[fittest_id*nr_of_genes+gene]

        # Selection
        tournament_selection(pop, n_pop, evals, size, nr_of_genes, size*nr_of_genes, nr_of_genes)

        # Crossing
        p_cross = 0.8
        linear_crossing(func, n_pop, 0, int(size*nr_of_genes*p_cross), nr_of_genes)

        # Mutation
        p_mute = 0.015
        mutation(p_mute, n_pop, 0, size*nr_of_genes, nr_of_genes)

        pop = np.frombuffer(n_pop)
        generation += 1

    fittest_id = np.argmin(evals)
    fittest_individual = pop[fittest_id*nr_of_genes:fittest_id*nr_of_genes+nr_of_genes]
    print("Generation number", generation)
    print("fittest individual", fittest_individual, func(fittest_individual))

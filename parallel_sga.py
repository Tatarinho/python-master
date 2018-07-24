"""
Implementation of concurrent Simple Genetic Algorithm
"""
#!/usr/bin/python
import sys
import multiprocessing as mp
import numpy as np
from algorithm import evaluation, tournament_selection, linear_crossing, mutation

def run(func, nr_of_genes, min_x, max_x, size, procs):
    """
    Run concurrent Simple Genetic Algorithm with parameters:
    func - evaluation function
    nr_of_genes - dimension of func (genes per individual)
    min_x - minimum value of randomly generated gene information
    max_x - maximum value of randomly generated gene information
    size - size of population, amount of individuals
    procs - amount of process run in parallel
    """
    print(func, min_x, max_x, nr_of_genes, size, procs)

    pop = np.random.uniform(min_x, max_x, (size*nr_of_genes))
    n_pop = mp.Array('f', size*nr_of_genes, lock=False)
    evals = mp.Array('f', size, lock=False)

    value = sys.float_info.max
    delta = 1e-2
    generation = 0
    while generation < 100:
        indexes = [int(size/procs)] * procs
        for idx in range(int(size%procs)):
            indexes[idx] += 1
        # Evaluation
        processes = np.empty(procs, dtype=mp.Process)
        for idx in range(procs):
            x_1 = sum(indexes[:idx]) * nr_of_genes
            x_2 = x_1 + indexes[idx] * nr_of_genes
            args = (func, pop, evals, x_1, x_2, nr_of_genes)
            process = mp.Process(target=evaluation, args=args)
            processes[idx] = process
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        value = min(evals)
        #print("generation", generation, "fittest value", value)
        if value < delta:
            break

        # Keep fittest element in population
        fittest_id = np.argmin(evals)
        for gene in range(nr_of_genes):
            n_pop[gene] = pop[fittest_id*nr_of_genes+gene]

        # Selection
        processes = np.empty(procs, dtype=mp.Process)
        x_1 = nr_of_genes
        for idx in range(procs):
            if idx == 0:
                x_2 = indexes[idx] * nr_of_genes
            else:
                x_2 = x_1 + indexes[idx] * nr_of_genes
            p_target = tournament_selection
            p_args = (pop, n_pop, evals, size, x_1, x_2, nr_of_genes)
            process = mp.Process(target=p_target, args=p_args)
            processes[idx] = process
            x_1 = x_2
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        # Crossing
        p_cross = 0.8
        q_to_cross = int(p_cross*len(n_pop)/nr_of_genes)

        PROCS = int(q_to_cross/2)
        crossing_idx = [int(PROCS/procs)] * procs
        it = 0
        while sum(crossing_idx) < PROCS:
            crossing_idx[it] += 1
            it += 1
        crossing_idx[:] = (value for value in crossing_idx if value != 0)

        processes = np.empty(len(crossing_idx), dtype=mp.Process)
        x_1 = 0
        for idx, c_idx in enumerate(crossing_idx):
            x_2 = x_1 + 2 * c_idx
            target = linear_crossing
            args = (func, n_pop, x_1, x_2, nr_of_genes)
            process = mp.Process(target=target, args=args)
            processes[idx] = process
            x_1 = x_2
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        # Mutation
        p_mute = 0.015
        processes = np.empty(procs, dtype=mp.Process)
        for idx in range(procs):
            x_1 = sum(indexes[:idx]) * nr_of_genes
            x_2 = x_1 + indexes[idx] * nr_of_genes
            args = (p_mute, n_pop, x_1, x_2, nr_of_genes)
            process = mp.Process(target=mutation, args=args)
            processes[idx] = process
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        pop = n_pop[0:size*nr_of_genes]
        generation += 1

    fittest_id = np.argmin(evals)
    fittest_individual = pop[fittest_id*nr_of_genes:fittest_id*nr_of_genes+nr_of_genes]
    print("Generation number", generation)
    print("fittest individual", fittest_individual, func(fittest_individual))

"""
This module contains optimization functions
"""
from math import cos, sqrt

def rosenbrock(X):
    """
    Return result of multidimensional Rosenbrock function
    """
    sum_r = 0
    for i in range(len(X)-1):
        sum_r += 100*(X[i+1]-X[i]**2)**2 + (1-X[i])**2
        #sum_r += rosenbrock(X[i], X[i+1])
    return sum_r

def griewank(X):
    """
    Returns result of multidimensional Griewank function
    """
    genes = [x for x in X]
    g_sum = 0
    for gene in genes:
        g_sum += gene * gene
    product = 1
    for idx, gene in enumerate(genes):
        product *= cos(gene / sqrt(idx + 1))

    return 1 + g_sum / 4000 - product

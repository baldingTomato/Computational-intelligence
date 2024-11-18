import pygad
import math
import numpy as np

# Funkcja wytrzymałości stopu
def endurance(x, y, z, u, v, w):
    return math.exp(-2 * (y - math.sin(x))**2) + math.sin(z * u) + math.cos(v * w)

# Funkcja fitness (zgodna z PyGAD 2.20.0)
def fitness_func(ga_instance, solution, solution_idx):
    x, y, z, u, v, w = solution  # Rozpakowanie chromosomu
    return endurance(x, y, z, u, v, w)  # Obliczenie wytrzymałości

# Parametry algorytmu
gene_space = {'low': 0.0, 'high': 1.0}  # Przestrzeń genów [0, 1)
num_genes = 6  # Chromosom: [x, y, z, u, v, w]
sol_per_pop = 30  # Liczba osobników w populacji
num_parents_mating = 10  # Liczba rodziców do krzyżowania
num_generations = 100  # Liczba pokoleń
mutation_percent_genes = 20  # Procent genów do mutacji

# Tworzenie instancji algorytmu
ga_instance = pygad.GA(
    num_generations=num_generations,
    sol_per_pop=sol_per_pop,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,  # Nowa funkcja fitness
    gene_space=gene_space,
    num_genes=num_genes,
    mutation_percent_genes=mutation_percent_genes,
    crossover_type="single_point",
    mutation_type="random",
)

# Uruchomienie algorytmu
ga_instance.run()

# Wyniki
best_solution, best_fitness, _ = ga_instance.best_solution()
print("Najlepszy chromosom (ilości metali):", best_solution)
print("Najlepsza wytrzymałość stopu:", best_fitness)

# Wykres konwergencji
ga_instance.plot_fitness()

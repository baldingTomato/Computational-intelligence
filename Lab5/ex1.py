import pygad
import numpy as np
import time

values = [100, 300, 200, 40, 500, 70, 100, 250, 300, 280, 300]
weights = [7, 7, 6, 2, 5, 6, 1, 3, 10, 3, 15]
weight_limit = 25
target_fitness = 1630
stop_flag = False  # Flaga globalna

def fitness_func(model, solution, solution_idx):
    total_value = np.sum(solution * values)
    total_weight = np.sum(solution * weights)
    if total_weight > weight_limit:
        return total_value - (total_weight - weight_limit) * 100  # Kara za nadwagę
    return total_value

def on_generation(ga_instance):
    global stop_flag
    if ga_instance.best_solution()[1] == target_fitness:
        stop_flag = True
        ga_instance.generations_completed = ga_instance.num_generations

def run_experiment():
    gene_space = [0, 1]
    sol_per_pop = 50
    num_genes = len(values)
    num_parents_mating = 20
    num_generations = 500
    keep_parents = 5
    parent_selection_type = "rank"
    crossover_type = "uniform"
    mutation_type = "random"
    mutation_percent_genes = 20

    ga_instance = pygad.GA(
        gene_space=gene_space,
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_func,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
        on_generation=on_generation
    )

    start_time = time.time()
    ga_instance.run()
    end_time = time.time()

    best_solution, best_fitness, _ = ga_instance.best_solution()
    return best_fitness, end_time - start_time

results = []
times = []

for i in range(10):
    fitness, elapsed_time = run_experiment()
    results.append(fitness == target_fitness)
    if fitness == target_fitness:
        times.append(elapsed_time)

success_rate = (sum(results) / len(results)) * 100
average_time = np.mean(times) if times else None

print(f"Skuteczność: {success_rate:.2f}%")
if average_time is not None:
    print(f"Średni czas działania (dla sukcesów): {average_time:.2f} s")
else:
    print("Nie znaleziono najlepszego rozwiązania w żadnym uruchomieniu.")

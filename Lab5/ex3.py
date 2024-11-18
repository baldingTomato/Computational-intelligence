import pygad
import time

# Labirynt jako macierz
maze = [
    [1, 0, 1, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 1, 1, 0, 1],
    [0, 0, 1, 1, 1],
    [1, 1, 1, 0, 2]
]
# Start i cel
start = (0, 0)
goal = (4, 4)

# Funkcja fitness
def fitness_func(ga_instance, solution, solution_idx):
    x, y = start  # Startowa pozycja
    score = 0
    for step in solution:
        if step == 0:  # Góra
            x -= 1
        elif step == 1:  # Dół
            x += 1
        elif step == 2:  # Lewo
            y -= 1
        elif step == 3:  # Prawo
            y += 1

        # Sprawdzenie granic labiryntu
        if x < 0 or y < 0 or x >= len(maze) or y >= len(maze[0]) or maze[x][y] == 0:
            break  # Ruch niepoprawny, przerwij

        score += 1  # Premia za poprawny ruch

        # Premia za dotarcie do celu
        if maze[x][y] == 2:
            score += 100  # Duża premia za sukces
            break

    # Kara za odległość od celu
    goal_x, goal_y = goal
    distance = abs(goal_x - x) + abs(goal_y - y)  # Manhattan
    score -= distance  # Kara za odległość

    return score

# Przestrzeń genów: 4 możliwe kierunki
gene_space = [0, 1, 2, 3]

# Callback na generację, zatrzymuje algorytm po znalezieniu rozwiązania
def on_generation(ga_instance):
    best_solution, best_fitness, _ = ga_instance.best_solution()
    if best_fitness > 100:  # 100+ oznacza, że znaleziono cel
        print(f"Rozwiązanie znalezione w pokoleniu: {ga_instance.generations_completed}")
        ga_instance.terminate_generation = True  # Wymuszenie zakończenia

# Parametry algorytmu genetycznego
ga_instance = pygad.GA(
    num_generations=50,            # Liczba pokoleń
    sol_per_pop=20,                # Liczba osobników w populacji
    num_parents_mating=5,          # Liczba rodziców do krzyżowania
    fitness_func=fitness_func,      # Funkcja fitness
    gene_space=gene_space,          # Przestrzeń genów
    num_genes=24,                   # Liczba genów (maksymalna liczba kroków)
    mutation_percent_genes=10,      # Procent genów do mutacji
    crossover_type="single_point",  # Krzyżowanie jednopunktowe
    mutation_type="random",         # Mutacja losowa
    on_generation=on_generation     # Callback na generację
)

# Funkcja do uruchomienia algorytmu i pomiaru czasu
def run_experiment():
    start_time = time.time()
    ga_instance.run()
    end_time = time.time()
    best_solution, best_fitness, _ = ga_instance.best_solution()
    return best_solution, best_fitness, end_time - start_time

# Powtarzamy eksperyment 10 razy
results = [run_experiment() for _ in range(10)]
average_time = sum(res[2] for res in results) / len(results)

# Wyświetlamy wyniki
print("Najlepsze wyniki dla 10 uruchomień:")
for i, (solution, fitness, exec_time) in enumerate(results):
    print(f"Uruchomienie {i+1}: Fitness = {fitness}, Czas = {exec_time:.2f}s")

print(f"\nŚredni czas: {average_time:.2f}s")

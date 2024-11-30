import numpy as np
import math
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

# Definicja funkcji endurance
def endurance(position):
    x, y, z, u, v, w = position
    return -(math.exp(-2 * (y - math.sin(x)) ** 2) + math.sin(z * u) + math.cos(v * w))

def f(swarm):
    """
    `swarm` to tablica Nx6, gdzie N to liczba czastek, a 6 to liczba wymiarow.
    """
    return np.array([endurance(p) for p in swarm])

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
bounds = (np.zeros(6), np.ones(6)) 

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds=bounds)

best_cost, best_pos = optimizer.optimize(f, iters=1000)

print(f"Najlepszy wynik: {best_cost}")
print(f"Najlepsza pozycja: {best_pos}")

plot_cost_history(optimizer.cost_history)
plt.title("Historia kosztu w algorytmie PSO")
plt.xlabel("Iteracje")
plt.ylabel("Koszt")
plt.show()

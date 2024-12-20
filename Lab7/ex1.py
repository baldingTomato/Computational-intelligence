import matplotlib.pyplot as plt
import random

from aco import AntColony


plt.style.use("dark_background")


COORDS = (
    (20, 52),
    (43, 50),
    (20, 84),
    (70, 65),
    (29, 90),
    (87, 83),
    (73, 23),
    (15, 34),
    (55, 12),
    (90, 70),
    (45, 45),
)


def random_coord():
    r = random.randint(0, len(COORDS))
    return r


def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))


plot_nodes()

colony = AntColony(COORDS, ant_count=500, alpha=1.0, beta=2.0, 
                    pheromone_evaporation_rate=0.3, pheromone_constant=1200.0,
                    iterations=400)

optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )


plt.show()



# Zwiększenie liczby mrówek (ant_count) poprawia jakość rozwiązania, ale wydłuża czas działania.
# Wyższe wartości alpha zwiększają znaczenie feromonów, co może prowadzić do szybszej konwergencji.
# Wyższe wartości beta zwiększają znaczenie odległości, co poprawia eksplorację na początku.
# Zmniejszenie współczynnika wyparowania feromonów (pheromone_evaporation_rate) wydłuża czas działania.


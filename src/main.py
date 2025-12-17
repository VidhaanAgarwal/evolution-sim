from config import Config
from evo_basic import make_population, evolve
from evo_food import spawn_food
from visualization import visualize_population

cfg = Config()

population = make_population(cfg)

# visualize initial behavior
visualize_population(population[:10], cfg)

for gen in range(20):
    population, best, mean, alive = evolve(population, cfg)
    print(
        f"Gen {gen:02d} | "
        f"best: {best:.2f} | "
        f"mean: {mean:.2f} | "
        f"alive: {alive}"
    )

# visualize evolved agents
visualize_population(population[:5], cfg)

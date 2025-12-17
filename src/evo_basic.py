import numpy as np
import math
from evo_food import sense_food, try_eat,respawn_food,spawn_food

# ===================== RNN =====================

class RNNNet:
    def __init__(self, inputs, hidden, outputs, cfg):
        self.cfg = cfg
        self.Wxh = np.random.randn(hidden, inputs) * cfg.W_INIT_SCALE
        self.Whh = np.random.randn(hidden, hidden) * cfg.WHH_INIT_SCALE
        self.Why = np.random.randn(outputs, hidden) * cfg.W_INIT_SCALE
        self.bh = np.zeros(hidden)
        self.by = np.zeros(outputs)
        self.h = np.zeros(hidden)

    def reset(self):
        self.h[:] = 0.0

    def calc(self, x):
        self.h = np.tanh(self.Wxh @ x + self.Whh @ self.h + self.bh)
        return np.tanh(self.Why @ self.h + self.by)


# ===================== Agent =====================

class Agent:
    def __init__(self, nn, cfg):
        self.nn = nn
        self.cfg = cfg
        self.food_eaten = 0
        self.x = np.random.uniform(0.3, 0.7)
        self.y = np.random.uniform(0.3, 0.7)
        self.angle = np.random.uniform(-math.pi, math.pi)
        self.energy = cfg.START_ENERGY
        self.turn_vel = 0.0
        self.steps_alive = 0
        self.alive = True


# ===================== Simulation =====================

def step_agent(agent, foods, cfg):
    if not agent.alive:
        return

    fl, fc, fr = sense_food(agent, foods, cfg)

    agent.steps_alive += 1


    inputs = np.array([
        agent.x,
        agent.y,
        agent.energy,
        math.cos(agent.angle),
        math.sin(agent.angle),
        fl, fc, fr
    ])

    forward, turn = agent.nn.calc(inputs)

    speed = cfg.MAX_SPEED_SCALE * math.log1p(math.exp(forward))

    agent.turn_vel = (
        cfg.TURN_SMOOTH * agent.turn_vel +
        cfg.TURN_RESPONSE * turn
    )
    agent.energy -= cfg.TURN_COST * abs(agent.turn_vel)


    agent.angle += agent.turn_vel * cfg.DT
    agent.angle = (agent.angle + math.pi) % (2 * math.pi) - math.pi

    agent.x += math.cos(agent.angle) * speed * cfg.DT
    agent.y += math.sin(agent.angle) * speed * cfg.DT

    agent.energy -= (
        cfg.BASE_ENERGY_LOSS +
        cfg.SPEED_ENERGY_COST * abs(speed)
    )

    agent.x = max(cfg.WORLD_MIN, min(cfg.WORLD_MAX, agent.x))
    agent.y = max(cfg.WORLD_MIN, min(cfg.WORLD_MAX, agent.y))

    try_eat(agent, foods, cfg)

    if agent.energy <= 0:
        agent.alive = False



def fitness(agent):
    return agent.food_eaten + 0.01 * agent.steps_alive



# ===================== Evolution =====================

def mutate(net, cfg):
    s = cfg.MUT_SIGMA
    net.Wxh += np.random.randn(*net.Wxh.shape) * s
    net.Why += np.random.randn(*net.Why.shape) * s
    net.Whh += np.random.randn(*net.Whh.shape) * (s * cfg.RECURRENT_MUT_SCALE)
    net.bh += np.random.randn(*net.bh.shape) * s
    net.by += np.random.randn(*net.by.shape) * s


def clone(net, cfg):
    new = RNNNet(
        net.Wxh.shape[1],
        net.Wxh.shape[0],
        net.Why.shape[0],
        cfg
    )
    new.Wxh = net.Wxh.copy()
    new.Whh = net.Whh.copy()
    new.Why = net.Why.copy()
    new.bh = net.bh.copy()
    new.by = net.by.copy()
    new.reset()
    return new


def evolve(population, cfg):
    # fresh world for the generation
    foods = spawn_food(cfg)

    # reset agents
    for agent in population:
        agent.alive = True
        agent.energy = cfg.START_ENERGY
        agent.nn.reset()
        agent.steps_alive = 0

    # GLOBAL TIME LOOP (this is the key)
    for _ in range(cfg.STEPS_PER_EPISODE):
        # step all agents
        for agent in population:
            if agent.alive:
                step_agent(agent, foods, cfg)
                agent.steps_alive += 1

        # WORLD UPDATE ONCE PER TIMESTEP
        respawn_food(foods, cfg)

    # now evaluate fitness
    population.sort(key=fitness, reverse=True)

    elite_n = max(1, int(cfg.ELITE_FRAC * len(population)))
    elites = population[:elite_n]

    new_population = []

    for i in range(min(2, elite_n)):
        new_population.append(Agent(clone(elites[i].nn, cfg), cfg))

    while len(new_population) < len(population):
        parent = np.random.choice(elites)
        child = clone(parent.nn, cfg)
        mutate(child, cfg)
        new_population.append(Agent(child, cfg))

    alive = sum(a.alive for a in population)
    best = fitness(population[0])
    mean = sum(fitness(a) for a in population) / len(population)

    return new_population, best, mean, alive



def make_population(cfg):
    return [
        Agent(
            RNNNet(cfg.INPUTS, cfg.HIDDEN, cfg.OUTPUTS, cfg),
            cfg
        )
        for _ in range(cfg.POP_SIZE)
    ]

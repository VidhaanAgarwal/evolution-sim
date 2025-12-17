import numpy as np
import random
import math

class Food:
    def __init__(self, x, y, energy):
        self.x = x
        self.y = y
        self.energy = energy
        self.alive = True


def spawn_food(cfg):
    foods = []

    for _ in range(cfg.FOOD_CLUMPS):
        cx = np.random.uniform(cfg.WORLD_MIN, cfg.WORLD_MAX)
        cy = np.random.uniform(cfg.WORLD_MIN, cfg.WORLD_MAX)

        for _ in range(cfg.FOOD_PER_CLUMP):
            fx = cx + random.gauss(0, cfg.FOOD_SPREAD)
            fy = cy + random.gauss(0, cfg.FOOD_SPREAD)

            fx = max(cfg.WORLD_MIN, min(cfg.WORLD_MAX, fx))
            fy = max(cfg.WORLD_MIN, min(cfg.WORLD_MAX, fy))

            foods.append(Food(fx, fy, cfg.FOOD_ENERGY))

    return foods


def sense_food(agent, foods, cfg):
    left = center = right = 0.0
    ax, ay = agent.x, agent.y

    for food in foods:
        if not food.alive:
            continue

        dx = food.x - ax
        dy = food.y - ay
        dist2 = dx * dx + dy * dy

        if dist2 > cfg.VISION_RADIUS ** 2:
            continue

        angle = math.atan2(dy, dx)
        rel = (angle - agent.angle + math.pi) % (2 * math.pi) - math.pi

        if abs(rel) > cfg.VISION_ANGLE / 2:
            continue

        strength = cfg.FOOD_SIGNAL_GAIN / (1.0 + dist2)

        if rel < -cfg.VISION_ANGLE / 6:
            left += strength
        elif rel > cfg.VISION_ANGLE / 6:
            right += strength
        else:
            center += strength

    return left, center, right

def respawn_food(foods, cfg):
    """Replace eaten food with new food elsewhere"""
    alive_count = sum(f.alive for f in foods)

    target = cfg.FOOD_CLUMPS * cfg.FOOD_PER_CLUMP
    missing = target - alive_count

    for _ in range(missing):
        if random.random() < cfg.FOOD_RESPAWN_RATE:
            foods.append(spawn_single_food(cfg))
def spawn_single_food(cfg):
    cx = np.random.uniform(cfg.WORLD_MIN, cfg.WORLD_MAX)
    cy = np.random.uniform(cfg.WORLD_MIN, cfg.WORLD_MAX)

    fx = cx + random.gauss(0, cfg.FOOD_SPREAD)
    fy = cy + random.gauss(0, cfg.FOOD_SPREAD)

    fx = max(cfg.WORLD_MIN, min(cfg.WORLD_MAX, fx))
    fy = max(cfg.WORLD_MIN, min(cfg.WORLD_MAX, fy))

    return Food(fx, fy, cfg.FOOD_ENERGY)


def try_eat(agent, foods, cfg):
    for food in foods:
        if not food.alive:
            continue

        if (agent.x - food.x) ** 2 + (agent.y - food.y) ** 2 < cfg.EAT_RADIUS ** 2:
            agent.energy += food.energy
            agent.food_eaten += 1
            food.alive = False
            break

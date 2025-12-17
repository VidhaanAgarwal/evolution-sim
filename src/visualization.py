import pygame
import math
from collections import deque
from evo_basic import step_agent
from evo_food import spawn_food, respawn_food

WIDTH, HEIGHT = 600, 600

def world_to_screen(x, y):
    return int(x * WIDTH), int(y * HEIGHT)


def draw_vision_cone(screen, agent, cfg, color=(80, 80, 120)):
    cx, cy = world_to_screen(agent.x, agent.y)
    radius = int(cfg.VISION_RADIUS * WIDTH)

    start = agent.angle - cfg.VISION_ANGLE / 2
    end   = agent.angle + cfg.VISION_ANGLE / 2

    points = [(cx, cy)]
    steps = 12  # purely visual

    for i in range(steps + 1):
        a = start + (end - start) * i / steps
        px = cx + math.cos(a) * radius
        py = cy + math.sin(a) * radius
        points.append((int(px), int(py)))

    s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.polygon(s, (*color, 40), points)
    screen.blit(s, (0, 0))


def wait_for_start(screen, clock):
    font = pygame.font.SysFont(None, 36)

    text = font.render("Press SPACE to start", True, (230, 230, 230))
    sub  = font.render("ESC to quit", True, (180, 180, 180))

    rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    rect2 = sub.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 40))

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return False
                if event.key == pygame.K_SPACE:
                    waiting = False

        screen.fill((20, 20, 20))
        screen.blit(text, rect)
        screen.blit(sub, rect2)
        pygame.display.flip()
        clock.tick(30)

    return True


def visualize_population(population, cfg):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Evolution Visualization")
    clock = pygame.time.Clock()

    # ---- WAIT SCREEN ----
    if not wait_for_start(screen, clock):
        return

    # ---- WORLD INIT ----
    foods = spawn_food(cfg)

    # ---- AGENT INIT ----
    for agent in population:
        agent.alive = True
        agent.energy = cfg.START_ENERGY
        agent.nn.reset()
        agent.trail = deque(maxlen=30)

    # ---- GLOBAL TIME LOOP ----
    for _ in range(cfg.STEPS_PER_EPISODE):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                return

        screen.fill((20, 20, 20))

        # draw food
        for food in foods:
            if food.alive:
                pygame.draw.circle(
                    screen,
                    (0, 200, 0),
                    world_to_screen(food.x, food.y),
                    4
                )

        # step agents
        for agent in population:
            if not agent.alive:
                continue

            step_agent(agent, foods, cfg)
            agent.trail.append((agent.x, agent.y))

            draw_vision_cone(screen, agent, cfg)

            # draw trail
            if len(agent.trail) > 1:
                pts = [world_to_screen(x, y) for x, y in agent.trail]
                pygame.draw.lines(screen, (120, 120, 120), False, pts, 1)

            # draw agent
            x, y = world_to_screen(agent.x, agent.y)
            pygame.draw.circle(screen, (0, 150, 255), (x, y), 7)

            # facing direction
            dx = math.cos(agent.angle) * 12
            dy = math.sin(agent.angle) * 12
            pygame.draw.line(
                screen, (255, 255, 255),
                (x, y), (x + dx, y + dy), 1
            )

        # world update once per timestep
        respawn_food(foods, cfg)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

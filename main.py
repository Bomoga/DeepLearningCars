"""
DeepLearningCars — NEAT self-driving car simulation
----------------------------------------------------
Usage:
    python main.py              # train for 50 generations
    python main.py --replay     # replay the saved winner (winner.pkl)
    python main.py --gen 100    # train for 100 generations
"""

import os
import sys
import pickle
import argparse

import pygame
import neat

from src.track      import Track
from src.renderer   import Renderer
from src.simulation import make_eval_genomes, START_X, START_Y, START_ANGLE
from src.car        import Car

WIDTH, HEIGHT = 1200, 800
CONFIG_PATH   = os.path.join(os.path.dirname(__file__), "config", "neat_config.txt")
WINNER_PATH   = os.path.join(os.path.dirname(__file__), "winner.pkl")


# ---------------------------------------------------------------------------
# Pygame initialisation (shared across training and replay)
# ---------------------------------------------------------------------------

def init_pygame() -> tuple[pygame.Surface, pygame.time.Clock]:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DeepLearningCars — NEAT")
    clock  = pygame.time.Clock()
    return screen, clock


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(generations: int) -> None:
    screen, clock = init_pygame()
    track    = Track()
    renderer = Renderer(screen)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(
        generation_interval=5,
        filename_prefix="neat-checkpoint-",
    ))

    eval_genomes = make_eval_genomes(screen, track, renderer, clock)
    winner = population.run(eval_genomes, generations)

    print(f"\nBest genome:\n{winner}")
    with open(WINNER_PATH, "wb") as f:
        pickle.dump(winner, f)
    print(f"Winner saved to {WINNER_PATH}")

    pygame.quit()


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay() -> None:
    if not os.path.exists(WINNER_PATH):
        print(f"No winner found at {WINNER_PATH}. Train first.")
        sys.exit(1)

    with open(WINNER_PATH, "rb") as f:
        winner = pickle.load(f)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )

    screen, clock = init_pygame()
    track    = Track()
    renderer = Renderer(screen)
    renderer.debug_rays = True  # always show rays in replay

    net = neat.nn.FeedForwardNetwork.create(winner, config)
    car = Car(START_X, START_Y, START_ANGLE)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    renderer.debug_rays = not renderer.debug_rays
                elif event.key == pygame.K_r:
                    car = Car(START_X, START_Y, START_ANGLE)

        if car.alive:
            readings = car.get_sensor_readings(track)
            output   = net.activate(readings)
            car.update(output[0], output[1])
            if track.is_colliding(car.position.x, car.position.y):
                car.alive = False

        renderer.draw(track, [car], generation=0, best_fitness=car.fitness)
        pygame.display.set_caption(
            f"DeepLearningCars — REPLAY  |  fitness: {car.fitness:.0f}"
            + ("  [DEAD — press R to reset]" if not car.alive else "")
        )
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DeepLearningCars NEAT simulation")
    parser.add_argument(
        "--replay", action="store_true",
        help="Replay the saved winner genome instead of training",
    )
    parser.add_argument(
        "--gen", type=int, default=50,
        help="Number of generations to train (default: 50)",
    )
    args = parser.parse_args()

    if args.replay:
        replay()
    else:
        train(args.gen)


if __name__ == "__main__":
    main()

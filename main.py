
import os
import sys
import argparse

import pygame

from src.track       import Track
from src.renderer    import Renderer
from src.car         import Car
from src.model.agent import DQNAgent, ACTIONS
from src.model.train import CHECKPOINTS, START_X, START_Y, START_ANGLE, MAX_FRAMES

WIDTH, HEIGHT   = 1200, 800
WEIGHTS_PATH    = os.path.join(os.path.dirname(__file__), "src", "model", "dqn_weights.pth")


def init_pygame() -> tuple[pygame.Surface, pygame.time.Clock]:
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DeepLearningCars — DQN")
    clock  = pygame.time.Clock()
    return screen, clock


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(num_episodes: int) -> None:
    screen, clock = init_pygame()
    track    = Track()
    renderer = Renderer(screen)
    agent    = DQNAgent()

    for episode in range(num_episodes):
        car             = Car(START_X, START_Y, START_ANGLE)
        state           = car.get_sensor_readings(track)
        total_reward    = 0.0
        next_checkpoint = 0

        for frame in range(MAX_FRAMES):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d:
                        renderer.debug_rays = not renderer.debug_rays

            action_idx      = agent.select_action(state)
            steer, throttle = ACTIONS[action_idx]
            prev_fitness    = car.fitness

            car.update(steer, throttle)

            done = track.is_colliding(car.position.x, car.position.y)
            if done:
                reward = -100.0
            else:
                reward = car.fitness - prev_fitness
                if next_checkpoint < len(CHECKPOINTS):
                    cx, cy, r = CHECKPOINTS[next_checkpoint]
                    if car.position.distance_to((cx, cy)) < r:
                        reward += 200.0
                        next_checkpoint += 1

            next_state = car.get_sensor_readings(track)
            agent.store(state, action_idx, reward, next_state, float(done))
            agent.train_step()

            state        = next_state
            total_reward += reward

            renderer.draw(track, [car], episode, total_reward)
            pygame.display.set_caption(
                f"DQN | Episode: {episode} | Reward: {total_reward:.1f} | Epsilon: {agent.epsilon:.3f}"
            )
            pygame.display.flip()
            clock.tick(60)

            if done:
                break

        agent.decay_epsilon()
        print(f"Episode {episode:4d} | Reward: {total_reward:8.1f} | Epsilon: {agent.epsilon:.3f}")

        if episode % 50 == 0:
            agent.save(WEIGHTS_PATH)
            print(f"  Model saved to {WEIGHTS_PATH}")

    pygame.quit()


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay() -> None:
    if not os.path.exists(WEIGHTS_PATH):
        print(f"No model found at {WEIGHTS_PATH}. Train first.")
        sys.exit(1)

    screen, clock = init_pygame()
    track    = Track()
    renderer = Renderer(screen)
    renderer.debug_rays = True

    agent = DQNAgent()
    agent.load(WEIGHTS_PATH)
    agent.epsilon = 0.0  # no exploration during replay

    car = Car(START_X, START_Y, START_ANGLE)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    renderer.debug_rays = not renderer.debug_rays
                elif event.key == pygame.K_r:
                    car = Car(START_X, START_Y, START_ANGLE)

        if car.alive:
            state      = car.get_sensor_readings(track)
            action_idx = agent.select_action(state)
            steer, throttle = ACTIONS[action_idx]
            car.update(steer, throttle)
            if track.is_colliding(car.position.x, car.position.y):
                car.alive = False

        renderer.draw(track, [car], generation=0, best_fitness=car.fitness)
        pygame.display.set_caption(
            f"DQN REPLAY | Fitness: {car.fitness:.0f}"
            + ("  [DEAD — press R to reset]" if not car.alive else "")
        )
        pygame.display.flip()
        clock.tick(60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DeepLearningCars DQN simulation")
    parser.add_argument("--replay", action="store_true", help="Replay saved model")
    parser.add_argument("--ep", type=int, default=500, help="Number of training episodes (default: 500)")
    args = parser.parse_args()

    if args.replay:
        replay()
    else:
        train(args.ep)


if __name__ == "__main__":
    main()
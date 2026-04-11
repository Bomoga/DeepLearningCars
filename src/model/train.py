import sys
import math
import pygame

from src.car         import Car
from src.track       import Track
from src.renderer    import Renderer
from src.model.agent import DQNAgent, ACTIONS

START_X, START_Y, START_ANGLE = 1015, 400, 90
MAX_FRAMES   = 600   # ~10 seconds per episode
NUM_EPISODES = 200   # small for baseline

def _make_checkpoints(n=16):
    cx, cy, rx, ry = 600, 400, 415, 275
    pts = []
    for i in range(n):
        a = math.radians(360 * i / n)
        x = int(cx + rx * math.cos(a))
        y = int(cy - ry * math.sin(a))
        pts.append((x, y, 55))
    return pts

CHECKPOINTS = _make_checkpoints()


def run():
    pygame.init()
    screen   = pygame.display.set_mode((1200, 800))
    clock    = pygame.time.Clock()
    track    = Track()
    renderer = Renderer(screen)
    agent    = DQNAgent()

    for episode in range(NUM_EPISODES):
        car             = Car(START_X, START_Y, START_ANGLE)
        state           = car.get_sensor_readings(track)
        total_reward    = 0.0
        next_checkpoint = 0

        for _ in range(MAX_FRAMES):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

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
            pygame.display.flip()
            clock.tick(60)

            if done:
                break

        agent.decay_epsilon()
        print(f"Episode {episode:4d} | Reward: {total_reward:8.1f} | Epsilon: {agent.epsilon:.3f}")

        if episode % 50 == 0:
            agent.save()

    pygame.quit()


if __name__ == "__main__":
    run()
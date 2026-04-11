from src.car import Car
from src.track import Track
from src.renderer import Renderer
from src.model.agent import DQNAgent, ACTIONS

START_X, START_Y, START_ANGLE = 1015, 400, 90
MAX_FRAMES   = 1200
NUM_EPISODES = 1000

# Checkpoint reward (same positions as your NEAT version)
import math
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


def compute_reward(car, genome_dummy, track, prev_fitness) -> float:
    """Reward = forward progress + checkpoint bonuses - collision penalty."""
    reward = car.fitness - prev_fitness   # velocity-based progress
    return reward


def run():
    pygame.init()
    screen   = pygame.display.set_mode((1200, 800))
    clock    = pygame.time.Clock()
    track    = Track()        # your existing track class
    renderer = Renderer(screen)     # your existing renderer

    agent    = DQNAgent()

    for episode in range(NUM_EPISODES):
        car           = Car(START_X, START_Y, START_ANGLE)
        state         = car.get_sensor_readings(track)
        total_reward  = 0.0
        next_checkpoint = 0

        for frame in range(MAX_FRAMES):
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Agent picks action
            action_idx        = agent.select_action(state)
            steer, throttle   = ACTIONS[action_idx]
            prev_fitness      = car.fitness

            # Step simulation
            car.update(steer, throttle)

            # Collision check
            done = track.is_colliding(car.position.x, car.position.y)
            if done:
                reward = -100.0   # big penalty for crashing
            else:
                reward = car.fitness - prev_fitness  # reward forward motion

                # Checkpoint bonus
                if next_checkpoint < len(CHECKPOINTS):
                    cx, cy, r = CHECKPOINTS[next_checkpoint]
                    if car.position.distance_to((cx, cy)) < r:
                        reward += 200.0
                        next_checkpoint += 1

            next_state = car.get_sensor_readings(track)

            # Store and train
            agent.store(state, action_idx, reward, next_state, float(done))
            agent.train_step()

            state        = next_state
            total_reward += reward

            # Render
            renderer.draw(track, [car], episode, total_reward)
            pygame.display.flip()
            clock.tick(60)

            if done:
                break

        agent.decay_epsilon()
        print(f"Episode {episode:4d} | Reward: {total_reward:8.1f} | Epsilon: {agent.epsilon:.3f}")

        # Save every 50 episodes
        if episode % 50 == 0:
            agent.save()

    pygame.quit()


if __name__ == "__main__":
    run()
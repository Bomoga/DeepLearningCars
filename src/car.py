import math
import os
import pygame

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

# Physics constants
MAX_SPEED       = 8.0    # pixels per frame
ACCELERATION    = 0.3    # px/frame² per unit throttle input
DECELERATION    = 0.15   # natural friction per frame
MAX_STEER_DEG   = 5.0    # max heading change per frame (degrees)

# Sensor ray offsets relative to the car's heading
SENSOR_ANGLES = [-90, -60, -30, 0, 30, 60, 90]

_car_img_cache: pygame.Surface | None = None


def _load_car_image() -> pygame.Surface:
    global _car_img_cache
    if _car_img_cache is None:
        path = os.path.join(ASSETS_DIR, "car.png")
        _car_img_cache = pygame.image.load(path).convert_alpha()
    return _car_img_cache


class Car:
    def __init__(self, x: float, y: float, angle: float = 0.0):
        """
        Parameters
        ----------
        x, y  : starting world position (pixels)
        angle : starting heading in degrees (0 = right, CCW positive)
        """
        self.position = pygame.Vector2(x, y)
        self.angle    = angle          # heading, degrees, CCW from +x axis
        self.velocity = 2.0            # start moving slowly so cars explore
        self.alive    = True
        self.fitness  = 0.0

        self._img     = _load_car_image()
        self._sensor_readings: list[float] = [1.0] * len(SENSOR_ANGLES)

        # Checkpoint tracking (set externally by simulation)
        self.next_checkpoint: int = 0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, steer: float, throttle: float) -> None:
        """
        Advance the car by one simulation frame.

        Parameters
        ----------
        steer    : [-1, 1]  — negative = left, positive = right
        throttle : [-1, 1]  — negative slows (no reverse), positive accelerates
        """
        # Steering
        self.angle += steer * MAX_STEER_DEG

        # Throttle / friction
        self.velocity += throttle * ACCELERATION - DECELERATION
        self.velocity = max(0.0, min(self.velocity, MAX_SPEED))

        # Move
        rad = math.radians(self.angle)
        self.position.x += math.cos(rad) * self.velocity
        self.position.y -= math.sin(rad) * self.velocity  # screen y inverted

        # Fitness: reward forward motion every frame
        self.fitness += self.velocity

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------

    def get_sensor_readings(self, track) -> list[float]:
        """Return 7 normalised ray distances as NEAT network inputs."""
        readings = []
        for offset in SENSOR_ANGLES:
            world_angle = self.angle + offset
            readings.append(track.cast_ray(self.position, world_angle))
        self._sensor_readings = readings
        return readings

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, screen: pygame.Surface, debug_rays: bool = False, track=None) -> None:
        # Rotate sprite (pygame rotates CCW, our angle is CCW from +x)
        rotated = pygame.transform.rotate(self._img, self.angle)
        rect    = rotated.get_rect(center=(int(self.position.x), int(self.position.y)))
        screen.blit(rotated, rect)

        if debug_rays and track is not None:
            self._draw_rays(screen, track)

    def _draw_rays(self, screen: pygame.Surface, track) -> None:
        for i, offset in enumerate(SENSOR_ANGLES):
            norm = self._sensor_readings[i]
            world_angle = self.angle + offset
            end = track.get_ray_endpoint(self.position, world_angle, norm)

            # Colour: green = clear, yellow = mid, red = close
            if norm > 0.7:
                colour = (0, 200, 0)
            elif norm > 0.3:
                colour = (220, 220, 0)
            else:
                colour = (220, 0, 0)

            pygame.draw.line(
                screen, colour,
                (int(self.position.x), int(self.position.y)),
                (int(end.x), int(end.y)),
                1,
            )

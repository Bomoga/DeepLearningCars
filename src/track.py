import math
import os
import pygame

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")


class Track:
    """Wraps the racetrack images and exposes collision / raycasting queries."""

    RAY_STEP = 3          # pixels per ray-march step
    MAX_RAY_LENGTH = 200  # pixels

    def __init__(self):
        track_path = os.path.join(ASSETS_DIR, "track.png")
        mask_path  = os.path.join(ASSETS_DIR, "track_mask.png")

        self.surface = pygame.image.load(track_path).convert()
        self._mask   = pygame.image.load(mask_path).convert()

        self.width  = self.surface.get_width()
        self.height = self.surface.get_height()

    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------

    def is_colliding(self, x: float, y: float) -> bool:
        """Return True when (x, y) is outside the road (black mask pixel)."""
        ix, iy = int(x), int(y)
        if ix < 0 or iy < 0 or ix >= self.width or iy >= self.height:
            return True
        r, g, b, *_ = self._mask.get_at((ix, iy))
        # White = road (r > 128), black = wall
        return r < 128

    # ------------------------------------------------------------------
    # Raycasting
    # ------------------------------------------------------------------

    def cast_ray(self, origin: pygame.Vector2, angle_deg: float) -> float:
        """
        March a ray from *origin* at *angle_deg* (world space, 0° = right).
        Returns normalised distance [0.0, 1.0] to the first wall hit.
        1.0 means the ray reached MAX_RAY_LENGTH without hitting a wall.
        """
        rad = math.radians(angle_deg)
        dx  = math.cos(rad) * self.RAY_STEP
        dy  = -math.sin(rad) * self.RAY_STEP   # screen y is inverted

        x, y = origin.x, origin.y
        dist = 0.0

        while dist < self.MAX_RAY_LENGTH:
            x += dx
            y += dy
            dist += self.RAY_STEP
            if self.is_colliding(x, y):
                break

        return dist / self.MAX_RAY_LENGTH

    def get_ray_endpoint(
        self, origin: pygame.Vector2, angle_deg: float, norm_dist: float
    ) -> pygame.Vector2:
        """Convert a normalised distance back to a world-space endpoint."""
        rad     = math.radians(angle_deg)
        raw     = norm_dist * self.MAX_RAY_LENGTH
        return pygame.Vector2(
            origin.x + math.cos(rad) * raw,
            origin.y - math.sin(rad) * raw,
        )

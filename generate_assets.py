"""
One-off script to generate track.png, track_mask.png, and car.png.
Run once: python generate_assets.py
"""

import pygame
import sys
import math
import os

WIDTH, HEIGHT = 1200, 800
TRACK_COLOR = (80, 80, 80)        # dark grey asphalt
BORDER_COLOR = (255, 255, 0)      # yellow kerbing lines
GRASS_COLOR = (34, 139, 34)       # green grass background
ROAD_WHITE = (255, 255, 255)      # road on mask
WALL_BLACK = (0, 0, 0)            # wall on mask
TRACK_WIDTH = 110                 # pixels wide


def build_track_path():
    """
    Returns a list of (x, y) points that trace the centreline of a
    figure-of-eight-free, smoothed racetrack loop.  The path is built
    from a small set of control points and then interpolated so that
    pygame can fill it as a polygon.
    """
    # Control points for the outer and inner ovals, combined into a
    # single closed loop with varying curvature so the cars face a real
    # driving challenge (not just a plain oval).
    cx, cy = WIDTH // 2, HEIGHT // 2

    # Outer loop – a rounded rectangle with asymmetric corners
    outer = []
    # We'll use an ellipse-based approximation with 4 arcs
    points = [
        # centre, rx, ry, start_angle_deg, end_angle_deg
        (cx - 300, cy - 150, 180, 120, 90, 270),   # left arc
        (cx + 300, cy - 150, 180, 120, 270, 450),  # right arc (=270..90)
    ]

    def arc_points(cx, cy, rx, ry, a0, a1, n=30):
        pts = []
        for i in range(n + 1):
            t = math.radians(a0 + (a1 - a0) * i / n)
            pts.append((int(cx + rx * math.cos(t)),
                        int(cy + ry * math.sin(t))))
        return pts

    # Build a proper loop:  think of a stadium / rounded-rectangle shape
    # but with some kinks on the straight sections to add chicanes.
    #
    # Outer boundary (clockwise)
    outer = (
        arc_points(cx - 330, cy, 160, 260, 90, 270)        # left hairpin
        + arc_points(cx + 330, cy, 160, 260, 270, 450)     # right hairpin
    )

    # Straight kinks (chicane) on the bottom straight
    chicane_offset = 60
    bottom_left  = (cx - 330 + 0,  cy + 260)
    bottom_right = (cx + 330 + 0,  cy + 260)
    outer_bottom = [
        (cx - 180, cy + 260),
        (cx - 80,  cy + 260 - chicane_offset),
        (cx,       cy + 260),
        (cx + 80,  cy + 260 - chicane_offset),
        (cx + 180, cy + 260),
    ]

    inner = (
        arc_points(cx - 330, cy, 60, 150, 90, 270)
        + arc_points(cx + 330, cy, 60, 150, 270, 450)
    )
    inner_bottom = [
        (cx - 120, cy + 150),
        (cx - 40,  cy + 150 - chicane_offset),
        (cx,       cy + 150),
        (cx + 40,  cy + 150 - chicane_offset),
        (cx + 120, cy + 150),
    ]

    return outer, inner


def draw_track(surface, mask_surface):
    cx, cy = WIDTH // 2, HEIGHT // 2

    def arc_points(acx, acy, rx, ry, a0, a1, n=60):
        pts = []
        for i in range(n + 1):
            t = math.radians(a0 + (a1 - a0) * i / n)
            pts.append((int(acx + rx * math.cos(t)),
                        int(acy + ry * math.sin(t))))
        return pts

    # --- Outer boundary of the track ---
    outer_rx, outer_ry = 470, 330
    inner_rx, inner_ry = 360, 220

    outer_pts = arc_points(cx, cy, outer_rx, outer_ry, 0, 360, n=120)
    inner_pts = arc_points(cx, cy, inner_rx, inner_ry, 0, 360, n=120)

    # Fill grass everywhere first
    surface.fill(GRASS_COLOR)
    mask_surface.fill(WALL_BLACK)

    # Draw track ring: outer filled ellipse, then inner ellipse cuts it out
    pygame.draw.polygon(surface, TRACK_COLOR, outer_pts)
    pygame.draw.polygon(surface, GRASS_COLOR, inner_pts)

    # Mask: white where track is
    pygame.draw.polygon(mask_surface, ROAD_WHITE, outer_pts)
    pygame.draw.polygon(mask_surface, WALL_BLACK, inner_pts)

    # Yellow kerb lines (dashed effect on borders)
    pygame.draw.lines(surface, BORDER_COLOR, True, outer_pts, 4)
    pygame.draw.lines(surface, BORDER_COLOR, True, inner_pts, 4)

    # Dashed start/finish line
    start_x = cx + outer_rx - (outer_rx - inner_rx) // 2
    for y in range(cy - 30, cy + 30, 10):
        color = (255, 255, 255) if (y // 10) % 2 == 0 else (0, 0, 0)
        pygame.draw.rect(surface, color, (start_x - 5, y, 10, 10))


def build_car_surface():
    """Create a small top-down car sprite (40x20 px)."""
    car = pygame.Surface((40, 20), pygame.SRCALPHA)
    # Body
    pygame.draw.rect(car, (220, 50, 50), (2, 2, 36, 16), border_radius=5)
    # Windshield
    pygame.draw.rect(car, (180, 220, 255), (20, 4, 14, 12), border_radius=3)
    # Wheels
    wheel_color = (30, 30, 30)
    pygame.draw.rect(car, wheel_color, (2, 1, 8, 4))
    pygame.draw.rect(car, wheel_color, (2, 15, 8, 4))
    pygame.draw.rect(car, wheel_color, (30, 1, 8, 4))
    pygame.draw.rect(car, wheel_color, (30, 15, 8, 4))
    return car


def main():
    pygame.init()
    surface = pygame.Surface((WIDTH, HEIGHT))
    mask_surface = pygame.Surface((WIDTH, HEIGHT))

    draw_track(surface, mask_surface)

    out_dir = os.path.join(os.path.dirname(__file__), "assets")
    os.makedirs(out_dir, exist_ok=True)

    pygame.image.save(surface, os.path.join(out_dir, "track.png"))
    pygame.image.save(mask_surface, os.path.join(out_dir, "track_mask.png"))

    car_surf = build_car_surface()
    pygame.image.save(car_surf, os.path.join(out_dir, "car.png"))

    pygame.quit()
    print("Assets saved to assets/")


if __name__ == "__main__":
    main()

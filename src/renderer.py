import pygame

HUD_COLOR    = (255, 255, 255)
DEAD_ALPHA   = 80   # 0-255


class Renderer:
    def __init__(self, screen: pygame.Surface):
        self._screen = screen
        self._font   = pygame.font.SysFont("monospace", 20)
        self.debug_rays = False   # toggled by D key

    # ------------------------------------------------------------------
    # Main draw call — invoke once per frame inside eval_genomes
    # ------------------------------------------------------------------

    def draw(self, track, cars, generation: int, best_fitness: float) -> None:
        # 1. Background track
        self._screen.blit(track.surface, (0, 0))

        alive_count = 0
        for car in cars:
            if car.alive:
                alive_count += 1
                car.draw(self._screen, debug_rays=self.debug_rays, track=track)
            else:
                self._draw_dead_car(car)

        # 2. HUD (drawn last so it's always on top)
        self._draw_hud(generation, alive_count, len(cars), best_fitness)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _draw_dead_car(self, car) -> None:
        """Render a dead car as a grey, semi-transparent blot."""
        import math
        from src.car import _load_car_image
        img  = _load_car_image()
        grey = pygame.transform.grayscale(img)
        rotated = pygame.transform.rotate(grey, car.angle)
        rotated.set_alpha(DEAD_ALPHA)
        rect = rotated.get_rect(center=(int(car.position.x), int(car.position.y)))
        self._screen.blit(rotated, rect)

    def _draw_hud(
        self, generation: int, alive: int, total: int, best_fitness: float
    ) -> None:
        lines = [
            f"Generation : {generation}",
            f"Alive      : {alive} / {total}",
            f"Best fit   : {best_fitness:.1f}",
            f"[D] rays   : {'ON' if self.debug_rays else 'OFF'}",
        ]
        for i, text in enumerate(lines):
            surf = self._font.render(text, True, HUD_COLOR)
            # Thin black shadow for readability on any background
            shadow = self._font.render(text, True, (0, 0, 0))
            self._screen.blit(shadow, (11, 11 + i * 26))
            self._screen.blit(surf,   (10, 10 + i * 26))

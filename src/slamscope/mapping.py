"""Occupancy grid mapping with an inverse sensor model.

Cells hold log odds rather than probabilities, so evidence accumulates by
addition and repeated observations of the same wall reinforce each other
without ever saturating at exactly 0 or 1.

The map is built from the *estimated* pose, not the true one. That is the
honest version: feed it dead reckoning and the map smears as odometry drifts;
feed it the MCL estimate and it stays sharp. Watching those two side by side
is the point of the whole project.
"""

from __future__ import annotations

import math

from .world import FREE, OCCUPIED, OccupancyMap, Pose

L_FREE = -0.45
L_OCC = 0.95
L_MIN = -6.0
L_MAX = 6.0


class LogOddsMap:
    def __init__(self, width: int, height: int, resolution: float = 0.25) -> None:
        self.width = width
        self.height = height
        self.resolution = resolution
        self.log_odds = [0.0] * (width * height)
        self.updates = 0

    # ------------------------------------------------------------ accessors

    def in_bounds(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.width and 0 <= cy < self.height

    def probability(self, cx: int, cy: int) -> float:
        if not self.in_bounds(cx, cy):
            return 0.5
        lo = self.log_odds[cy * self.width + cx]
        return 1.0 - 1.0 / (1.0 + math.exp(lo))

    def _bump(self, cx: int, cy: int, delta: float) -> None:
        if not self.in_bounds(cx, cy):
            return
        i = cy * self.width + cx
        self.log_odds[i] = max(L_MIN, min(L_MAX, self.log_odds[i] + delta))

    # -------------------------------------------------------------- updates

    def integrate(
        self,
        pose: Pose,
        scan: list[float],
        angles: list[float],
        max_range: float,
        clear_on_no_return: bool = False,
    ) -> None:
        """Fold one scan into the map from the given pose.

        Beams that come back at max range are skipped by default. A max-range
        reading means "no return", which is ambiguous: it might be open space,
        or the beam might have been absorbed or hit at a glancing angle. Tracing
        free space along them punched holes straight through walls -- it was
        responsible for 625 of 630 mapping errors in the maze, marking real wall
        cells as free.
        """
        res = self.resolution
        ox, oy = pose.x, pose.y
        for angle, r in zip(angles, scan, strict=False):
            beam = pose.theta + angle
            hit = r < max_range - 1e-6
            if not hit and not clear_on_no_return:
                continue
            # Free space is only claimed up to just short of the endpoint.
            free_len = r - res * 0.5 if hit else r
            self._trace_free(ox, oy, beam, free_len)
            if hit:
                self._bump(*self._cell(ox + r * math.cos(beam), oy + r * math.sin(beam)), L_OCC)
        self.updates += 1

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(x / self.resolution), int(y / self.resolution))

    def _trace_free(self, ox: float, oy: float, theta: float, length: float) -> None:
        """Mark cells along the beam as free, using the same DDA as raycasting."""
        if length <= 0:
            return
        res = self.resolution
        cx, cy = self._cell(ox, oy)
        dx, dy = math.cos(theta), math.sin(theta)
        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1

        if abs(dx) < 1e-12:
            t_max_x = t_delta_x = math.inf
        else:
            t_max_x = ((cx + (1 if dx > 0 else 0)) * res - ox) / dx
            t_delta_x = res / abs(dx)
        if abs(dy) < 1e-12:
            t_max_y = t_delta_y = math.inf
        else:
            t_max_y = ((cy + (1 if dy > 0 else 0)) * res - oy) / dy
            t_delta_y = res / abs(dy)

        travelled = 0.0
        self._bump(cx, cy, L_FREE)
        while travelled < length:
            if t_max_x < t_max_y:
                cx += step_x
                travelled = t_max_x
                t_max_x += t_delta_x
            else:
                cy += step_y
                travelled = t_max_y
                t_max_y += t_delta_y
            if not self.in_bounds(cx, cy):
                return
            if travelled < length:
                self._bump(cx, cy, L_FREE)

    # ------------------------------------------------------------- accuracy

    def to_binary(self, threshold: float = 0.5) -> OccupancyMap:
        out = OccupancyMap(self.width, self.height, self.resolution)
        for cy in range(self.height):
            for cx in range(self.width):
                if self.probability(cx, cy) > threshold:
                    out.set(cx, cy, OCCUPIED)
        return out

    def agreement(self, truth: OccupancyMap, min_confidence: float = 0.1) -> tuple[float, int]:
        """Fraction of *decided* cells that match ground truth, and how many.

        Cells never observed sit at probability 0.5 and are excluded -- scoring
        them would just reward a map for being mostly unexplored.
        """
        correct = decided = 0
        for cy in range(min(self.height, truth.height)):
            for cx in range(min(self.width, truth.width)):
                p = self.probability(cx, cy)
                if abs(p - 0.5) < min_confidence:
                    continue
                decided += 1
                predicted = OCCUPIED if p > 0.5 else FREE
                if predicted == truth.get(cx, cy):
                    correct += 1
        return (correct / decided if decided else 0.0, decided)

    def explored_fraction(self, min_confidence: float = 0.1) -> float:
        decided = sum(
            1 for i in range(len(self.log_odds))
            if abs(1.0 - 1.0 / (1.0 + math.exp(self.log_odds[i])) - 0.5) >= min_confidence
        )
        return decided / len(self.log_odds)

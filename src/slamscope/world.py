"""World geometry: poses, the occupancy map, and raycasting.

The map is a coarse occupancy grid in world units (metres). Raycasting uses
a DDA grid traversal rather than fixed-step sampling, because fixed steps
either miss thin walls or waste time oversampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

FREE = 0
OCCUPIED = 1


def wrap_angle(theta: float) -> float:
    """Fold an angle into (-pi, pi]. Every angle difference must go through this."""
    return math.atan2(math.sin(theta), math.cos(theta))


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    theta: float

    def __iter__(self):
        return iter((self.x, self.y, self.theta))

    def relative_to(self, other: Pose) -> tuple[float, float, float]:
        """This pose expressed in `other`'s frame."""
        dx, dy = self.x - other.x, self.y - other.y
        c, s = math.cos(-other.theta), math.sin(-other.theta)
        return (dx * c - dy * s, dx * s + dy * c, wrap_angle(self.theta - other.theta))

    def distance_to(self, other: Pose) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


class OccupancyMap:
    """A binary occupancy grid with a metric resolution."""

    def __init__(self, width: int, height: int, resolution: float = 0.25) -> None:
        self.width = width
        self.height = height
        self.resolution = resolution
        self.cells = bytearray(width * height)

    # ------------------------------------------------------------ accessors

    def index(self, cx: int, cy: int) -> int:
        return cy * self.width + cx

    def in_bounds(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.width and 0 <= cy < self.height

    def get(self, cx: int, cy: int) -> int:
        if not self.in_bounds(cx, cy):
            return OCCUPIED  # outside the map counts as wall
        return self.cells[self.index(cx, cy)]

    def set(self, cx: int, cy: int, value: int) -> None:
        if self.in_bounds(cx, cy):
            self.cells[self.index(cx, cy)] = value

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(x / self.resolution), int(y / self.resolution))

    def cell_to_world(self, cx: int, cy: int) -> tuple[float, float]:
        """Centre of the cell, in metres."""
        return ((cx + 0.5) * self.resolution, (cy + 0.5) * self.resolution)

    def is_occupied_world(self, x: float, y: float) -> bool:
        return self.get(*self.world_to_cell(x, y)) == OCCUPIED

    @property
    def extent(self) -> tuple[float, float]:
        return (self.width * self.resolution, self.height * self.resolution)

    # ---------------------------------------------------------- raycasting

    def raycast(self, x: float, y: float, theta: float, max_range: float) -> float:
        """Distance from (x, y) along theta to the first occupied cell.

        Amanatides-Woo DDA traversal: step cell by cell along whichever axis
        boundary comes first. Exact, and no thin wall can be stepped over.
        """
        res = self.resolution
        cx, cy = self.world_to_cell(x, y)
        if self.get(cx, cy) == OCCUPIED:
            return 0.0

        dx, dy = math.cos(theta), math.sin(theta)
        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1

        # Distance along the ray to the next cell boundary in each axis.
        if abs(dx) < 1e-12:
            t_max_x, t_delta_x = math.inf, math.inf
        else:
            boundary = (cx + (1 if dx > 0 else 0)) * res
            t_max_x = (boundary - x) / dx
            t_delta_x = res / abs(dx)

        if abs(dy) < 1e-12:
            t_max_y, t_delta_y = math.inf, math.inf
        else:
            boundary = (cy + (1 if dy > 0 else 0)) * res
            t_max_y = (boundary - y) / dy
            t_delta_y = res / abs(dy)

        travelled = 0.0
        while travelled <= max_range:
            if t_max_x < t_max_y:
                cx += step_x
                travelled = t_max_x
                t_max_x += t_delta_x
            else:
                cy += step_y
                travelled = t_max_y
                t_max_y += t_delta_y

            if not self.in_bounds(cx, cy):
                return min(travelled, max_range)
            if self.cells[self.index(cx, cy)] == OCCUPIED:
                return min(travelled, max_range)

        return max_range

    def free_cells(self) -> list[tuple[int, int]]:
        return [
            (cx, cy)
            for cy in range(self.height)
            for cx in range(self.width)
            if self.cells[self.index(cx, cy)] == FREE
        ]

    def copy(self) -> OccupancyMap:
        m = OccupancyMap(self.width, self.height, self.resolution)
        m.cells = bytearray(self.cells)
        return m

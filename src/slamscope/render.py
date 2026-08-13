"""Rendering.

Everything paints into a flat array of colours, one entry per map cell, and is
converted to half-block characters at the end: one terminal cell shows two map
cells stacked, top as foreground, bottom as background. Painting first and
converting once keeps layer ordering obvious -- later paints simply overwrite.
"""

from __future__ import annotations

import math

from rich.text import Text

from .localization import ParticleFilter
from .mapping import LogOddsMap
from .world import OCCUPIED, OccupancyMap, Pose

HALF = "\u2580"

BG = "#0b0e14"
WALL = "#5b6577"
FREE_SPACE = "#141922"
UNKNOWN = "#1b2029"

TRUE_TRAIL = "#22c55e"
ODOM_TRAIL = "#ef4444"
EST_TRAIL = "#38bdf8"
ROBOT = "#fde047"
BEAM = "#334155"
PARTICLE = "#a78bfa"
PARTICLE_HOT = "#f0abfc"
ESTIMATE = "#38bdf8"

# Probability shading for the built map, darkest free to lightest occupied.
OCC_SHADES = ["#141922", "#242c3a", "#3a4456", "#5b6577", "#8b96a8", "#c3cbd8"]


class Painter:
    """A colour buffer the size of the map."""

    def __init__(self, width: int, height: int, resolution: float) -> None:
        self.width = width
        self.height = height
        self.resolution = resolution
        self.buf = [BG] * (width * height)

    def clear(self, colour: str = BG) -> None:
        self.buf = [colour] * (self.width * self.height)

    def cell(self, cx: int, cy: int, colour: str) -> None:
        if 0 <= cx < self.width and 0 <= cy < self.height:
            self.buf[cy * self.width + cx] = colour

    def point(self, x: float, y: float, colour: str) -> None:
        self.cell(int(x / self.resolution), int(y / self.resolution), colour)

    def line(self, x0: float, y0: float, x1: float, y1: float, colour: str) -> None:
        """Bresenham in cell space, used for lidar beams and trails."""
        cx0, cy0 = int(x0 / self.resolution), int(y0 / self.resolution)
        cx1, cy1 = int(x1 / self.resolution), int(y1 / self.resolution)
        dx, dy = abs(cx1 - cx0), -abs(cy1 - cy0)
        sx = 1 if cx0 < cx1 else -1
        sy = 1 if cy0 < cy1 else -1
        err = dx + dy
        while True:
            self.cell(cx0, cy0, colour)
            if cx0 == cx1 and cy0 == cy1:
                return
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                cx0 += sx
            if e2 <= dx:
                err += dx
                cy0 += sy

    def to_text(self) -> Text:
        text = Text()
        rows = (self.height + 1) // 2
        for row in range(rows):
            top_row = row * 2
            bottom_row = top_row + 1
            run: list[str] = []
            run_style: str | None = None
            for cx in range(self.width):
                top = self.buf[top_row * self.width + cx]
                bottom = (
                    self.buf[bottom_row * self.width + cx] if bottom_row < self.height else BG
                )
                style = f"{top} on {bottom}"
                if run_style is None:
                    run_style = style
                if style != run_style:
                    text.append("".join(run), style=run_style)
                    run, run_style = [], style
                run.append(HALF)
            if run:
                text.append("".join(run), style=run_style or "")
            if row != rows - 1:
                text.append("\n")
        return text


def paint_world(p: Painter, world: OccupancyMap) -> None:
    for cy in range(world.height):
        for cx in range(world.width):
            p.cell(cx, cy, WALL if world.get(cx, cy) == OCCUPIED else FREE_SPACE)


def paint_built_map(p: Painter, built: LogOddsMap) -> None:
    for cy in range(built.height):
        for cx in range(built.width):
            prob = built.probability(cx, cy)
            if abs(prob - 0.5) < 0.06:
                p.cell(cx, cy, UNKNOWN)
            else:
                idx = min(len(OCC_SHADES) - 1, int(prob * len(OCC_SHADES)))
                p.cell(cx, cy, OCC_SHADES[idx])


def paint_trail(p: Painter, trail: list[tuple[float, float]], colour: str, stride: int = 1) -> None:
    for x, y in trail[::stride]:
        p.point(x, y, colour)


def paint_beams(p: Painter, pose: Pose, scan: list[float], angles: list[float]) -> None:
    for angle, r in zip(angles, scan, strict=False):
        beam = pose.theta + angle
        p.line(pose.x, pose.y, pose.x + r * math.cos(beam), pose.y + r * math.sin(beam), BEAM)


def paint_robot(p: Painter, pose: Pose, colour: str = ROBOT, heading: float = 0.7) -> None:
    p.line(
        pose.x,
        pose.y,
        pose.x + heading * math.cos(pose.theta),
        pose.y + heading * math.sin(pose.theta),
        colour,
    )
    p.point(pose.x, pose.y, colour)


def paint_particles(p: Painter, pf: ParticleFilter) -> None:
    """Heavier particles are drawn brighter, and last, so they win overlaps."""
    if not pf.particles:
        return
    top = max(pt.weight for pt in pf.particles) or 1.0
    ordered = sorted(pf.particles, key=lambda pt: pt.weight)
    for particle in ordered:
        hot = particle.weight > top * 0.35
        p.point(particle.pose.x, particle.pose.y, PARTICLE_HOT if hot else PARTICLE)

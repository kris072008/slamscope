"""Environments to drive around in.

Symmetry is the enemy of global localization: in a plain rectangular room every
corner looks alike and the posterior stays multi-modal forever. Each of these
has enough asymmetric structure to be identifiable, except `symmetric`, which
is included precisely so you can watch the failure.
"""

from __future__ import annotations

import random

from .world import OCCUPIED, OccupancyMap, Pose


def _walled(width: int, height: int, resolution: float) -> OccupancyMap:
    m = OccupancyMap(width, height, resolution)
    for cx in range(width):
        m.set(cx, 0, OCCUPIED)
        m.set(cx, height - 1, OCCUPIED)
    for cy in range(height):
        m.set(0, cy, OCCUPIED)
        m.set(width - 1, cy, OCCUPIED)
    return m


def _hline(m: OccupancyMap, x0: int, x1: int, y: int) -> None:
    for x in range(x0, x1):
        m.set(x, y, OCCUPIED)


def _vline(m: OccupancyMap, x: int, y0: int, y1: int) -> None:
    for y in range(y0, y1):
        m.set(x, y, OCCUPIED)


def office(width: int = 72, height: int = 48, resolution: float = 0.25) -> OccupancyMap:
    """Rooms off a corridor. Distinctive enough to localize in quickly."""
    m = _walled(width, height, resolution)
    mid = height // 2
    _hline(m, 0, int(width * 0.30), mid - 4)
    _hline(m, int(width * 0.42), width, mid - 4)
    _hline(m, 0, int(width * 0.62), mid + 4)
    _hline(m, int(width * 0.74), width, mid + 4)
    _vline(m, int(width * 0.22), 1, mid - 4)
    _vline(m, int(width * 0.55), 1, mid - 8)
    _vline(m, int(width * 0.78), 1, mid - 4)
    _vline(m, int(width * 0.34), mid + 4, height - 1)
    _vline(m, int(width * 0.66), mid + 8, height - 1)
    return m


def maze(width: int = 72, height: int = 48, resolution: float = 0.25) -> OccupancyMap:
    """Recursive-backtracker maze at double cell scale, so corridors fit the robot."""
    m = OccupancyMap(width, height, resolution)
    for i in range(width * height):
        m.cells[i] = OCCUPIED

    # 5-cell corridors on a 6-cell pitch: 1.25 m wide at the default
    # resolution, which the reactive controller can actually drive through.
    # Narrower than about a metre and it spins in place forever.
    cw, ch = (width - 1) // 6, (height - 1) // 6

    def carve(cell_x: int, cell_y: int) -> None:
        stack = [(cell_x, cell_y)]
        visited = {(cell_x, cell_y)}
        _block(m, cell_x, cell_y)
        while stack:
            x, y = stack[-1]
            options = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < cw and 0 <= ny < ch and (nx, ny) not in visited:
                    options.append((nx, ny, dx, dy))
            if not options:
                stack.pop()
                continue
            nx, ny, dx, dy = random.choice(options)
            visited.add((nx, ny))
            _block(m, nx, ny)
            _corridor(m, x, y, dx, dy)
            stack.append((nx, ny))

    carve(0, 0)
    return m


def _block(m: OccupancyMap, cell_x: int, cell_y: int) -> None:
    bx, by = 1 + cell_x * 6, 1 + cell_y * 6
    for y in range(by, by + 5):
        for x in range(bx, bx + 5):
            m.set(x, y, 0)


def _corridor(m: OccupancyMap, x: int, y: int, dx: int, dy: int) -> None:
    bx, by = 1 + x * 6, 1 + y * 6
    if dx:
        gx = bx + 5 if dx > 0 else bx - 1
        for yy in range(by, by + 5):
            m.set(gx, yy, 0)
    else:
        gy = by + 5 if dy > 0 else by - 1
        for xx in range(bx, bx + 5):
            m.set(xx, gy, 0)


def cave(width: int = 72, height: int = 48, resolution: float = 0.25) -> OccupancyMap:
    """Cellular-automaton cave: organic walls, very few straight lines."""
    m = _walled(width, height, resolution)
    for cy in range(2, height - 2):
        for cx in range(2, width - 2):
            if random.random() < 0.42:
                m.set(cx, cy, OCCUPIED)

    for _ in range(4):
        snapshot = bytearray(m.cells)
        for cy in range(1, height - 1):
            for cx in range(1, width - 1):
                n = sum(
                    snapshot[(cy + dy) * width + (cx + dx)] == OCCUPIED
                    for dy in (-1, 0, 1)
                    for dx in (-1, 0, 1)
                    if (dx, dy) != (0, 0)
                )
                m.set(cx, cy, OCCUPIED if n >= 5 else 0)
    for cx in range(width):
        m.set(cx, 0, OCCUPIED)
        m.set(cx, height - 1, OCCUPIED)
    for cy in range(height):
        m.set(0, cy, OCCUPIED)
        m.set(width - 1, cy, OCCUPIED)
    return m


def pillars(width: int = 72, height: int = 48, resolution: float = 0.25) -> OccupancyMap:
    """Open hall with irregular pillars. Easy to drive, harder to localize."""
    m = _walled(width, height, resolution)
    random.seed(random.randrange(1 << 30))
    for _ in range(18):
        cx, cy = random.randrange(4, width - 4), random.randrange(4, height - 4)
        w, h = random.randint(1, 4), random.randint(1, 4)
        for y in range(cy, min(cy + h, height - 1)):
            for x in range(cx, min(cx + w, width - 1)):
                m.set(x, y, OCCUPIED)
    return m


def symmetric(width: int = 72, height: int = 48, resolution: float = 0.25) -> OccupancyMap:
    """Four-fold symmetric. Global localization should stay multi-modal here."""
    m = _walled(width, height, resolution)
    cx, cy = width // 2, height // 2
    for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        _vline(m, cx + dx * 10, min(cy, cy + dy * 12), max(cy, cy + dy * 12))
        _hline(m, min(cx, cx + dx * 16), max(cx, cx + dx * 16), cy + dy * 12)
    return m


ENVIRONMENTS = {
    "office": office,
    "maze": maze,
    "cave": cave,
    "pillars": pillars,
    "symmetric": symmetric,
}


def find_open_pose(world: OccupancyMap, clearance: float = 0.6) -> Pose:
    """A random free spot with room around it, so the robot doesn't start stuck."""
    candidates = []
    for cy in range(world.height):
        for cx in range(world.width):
            if world.get(cx, cy) != 0:
                continue
            x, y = world.cell_to_world(cx, cy)
            if min(
                world.raycast(x, y, a * 1.5708, clearance + 0.1) for a in range(4)
            ) >= clearance:
                candidates.append((x, y))
    if not candidates:
        return Pose(world.extent[0] / 2, world.extent[1] / 2, 0.0)
    x, y = random.choice(candidates)
    return Pose(x, y, random.uniform(-3.14159, 3.14159))

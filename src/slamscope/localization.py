"""Monte Carlo Localization.

The sensor model is the likelihood field (Thrun et al., Ch. 6.4) rather than
the beam model: for each scan endpoint, look up the distance to the nearest
obstacle and score it under a Gaussian. It needs one distance transform of the
map up front, and after that a scan costs one array lookup per beam instead of
one raycast per beam per particle -- with 400 particles and 24 beams that is
the difference between interactive and unusable.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .robot import Lidar, MotionNoise, sample_motion
from .world import OCCUPIED, OccupancyMap, Pose, wrap_angle


class LikelihoodField:
    """Euclidean distance to the nearest occupied cell, in metres."""

    def __init__(self, world: OccupancyMap) -> None:
        self.width, self.height = world.width, world.height
        self.resolution = world.resolution
        self.distance = self._distance_transform(world)
        self.max_distance = max(self.distance) if self.distance else 0.0

    def _distance_transform(self, world: OccupancyMap) -> list[float]:
        """Danielsson-style vector propagation: near-exact Euclidean, two passes.

        Each cell carries the offset to its nearest obstacle rather than a
        scalar, so diagonal information is not lost the way it is with a plain
        BFS (which would give Manhattan distance).
        """
        w, h = self.width, self.height
        big = float(w + h) * 2
        vx = [0.0] * (w * h)
        vy = [0.0] * (w * h)
        dist = [big] * (w * h)

        for cy in range(h):
            for cx in range(w):
                i = cy * w + cx
                if world.cells[i] == OCCUPIED:
                    dist[i] = 0.0
                    vx[i] = vy[i] = 0.0

        def relax(i: int, j: int, dx: int, dy: int) -> None:
            """Take j's nearest-obstacle vector, re-express it relative to i.

            (dx, dy) is the offset *from i to j*, so j's obstacle sits at
            v[j] + (dx, dy) when measured from i.
            """
            nx, ny = vx[j] + dx, vy[j] + dy
            d = math.hypot(nx, ny)
            if d < dist[i]:
                dist[i] = d
                vx[i], vy[i] = nx, ny

        # Forward pass: neighbours above and to the left.
        for cy in range(h):
            for cx in range(w):
                i = cy * w + cx
                if cy > 0:
                    relax(i, i - w, 0, -1)
                    if cx > 0:
                        relax(i, i - w - 1, -1, -1)
                    if cx < w - 1:
                        relax(i, i - w + 1, 1, -1)
                if cx > 0:
                    relax(i, i - 1, -1, 0)

        # Backward pass: neighbours below and to the right.
        for cy in range(h - 1, -1, -1):
            for cx in range(w - 1, -1, -1):
                i = cy * w + cx
                if cy < h - 1:
                    relax(i, i + w, 0, 1)
                    if cx < w - 1:
                        relax(i, i + w + 1, 1, 1)
                    if cx > 0:
                        relax(i, i + w - 1, -1, 1)
                if cx < w - 1:
                    relax(i, i + 1, 1, 0)

        return [d * self.resolution for d in dist]

    def contains(self, x: float, y: float) -> bool:
        cx, cy = int(x / self.resolution), int(y / self.resolution)
        return 0 <= cx < self.width and 0 <= cy < self.height

    def at(self, x: float, y: float) -> float:
        """Distance to the nearest obstacle, in metres.

        Out of bounds returns a large distance, NOT zero. Returning zero says
        "an obstacle is exactly here", which is the highest possible score --
        it turned the region outside the map into an attractor and every
        particle escaped there. Off-map must be penalised, not rewarded.
        """
        cx, cy = int(x / self.resolution), int(y / self.resolution)
        if not (0 <= cx < self.width and 0 <= cy < self.height):
            return self.max_distance
        return self.distance[cy * self.width + cx]

    def is_obstacle(self, x: float, y: float) -> bool:
        return self.contains(x, y) and self.at(x, y) < self.resolution * 0.5


@dataclass
class SensorModel:
    """Likelihood field parameters.

    `temperature` divides the summed log-likelihood. Beams are treated as
    independent, but adjacent lidar rays see the same wall, so multiplying 24
    of them produces a distribution far sharper than the real information
    content. Untempered, a single particle takes essentially all the weight on
    the first update (measured effective sample size: 3 out of 2000) and the
    filter dies before it can converge. Dividing by ~6 restores a usable
    spread. This is the standard practical fix for an overconfident sensor
    model in MCL.
    """

    sigma: float = 0.25  # wider than the true noise, deliberately
    z_hit: float = 0.85
    z_rand: float = 0.15
    temperature: float = 6.0

    def beam_log_likelihood(self, dist: float, max_range: float) -> float:
        gaussian = math.exp(-(dist * dist) / (2 * self.sigma * self.sigma))
        p = self.z_hit * gaussian + self.z_rand / max_range
        return math.log(max(p, 1e-12))


@dataclass
class Particle:
    pose: Pose
    weight: float = 1.0


@dataclass
class ParticleFilter:
    field_: LikelihoodField
    lidar: Lidar
    noise: MotionNoise = field(default_factory=MotionNoise)
    sensor: SensorModel = field(default_factory=SensorModel)
    count: int = 400
    roughening: float = 0.02  # metres of jitter added after resampling
    injection: float = 0.0  # fraction of particles redrawn uniformly each update
    particles: list[Particle] = field(default_factory=list)
    resamples: int = 0
    ess: float = 0.0

    # -------------------------------------------------------- initialisation

    def initialise_uniform(self, world: OccupancyMap) -> None:
        """Global localization: no idea where the robot is."""
        free = world.free_cells()
        self.particles = []
        for _ in range(self.count):
            cx, cy = random.choice(free)
            x, y = world.cell_to_world(cx, cy)
            x += random.uniform(-0.5, 0.5) * world.resolution
            y += random.uniform(-0.5, 0.5) * world.resolution
            self.particles.append(Particle(Pose(x, y, random.uniform(-math.pi, math.pi))))
        self._normalise()

    def initialise_around(self, pose: Pose, spread: float = 0.5) -> None:
        """Pose tracking: roughly known start."""
        self.particles = [
            Particle(
                Pose(
                    pose.x + random.gauss(0, spread),
                    pose.y + random.gauss(0, spread),
                    wrap_angle(pose.theta + random.gauss(0, 0.2)),
                )
            )
            for _ in range(self.count)
        ]
        self._normalise()

    # ------------------------------------------------------------- the cycle

    def predict(self, v: float, omega: float, dt: float) -> None:
        for p in self.particles:
            p.pose = sample_motion(p.pose, v, omega, dt, self.noise)

    def set_injection(self, fraction: float, world: OccupancyMap) -> None:
        """Enable AMCL-style random particle injection.

        Global localization can converge early onto a plausible but wrong
        hypothesis; once every particle sits there, no amount of further
        evidence can recover, because the correct pose has no particles left to
        reweight. Replacing a small fraction with uniformly drawn poses each
        step keeps a recovery path alive. Without it, global mode converged in
        2 of 8 runs.
        """
        self.injection = fraction
        self._world = world

    def _inject(self) -> None:
        world = getattr(self, "_world", None)
        if not world or self.injection <= 0 or not self.particles:
            return
        free = world.free_cells()
        if not free:
            return
        n = max(1, int(len(self.particles) * self.injection))
        weight = 1.0 / len(self.particles)
        # Replace the lightest particles, never the ones carrying the belief.
        self.particles.sort(key=lambda p: p.weight)
        for i in range(n):
            cx, cy = random.choice(free)
            x, y = world.cell_to_world(cx, cy)
            self.particles[i] = Particle(
                Pose(x, y, random.uniform(-math.pi, math.pi)), weight
            )

    def update(self, scan: list[float]) -> None:
        """Reweight every particle by how well it explains the scan."""
        if not self.particles:
            return
        angles = self.lidar.angles()
        max_range = self.lidar.spec.max_range

        log_weights = []
        for p in self.particles:
            px, py, ptheta = p.pose
            # A particle off the map or inside a wall is impossible, not
            # merely unlikely. Without this it can still score well by
            # explaining beams from an invalid position.
            if not self.field_.contains(px, py) or self.field_.is_obstacle(px, py):
                log_weights.append(-1e6)
                continue
            total = 0.0
            for angle, r in zip(angles, scan, strict=False):
                if r >= max_range - 1e-6:
                    continue  # no return: carries no information about obstacles
                beam = ptheta + angle
                ex = px + r * math.cos(beam)
                ey = py + r * math.sin(beam)
                total += self.sensor.beam_log_likelihood(self.field_.at(ex, ey), max_range)
            log_weights.append(total / self.sensor.temperature)

        # Subtract the max before exponentiating, or everything underflows to 0.
        top = max(log_weights)
        for p, lw in zip(self.particles, log_weights, strict=False):
            p.weight = math.exp(lw - top)
        self._normalise()

        self.ess = self.effective_sample_size()
        if self.ess < self.count / 2:
            self.resample()
        self._inject()

    def resample(self) -> None:
        """Low-variance (systematic) resampling: one random number, evenly spaced."""
        n = len(self.particles)
        if n == 0:
            return
        step = 1.0 / n
        r = random.uniform(0, step)
        cumulative = self.particles[0].weight
        i = 0
        new: list[Particle] = []
        for m in range(n):
            u = r + m * step
            while u > cumulative and i < n - 1:
                i += 1
                cumulative += self.particles[i].weight
            src = self.particles[i].pose
            # Roughening: duplicated particles are identical, so without a
            # little jitter the filter loses diversity it can never recover.
            if self.roughening > 0:
                new.append(
                    Particle(
                        Pose(
                            src.x + random.gauss(0, self.roughening),
                            src.y + random.gauss(0, self.roughening),
                            wrap_angle(src.theta + random.gauss(0, self.roughening * 0.5)),
                        ),
                        1.0 / n,
                    )
                )
            else:
                new.append(Particle(Pose(src.x, src.y, src.theta), 1.0 / n))
        self.particles = new
        self.resamples += 1

    # ------------------------------------------------------------- estimates

    def _normalise(self) -> None:
        total = sum(p.weight for p in self.particles)
        if total <= 0:
            for p in self.particles:
                p.weight = 1.0 / len(self.particles)
            return
        for p in self.particles:
            p.weight /= total

    def effective_sample_size(self) -> float:
        sq = sum(p.weight * p.weight for p in self.particles)
        return 1.0 / sq if sq > 0 else 0.0

    def estimate(self) -> Pose:
        """Best pose estimate: the weighted mean of the heaviest cluster.

        A plain weighted mean over every particle is wrong whenever the
        posterior is multi-modal, which is the normal state of global
        localization before it resolves: with clouds at opposite ends of a
        corridor the mean lands in the middle, where the robot certainly is
        not. Clustering first and averaging within the dominant mode fixes
        that, and reduces to the plain mean once the cloud is unimodal.
        """
        return self._mean_of(self._dominant_cluster())

    def mean_estimate(self) -> Pose:
        """Weighted mean over all particles, ignoring modes."""
        return self._mean_of(self.particles)

    @staticmethod
    def _mean_of(particles: list[Particle]) -> Pose:
        total = sum(p.weight for p in particles) or 1.0
        x = sum(p.weight * p.pose.x for p in particles) / total
        y = sum(p.weight * p.pose.y for p in particles) / total
        cos_t = sum(p.weight * math.cos(p.pose.theta) for p in particles) / total
        sin_t = sum(p.weight * math.sin(p.pose.theta) for p in particles) / total
        return Pose(x, y, math.atan2(sin_t, cos_t))

    def _dominant_cluster(self, cell: float = 1.0) -> list[Particle]:
        """Bin particles into coarse spatial cells; return the heaviest bin.

        A full clustering pass is unnecessary here -- binning at roughly the
        scale of the ambiguity separates competing hypotheses well enough and
        costs one pass over the particles.
        """
        if not self.particles:
            return []
        bins: dict[tuple[int, int], list[Particle]] = {}
        for p in self.particles:
            key = (int(p.pose.x / cell), int(p.pose.y / cell))
            bins.setdefault(key, []).append(p)
        if len(bins) == 1:
            return self.particles

        def bin_weight(key: tuple[int, int]) -> float:
            # Include the 8 neighbours so a mode split across a bin edge is
            # not penalised against one that happens to sit centred.
            total = 0.0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for p in bins.get((key[0] + dx, key[1] + dy), ()):
                        total += p.weight
            return total

        best = max(bins, key=bin_weight)
        cluster: list[Particle] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cluster.extend(bins.get((best[0] + dx, best[1] + dy), ()))
        return cluster

    def cluster_fraction(self) -> float:
        """Weight share of the dominant mode: 1.0 means fully resolved."""
        cluster = self._dominant_cluster()
        total = sum(p.weight for p in self.particles) or 1.0
        return sum(p.weight for p in cluster) / total

    def spread(self) -> float:
        """Weighted standard deviation of particle position -- the uncertainty."""
        mean = self.estimate()
        var = sum(
            p.weight * ((p.pose.x - mean.x) ** 2 + (p.pose.y - mean.y) ** 2)
            for p in self.particles
        )
        return math.sqrt(max(var, 0.0))

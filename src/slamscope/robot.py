"""The robot: how it moves, and what it sees.

Motion uses the velocity motion model from Thrun, Burgard & Fox, *Probabilistic
Robotics* (Ch. 5), with six noise parameters. The same `sample` function drives
both the simulated robot and every particle in the filter, which matters: if the
filter's motion model disagrees with reality, MCL diverges and the demo lies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .world import OccupancyMap, Pose, wrap_angle


@dataclass
class MotionNoise:
    """alpha1..alpha6 from the velocity motion model.

    a1, a2 corrupt translation; a3, a4 corrupt rotation; a5, a6 add the final
    orientation jitter that pure v/omega integration cannot produce.
    """

    a1: float = 0.08
    a2: float = 0.02
    a3: float = 0.02
    a4: float = 0.08
    a5: float = 0.01
    a6: float = 0.01


def sample_motion(pose: Pose, v: float, omega: float, dt: float, noise: MotionNoise) -> Pose:
    """Draw one sample from p(x' | x, u). Set every alpha to 0 for exact motion."""
    v_hat = v + _gauss(noise.a1 * v * v + noise.a2 * omega * omega)
    omega_hat = omega + _gauss(noise.a3 * v * v + noise.a4 * omega * omega)
    gamma_hat = _gauss(noise.a5 * v * v + noise.a6 * omega * omega)

    x, y, theta = pose
    if abs(omega_hat) < 1e-6:
        # Straight-line limit: the arc formula divides by omega.
        nx = x + v_hat * dt * math.cos(theta)
        ny = y + v_hat * dt * math.sin(theta)
        ntheta = wrap_angle(theta + gamma_hat * dt)
    else:
        r = v_hat / omega_hat
        nx = x - r * math.sin(theta) + r * math.sin(theta + omega_hat * dt)
        ny = y + r * math.cos(theta) - r * math.cos(theta + omega_hat * dt)
        ntheta = wrap_angle(theta + omega_hat * dt + gamma_hat * dt)
    return Pose(nx, ny, ntheta)


def _gauss(variance: float) -> float:
    return random.gauss(0.0, math.sqrt(variance)) if variance > 0 else 0.0


@dataclass
class LidarSpec:
    beams: int = 24
    max_range: float = 8.0
    sigma: float = 0.08  # range noise, metres
    dropout: float = 0.02  # probability a beam returns max range
    fov: float = 2 * math.pi


@dataclass
class Lidar:
    spec: LidarSpec = field(default_factory=LidarSpec)

    def angles(self) -> list[float]:
        n = self.spec.beams
        if self.spec.fov >= 2 * math.pi - 1e-9:
            return [2 * math.pi * i / n for i in range(n)]
        start = -self.spec.fov / 2
        return [start + self.spec.fov * i / (n - 1) for i in range(n)]

    def scan(self, pose: Pose, world: OccupancyMap, noisy: bool = True) -> list[float]:
        out = []
        for a in self.angles():
            r = world.raycast(pose.x, pose.y, wrap_angle(pose.theta + a), self.spec.max_range)
            if noisy:
                if random.random() < self.spec.dropout:
                    r = self.spec.max_range
                else:
                    r = max(0.0, min(self.spec.max_range, r + random.gauss(0, self.spec.sigma)))
            out.append(r)
        return out


@dataclass
class Robot:
    """Ground truth pose plus the dead-reckoned estimate built from odometry."""

    pose: Pose
    noise: MotionNoise = field(default_factory=MotionNoise)
    lidar: Lidar = field(default_factory=Lidar)
    odom: Pose | None = None
    trail: list[tuple[float, float]] = field(default_factory=list)
    odom_trail: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.odom is None:
            self.odom = self.pose

    def drive(self, v: float, omega: float, dt: float, world: OccupancyMap) -> bool:
        """Move, refusing to enter a wall. Returns False if the move was blocked."""
        candidate = sample_motion(self.pose, v, omega, dt, self.noise)
        if world.is_occupied_world(candidate.x, candidate.y):
            return False

        self.pose = candidate
        # Dead reckoning integrates the *commanded* velocities: no noise is
        # subtracted out, which is exactly why it drifts.
        self.odom = sample_motion(self.odom, v, omega, dt, MotionNoise(0, 0, 0, 0, 0, 0))
        self.trail.append((self.pose.x, self.pose.y))
        self.odom_trail.append((self.odom.x, self.odom.y))
        if len(self.trail) > 4000:
            self.trail.pop(0)
            self.odom_trail.pop(0)
        return True

    def scan(self, world: OccupancyMap) -> list[float]:
        return self.lidar.scan(self.pose, world)

    @property
    def odom_error(self) -> float:
        return self.pose.distance_to(self.odom) if self.odom else 0.0

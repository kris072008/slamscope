"""The simulation loop, independent of any UI.

Keeping this separate from the Textual app is what makes the headless
benchmark possible: the same code path produces the numbers in the README and
drives the animation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .environments import ENVIRONMENTS, find_open_pose
from .localization import LikelihoodField, ParticleFilter, SensorModel
from .mapping import LogOddsMap
from .robot import Lidar, LidarSpec, MotionNoise, Robot
from .world import OccupancyMap, Pose


@dataclass
class Config:
    environment: str = "office"
    particles: int = 600
    beams: int = 32
    max_range: float = 8.0
    global_localization: bool = False
    map_from: str = "mcl"  # mcl | odom | truth
    dt: float = 0.1
    injection: float = 0.10  # random particle fraction, global mode only
    seed: int | None = None


@dataclass
class Simulation:
    config: Config = field(default_factory=Config)

    def __post_init__(self) -> None:
        if self.config.seed is not None:
            random.seed(self.config.seed)
        self.world: OccupancyMap = ENVIRONMENTS[self.config.environment]()
        self.field = LikelihoodField(self.world)
        self.lidar = Lidar(
            LidarSpec(beams=self.config.beams, max_range=self.config.max_range, sigma=0.08)
        )
        self.robot = Robot(pose=find_open_pose(self.world), lidar=self.lidar)
        self.pf = ParticleFilter(
            field_=self.field,
            lidar=self.lidar,
            count=self.config.particles,
            noise=MotionNoise(),
            sensor=SensorModel(),
        )
        if self.config.global_localization:
            self.pf.initialise_uniform(self.world)
            # Recovery path for global mode; see ParticleFilter.set_injection.
            self.pf.set_injection(self.config.injection, self.world)
        else:
            self.pf.initialise_around(self.robot.pose, 0.5)

        self.built = LogOddsMap(self.world.width, self.world.height, self.world.resolution)
        self.scan: list[float] = []
        self.steps = 0
        self.mcl_error = 0.0
        self.est_trail: list[tuple[float, float]] = []

    # ---------------------------------------------------------------- driving

    def _choose_command(self) -> tuple[float, float]:
        """Reactive wander: drive forward, turn toward whichever side is more open."""
        pose = self.robot.pose
        ahead = self.world.raycast(pose.x, pose.y, pose.theta, 4.0)
        left = self.world.raycast(pose.x, pose.y, pose.theta + 0.7, 4.0)
        right = self.world.raycast(pose.x, pose.y, pose.theta - 0.7, 4.0)
        if ahead < 1.1:
            return 0.1, 2.0 if left > right else -2.0
        if ahead < 2.2:
            return 0.45, 0.9 if left > right else -0.9
        return 0.95, random.gauss(0, 0.12)

    def step(self) -> None:
        v, omega = self._choose_command()
        if not self.robot.drive(v, omega, self.config.dt, self.world):
            # Blocked: spin in place rather than grinding into the wall.
            self.robot.pose = Pose(
                self.robot.pose.x, self.robot.pose.y, self.robot.pose.theta + 0.8
            )
            v, omega = 0.0, 0.8

        self.pf.predict(v, omega, self.config.dt)
        self.scan = self.robot.scan(self.world)
        self.pf.update(self.scan)

        estimate = self.pf.estimate()
        self.mcl_error = estimate.distance_to(self.robot.pose)
        self.est_trail.append((estimate.x, estimate.y))
        if len(self.est_trail) > 4000:
            self.est_trail.pop(0)

        source = {
            "mcl": estimate,
            "odom": self.robot.odom or estimate,
            "truth": self.robot.pose,
        }[self.config.map_from]
        self.built.integrate(source, self.scan, self.lidar.angles(), self.config.max_range)
        self.steps += 1

    # ---------------------------------------------------------------- metrics

    @property
    def estimate(self) -> Pose:
        return self.pf.estimate()

    def heading_error(self) -> float:
        est = self.pf.estimate()
        return abs(math.atan2(
            math.sin(est.theta - self.robot.pose.theta),
            math.cos(est.theta - self.robot.pose.theta),
        ))

    def map_accuracy(self) -> tuple[float, int]:
        return self.built.agreement(self.world)

"""Stream a slamscope run into the Rerun viewer.

Logs, per step: the true world, the map the robot is building, the particle
cloud, true / dead-reckoned / MCL paths, the lidar beams, and the error curves.

    pip install -e . rerun-sdk
    python examples/rerun_viewer.py                 # opens the Rerun viewer
    python examples/rerun_viewer.py cave --global   # another world, global localization
    python examples/rerun_viewer.py --save run.rrd  # write a recording instead
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from slamscope.environments import ENVIRONMENTS
from slamscope.sim import Config, Simulation

TRUE = [80, 200, 120]
ODOM = [230, 80, 80]
MCL = [90, 160, 255]
PARTICLE = [170, 110, 255]
BEAM = [255, 200, 60]


def to_px(sim: Simulation, x: float, y: float) -> tuple[float, float]:
    """Metres -> image pixels, flipping y so north is up in the viewer."""
    res = sim.world.resolution
    return x / res, sim.world.height - y / res


def world_image(sim: Simulation) -> np.ndarray:
    w = sim.world
    grid = np.frombuffer(bytes(w.cells), dtype=np.uint8).reshape(w.height, w.width)
    return np.flipud(np.where(grid > 0, 40, 235).astype(np.uint8))


def built_image(sim: Simulation) -> np.ndarray:
    m = sim.built
    lo = np.array(m.log_odds, dtype=np.float32).reshape(m.height, m.width)
    p = 1.0 - 1.0 / (1.0 + np.exp(lo))  # occupancy probability
    return np.flipud(((1.0 - p) * 255).astype(np.uint8))


OVERLAYS = ["world/particles", "world/paths/**", "world/robot/**", "world/lidar"]

# Without a blueprint Rerun gives each image its own view, so the particles and
# paths never get drawn on top of a map.
BLUEPRINT = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Vertical(
            rrb.Spatial2DView(name="ground truth", origin="world",
                              contents=["world/true_map", *OVERLAYS]),
            rrb.Spatial2DView(name="map the robot built", origin="world",
                              contents=["world/built_map", *OVERLAYS]),
        ),
        rrb.Vertical(
            rrb.TimeSeriesView(name="position error (m)",
                               contents=["error/mcl_m", "error/odom_m"]),
            rrb.TimeSeriesView(name="heading error (rad)", contents=["error/heading_rad"]),
            rrb.TimeSeriesView(name="effective sample size",
                               contents=["filter/effective_sample_size"]),
        ),
        column_shares=[3, 2],
    ),
    collapse_panels=True,
)


def log_step(sim: Simulation) -> None:
    rr.set_time("step", sequence=sim.steps)

    rr.log("world/built_map", rr.Image(built_image(sim)))

    rr.log(
        "world/particles",
        rr.Points2D([to_px(sim, p.pose.x, p.pose.y) for p in sim.pf.particles],
                    colors=PARTICLE, radii=0.6),
    )

    truth, est, odom = sim.robot.pose, sim.estimate, sim.robot.odom
    rr.log("world/paths/true", rr.LineStrips2D([[to_px(sim, *xy) for xy in sim.robot.trail]],
                                               colors=TRUE))
    rr.log("world/paths/odom", rr.LineStrips2D([[to_px(sim, *xy) for xy in sim.robot.odom_trail]],
                                               colors=ODOM))
    rr.log("world/paths/mcl", rr.LineStrips2D([[to_px(sim, *xy) for xy in sim.est_trail]],
                                              colors=MCL))
    rr.log("world/robot/true", rr.Points2D([to_px(sim, truth.x, truth.y)], colors=TRUE, radii=2.5))
    rr.log("world/robot/mcl", rr.Points2D([to_px(sim, est.x, est.y)], colors=MCL, radii=2.5))

    origin = to_px(sim, truth.x, truth.y)
    ends = [
        to_px(sim, truth.x + r * math.cos(truth.theta + a), truth.y + r * math.sin(truth.theta + a))
        for a, r in zip(sim.lidar.angles(), sim.scan, strict=False)
    ]
    rr.log("world/lidar", rr.LineStrips2D([[origin, e] for e in ends], colors=BEAM, radii=0.15))

    rr.log("error/mcl_m", rr.Scalars(sim.mcl_error))
    rr.log("error/odom_m", rr.Scalars(sim.robot.odom_error))
    rr.log("error/heading_rad", rr.Scalars(sim.heading_error()))
    rr.log("filter/effective_sample_size", rr.Scalars(sim.pf.ess))
    if odom is not None:
        rr.log("world/robot/odom",
               rr.Points2D([to_px(sim, odom.x, odom.y)], colors=ODOM, radii=2.5))


def main() -> None:
    p = argparse.ArgumentParser(description="Visualise a slamscope run in Rerun.")
    p.add_argument("environment", nargs="?", default="office", choices=sorted(ENVIRONMENTS))
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--particles", type=int, default=600)
    p.add_argument("--global", dest="global_localization", action="store_true",
                   help="start with no idea where the robot is")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--save", metavar="FILE",
                   help="write an .rrd recording instead of opening the viewer")
    args = p.parse_args()

    rr.init("slamscope", spawn=args.save is None)
    if args.save:
        rr.save(args.save)
    rr.send_blueprint(BLUEPRINT)

    sim = Simulation(Config(
        environment=args.environment,
        particles=args.particles,
        global_localization=args.global_localization,
        seed=args.seed,
    ))
    rr.log("world/true_map", rr.Image(world_image(sim)), static=True)

    for _ in range(args.steps):
        sim.step()
        log_step(sim)

    acc, _ = sim.map_accuracy()
    print(f"{sim.steps} steps  MCL error {sim.mcl_error:.2f} m  "
          f"odom error {sim.robot.odom_error:.2f} m  map agreement {acc:.1%}")


if __name__ == "__main__":
    main()

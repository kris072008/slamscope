"""Record a slamscope run to MCAP for the Foxglove app.

Topics (all in the `world` frame, metres):
    /map/true, /map/built   foxglove.Grid       true world, map the robot is building
    /particles              PosesInFrame        the particle cloud, with headings
    /pose/true, /pose/mcl, /pose/odom   PoseInFrame
    /scan                   LaserScan           lidar from the true pose
    /paths                  SceneUpdate         true / odometry / MCL trails
    /metrics                JSON                position + heading error, ESS

    pip install -e . foxglove-sdk
    python examples/foxglove_mcap.py                          # writes slamscope.mcap
    python examples/foxglove_mcap.py office --global --seed 3 -o global.mcap

Open the file in Foxglove (app.foxglove.dev or desktop), add a 3D panel for the
maps, particles and scan, and a Plot panel on /metrics.
"""

from __future__ import annotations

import argparse
import math

import foxglove
from foxglove.messages import (
    Color,
    Grid,
    LaserScan,
    LinePrimitive,
    LinePrimitiveLineType,
    PackedElementField,
    PackedElementFieldNumericType,
    Point3,
    Pose,
    PoseInFrame,
    PosesInFrame,
    Quaternion,
    SceneEntity,
    SceneUpdate,
    Timestamp,
    Vector2,
    Vector3,
)

from slamscope.environments import ENVIRONMENTS
from slamscope.sim import Config, Simulation
from slamscope.world import Pose as SimPose

FRAME = "world"
U8 = PackedElementFieldNumericType.Uint8
RGBA_FIELDS = [
    PackedElementField(name=name, offset=i, type=U8)
    for i, name in enumerate(("red", "green", "blue", "alpha"))
]


def stamp(ns: int) -> Timestamp:
    return Timestamp(sec=ns // 1_000_000_000, nsec=ns % 1_000_000_000)


def to_pose(p: SimPose) -> Pose:
    return Pose(
        position=Vector3(x=p.x, y=p.y, z=0.0),
        orientation=Quaternion(x=0.0, y=0.0, z=math.sin(p.theta / 2), w=math.cos(p.theta / 2)),
    )


def grid(ts: Timestamp, width: int, height: int, res: float, greys: list[int], z: float) -> Grid:
    """Greyscale grid; row 0 is y = 0, matching slamscope's cell indexing."""
    data = bytearray()
    for g in greys:
        data += bytes((g, g, g, 255))
    return Grid(
        timestamp=ts,
        frame_id=FRAME,
        pose=Pose(position=Vector3(x=0.0, y=0.0, z=z), orientation=Quaternion(w=1.0)),
        column_count=width,
        cell_size=Vector2(x=res, y=res),
        row_stride=width * 4,
        cell_stride=4,
        fields=RGBA_FIELDS,
        data=bytes(data),
    )


def trail(entity_id: str, ts: Timestamp, points: list[tuple[float, float]],
          rgb: tuple[float, float, float]) -> SceneEntity:
    return SceneEntity(
        timestamp=ts,
        frame_id=FRAME,
        id=entity_id,
        lines=[LinePrimitive(
            type=LinePrimitiveLineType.LineStrip,
            thickness=0.06,
            points=[Point3(x=x, y=y, z=0.05) for x, y in points],
            color=Color(r=rgb[0], g=rgb[1], b=rgb[2], a=1.0),
        )],
    )


def log_step(sim: Simulation, ns: int) -> None:
    ts = stamp(ns)
    world = sim.world

    built = [round((1.0 - p) * 255) for p in (
        1.0 - 1.0 / (1.0 + math.exp(lo)) for lo in sim.built.log_odds
    )]
    foxglove.log("/map/built",
                 grid(ts, world.width, world.height, world.resolution, built, z=-0.02),
                 log_time=ns)

    foxglove.log("/particles", PosesInFrame(
        timestamp=ts, frame_id=FRAME, poses=[to_pose(p.pose) for p in sim.pf.particles],
    ), log_time=ns)

    for name, pose in (("true", sim.robot.pose), ("mcl", sim.estimate), ("odom", sim.robot.odom)):
        if pose is not None:
            foxglove.log(f"/pose/{name}", PoseInFrame(timestamp=ts, frame_id=FRAME,
                                                      pose=to_pose(pose)), log_time=ns)

    angles = sim.lidar.angles()
    foxglove.log("/scan", LaserScan(
        timestamp=ts,
        frame_id=FRAME,
        pose=to_pose(sim.robot.pose),
        start_angle=angles[0],
        end_angle=angles[-1],
        ranges=list(sim.scan),
        intensities=[],
    ), log_time=ns)

    foxglove.log("/paths", SceneUpdate(entities=[
        trail("true", ts, sim.robot.trail, (0.3, 0.8, 0.45)),
        trail("odom", ts, sim.robot.odom_trail, (0.9, 0.3, 0.3)),
        trail("mcl", ts, sim.est_trail, (0.35, 0.6, 1.0)),
    ]), log_time=ns)

    foxglove.log("/metrics", {
        "mcl_error_m": sim.mcl_error,
        "odom_error_m": sim.robot.odom_error,
        "heading_error_rad": sim.heading_error(),
        "effective_sample_size": sim.pf.ess,
    }, log_time=ns)


def main() -> None:
    p = argparse.ArgumentParser(description="Record a slamscope run to MCAP for Foxglove.")
    p.add_argument("environment", nargs="?", default="office", choices=sorted(ENVIRONMENTS))
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--particles", type=int, default=600)
    p.add_argument("--global", dest="global_localization", action="store_true",
                   help="start with no idea where the robot is")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("-o", "--output", default="slamscope.mcap")
    args = p.parse_args()

    sim = Simulation(Config(
        environment=args.environment,
        particles=args.particles,
        global_localization=args.global_localization,
        seed=args.seed,
    ))
    dt_ns = round(sim.config.dt * 1e9)

    with foxglove.open_mcap(args.output, allow_overwrite=True):
        w = sim.world
        truth = [235 if w.get(cx, cy) == 0 else 40
                 for cy in range(w.height) for cx in range(w.width)]
        foxglove.log("/map/true",
                     grid(stamp(0), w.width, w.height, w.resolution, truth, z=-0.04),
                     log_time=0)
        for _ in range(args.steps):
            sim.step()
            log_step(sim, sim.steps * dt_ns)

    acc, _ = sim.map_accuracy()
    print(f"wrote {args.output}: {sim.steps} steps  MCL error {sim.mcl_error:.2f} m  "
          f"odom error {sim.robot.odom_error:.2f} m  map agreement {acc:.1%}")


if __name__ == "__main__":
    main()

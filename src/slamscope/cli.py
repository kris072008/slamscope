"""Command-line entry point."""

from __future__ import annotations

import argparse
import random
import statistics
import time

from . import __version__
from .environments import ENVIRONMENTS
from .sim import Config, Simulation


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="slamscope",
        description="Particle-filter localization and occupancy mapping, in your terminal.",
    )
    p.add_argument(
        "environment", nargs="?", default="office", choices=sorted(ENVIRONMENTS),
        help="which world to drive around (default: office)",
    )
    p.add_argument("-n", "--particles", type=int, default=600, help="particle count")
    p.add_argument("-b", "--beams", type=int, default=32, help="lidar beams per scan")
    p.add_argument("--range", type=float, default=8.0, help="lidar max range in metres")
    p.add_argument(
        "-g", "--global", dest="global_localization", action="store_true",
        help="start with no idea where the robot is (uniform prior)",
    )
    p.add_argument(
        "--map-from", choices=["mcl", "odom", "truth"], default="mcl",
        help="which pose estimate builds the map (default: mcl)",
    )
    p.add_argument("--seed", type=int, help="seed for a reproducible run")
    p.add_argument("--list", action="store_true", help="list environments and exit")
    p.add_argument(
        "--bench", type=int, metavar="STEPS", nargs="?", const=400,
        help="run headless for STEPS and print metrics (default 400)",
    )
    p.add_argument("--trials", type=int, default=1, help="repeat --bench this many times")
    p.add_argument("-V", "--version", action="version", version=f"slamscope {__version__}")
    return p


def run_bench(args) -> int:
    results = []
    print(f"environment  {args.environment}")
    print(f"particles    {args.particles}   beams {args.beams}   range {args.range} m")
    print(f"mode         {'global localization' if args.global_localization else 'pose tracking'}")
    print(f"steps        {args.bench}   trials {args.trials}\n")
    print(f"{'trial':>5} {'MCL err':>9} {'odom err':>9} {'ratio':>7} "
          f"{'heading':>8} {'map':>7} {'ms/step':>8}")
    print("-" * 60)

    for trial in range(args.trials):
        # Offset by the trial index, or every trial would be the same run.
        seed = (args.seed + trial) if args.seed is not None else random.randrange(1 << 30)
        config = Config(
            environment=args.environment,
            particles=args.particles,
            beams=args.beams,
            max_range=args.range,
            global_localization=args.global_localization,
            map_from=args.map_from,
            seed=seed,
        )
        sim = Simulation(config)
        errors = []
        t0 = time.perf_counter()
        for _ in range(args.bench):
            sim.step()
            errors.append(sim.mcl_error)
        ms = (time.perf_counter() - t0) * 1000 / args.bench

        mcl = statistics.mean(errors[-50:])
        odom = sim.robot.odom_error
        accuracy, _ = sim.map_accuracy()
        ratio = odom / mcl if mcl > 1e-9 else float("inf")
        results.append((mcl, odom, accuracy))
        print(f"{trial + 1:>5} {mcl:>9.3f} {odom:>9.3f} {ratio:>6.1f}x "
              f"{sim.heading_error():>8.3f} {accuracy * 100:>6.1f}% {ms:>8.1f}")

    if args.trials > 1:
        mcls = [r[0] for r in results]
        odoms = [r[1] for r in results]
        accs = [r[2] for r in results]
        print("-" * 60)
        print(f"{'mean':>5} {statistics.mean(mcls):>9.3f} {statistics.mean(odoms):>9.3f} "
              f"{statistics.mean(odoms) / statistics.mean(mcls):>6.1f}x "
              f"{'':>8} {statistics.mean(accs) * 100:>6.1f}%")
        converged = sum(1 for m in mcls if m < 0.5)
        print(f"\nconverged (< 0.5 m) in {converged}/{args.trials} trials")
        beat = sum(1 for m, o, _ in results if m < o)
        print(f"beat dead reckoning in {beat}/{args.trials} trials")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        for name, fn in sorted(ENVIRONMENTS.items()):
            doc = ((fn.__doc__ or "").strip().splitlines() or [""])[0]
            print(f"  {name:10} {doc}")
        return 0

    if args.bench:
        return run_bench(args)

    from .app import SlamscopeApp

    SlamscopeApp(
        Config(
            environment=args.environment,
            particles=args.particles,
            beams=args.beams,
            max_range=args.range,
            global_localization=args.global_localization,
            map_from=args.map_from,
            seed=args.seed,
        )
    ).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

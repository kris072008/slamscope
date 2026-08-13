import math
import random

import pytest

from slamscope.environments import ENVIRONMENTS, find_open_pose
from slamscope.localization import LikelihoodField, ParticleFilter
from slamscope.mapping import LogOddsMap
from slamscope.robot import Lidar, LidarSpec, MotionNoise, Robot, sample_motion
from slamscope.sim import Config, Simulation
from slamscope.world import FREE, OCCUPIED, OccupancyMap, Pose, wrap_angle


@pytest.fixture
def room():
    """10 m x 10 m room, walls one cell thick, interior clear."""
    m = OccupancyMap(20, 20, 0.5)
    for i in range(20):
        m.set(i, 0, OCCUPIED)
        m.set(i, 19, OCCUPIED)
        m.set(0, i, OCCUPIED)
        m.set(19, i, OCCUPIED)
    return m


class TestAngles:
    @pytest.mark.parametrize(
        "raw,expected",
        [(0, 0), (math.pi, math.pi), (3 * math.pi, math.pi), (-3 * math.pi, math.pi),
         (2 * math.pi, 0), (math.pi / 2, math.pi / 2), (-7.3, -7.3 + 2 * math.pi)],
    )
    def test_wrap(self, raw, expected):
        """Compare as directions: +pi and -pi are the same angle."""
        wrapped = wrap_angle(raw)
        assert math.cos(wrapped) == pytest.approx(math.cos(expected), abs=1e-9)
        assert math.sin(wrapped) == pytest.approx(math.sin(expected), abs=1e-9)

    def test_wrap_always_in_range(self):
        random.seed(0)
        for _ in range(500):
            a = random.uniform(-100, 100)
            assert -math.pi - 1e-9 <= wrap_angle(a) <= math.pi + 1e-9


class TestPose:
    def test_relative_to_self_is_zero(self):
        p = Pose(3, 4, 1.2)
        dx, dy, dt = p.relative_to(p)
        assert (dx, dy, dt) == pytest.approx((0, 0, 0), abs=1e-9)

    def test_relative_frame_rotation(self):
        origin = Pose(0, 0, math.pi / 2)
        ahead = Pose(0, 5, math.pi / 2)
        dx, dy, _ = ahead.relative_to(origin)
        assert (dx, dy) == pytest.approx((5, 0), abs=1e-9)


class TestRaycast:
    def test_cardinal_directions(self, room):
        for theta in (0, math.pi / 2, math.pi, -math.pi / 2):
            assert room.raycast(5, 5, theta, 50) == pytest.approx(4.5, abs=1e-9)

    def test_diagonal(self, room):
        assert room.raycast(5, 5, math.pi / 4, 50) == pytest.approx(4.5 * math.sqrt(2), abs=1e-9)

    def test_max_range_clamps(self, room):
        assert room.raycast(5, 5, 0, 2.0) == 2.0

    def test_starting_inside_wall_is_zero(self, room):
        assert room.raycast(0.1, 5, 0, 10) == 0.0

    def test_thin_wall_is_not_skipped(self):
        """A one-cell wall must stop the ray; fixed-step sampling would miss it."""
        m = OccupancyMap(40, 40, 0.25)
        for cy in range(40):
            m.set(20, cy, OCCUPIED)
        assert m.raycast(1, 5, 0, 50) == pytest.approx(4.0, abs=1e-9)

    def test_shallow_angle_through_thin_wall(self):
        m = OccupancyMap(40, 40, 0.25)
        for cy in range(40):
            m.set(20, cy, OCCUPIED)
        angle = math.radians(5)
        assert m.raycast(1, 5, angle, 50) == pytest.approx(4.0 / math.cos(angle), abs=0.02)

    def test_endpoint_always_lands_in_an_obstacle(self, room):
        random.seed(1)
        for _ in range(400):
            x, y = random.uniform(0.6, 9.4), random.uniform(0.6, 9.4)
            theta = random.uniform(-math.pi, math.pi)
            r = room.raycast(x, y, theta, 50)
            ex = x + math.cos(theta) * (r + 0.03)
            ey = y + math.sin(theta) * (r + 0.03)
            assert room.is_occupied_world(ex, ey)


class TestLikelihoodField:
    def test_matches_brute_force_exactly(self):
        m = OccupancyMap(30, 20, 0.25)
        for cy in range(20):
            m.set(0, cy, OCCUPIED)
            m.set(29, cy, OCCUPIED)
        for cx in range(30):
            m.set(cx, 0, OCCUPIED)
            m.set(cx, 19, OCCUPIED)
        m.set(15, 10, OCCUPIED)

        field = LikelihoodField(m)
        occupied = [
            (cx, cy) for cy in range(20) for cx in range(30) if m.get(cx, cy) == OCCUPIED
        ]
        for cy in range(20):
            for cx in range(30):
                truth = min(math.hypot(cx - ox, cy - oy) for ox, oy in occupied) * 0.25
                assert field.distance[cy * 30 + cx] == pytest.approx(truth, abs=1e-9)

    def test_zero_on_obstacles(self, room):
        field = LikelihoodField(room)
        assert field.at(0.1, 5.0) == pytest.approx(0.0)

    def test_off_map_is_penalised_not_rewarded(self, room):
        """Regression: off-map used to return 0.0, the best possible score.

        That made everywhere outside the world look like a perfect obstacle
        match, so the entire particle cloud escaped the map and the filter
        locked onto a pose 18 m outside a 15 m world.
        """
        field = LikelihoodField(room)
        assert field.at(-5.0, -5.0) > 1.0
        assert field.at(500.0, 500.0) == field.max_distance
        assert not field.contains(-1.0, 5.0)

    def test_is_obstacle(self, room):
        field = LikelihoodField(room)
        assert field.is_obstacle(0.1, 5.0)
        assert not field.is_obstacle(5.0, 5.0)


class TestMotionModel:
    def test_zero_noise_straight_line(self):
        noise = MotionNoise(0, 0, 0, 0, 0, 0)
        end = sample_motion(Pose(0, 0, 0), 1.0, 0.0, 2.0, noise)
        assert (end.x, end.y, end.theta) == pytest.approx((2.0, 0.0, 0.0), abs=1e-9)

    def test_zero_noise_pure_rotation(self):
        noise = MotionNoise(0, 0, 0, 0, 0, 0)
        end = sample_motion(Pose(0, 0, 0), 0.0, 1.0, math.pi, noise)
        assert end.theta == pytest.approx(wrap_angle(math.pi), abs=1e-6)
        assert (end.x, end.y) == pytest.approx((0, 0), abs=1e-6)

    def test_arc_radius_is_correct(self):
        """A quarter turn at v/omega = r should land r away in both axes."""
        noise = MotionNoise(0, 0, 0, 0, 0, 0)
        v, omega = 1.0, math.pi / 2
        end = sample_motion(Pose(0, 0, 0), v, omega, 1.0, noise)
        r = v / omega
        assert (end.x, end.y) == pytest.approx((r, r), abs=1e-9)

    def test_noise_produces_spread(self):
        random.seed(0)
        noise = MotionNoise()
        ends = [sample_motion(Pose(0, 0, 0), 1.0, 0.1, 1.0, noise) for _ in range(200)]
        xs = [p.x for p in ends]
        assert max(xs) - min(xs) > 0.01

    def test_dead_reckoning_drifts(self, room):
        random.seed(3)
        robot = Robot(pose=Pose(5, 5, 0))
        for _ in range(150):
            robot.drive(0.5, 0.1, 0.1, room)
        assert robot.odom_error > 0.05


class TestLidar:
    def test_beam_count_and_range(self, room):
        lidar = Lidar(LidarSpec(beams=16, max_range=6.0))
        scan = lidar.scan(Pose(5, 5, 0), room)
        assert len(scan) == 16
        assert all(0 <= r <= 6.0 for r in scan)

    def test_noiseless_scan_matches_raycast(self, room):
        lidar = Lidar(LidarSpec(beams=8, max_range=20.0))
        pose = Pose(5, 5, 0.3)
        scan = lidar.scan(pose, room, noisy=False)
        for angle, r in zip(lidar.angles(), scan, strict=False):
            assert r == pytest.approx(
                room.raycast(pose.x, pose.y, pose.theta + angle, 20.0), abs=1e-9
            )

    def test_limited_fov_spans_correctly(self):
        lidar = Lidar(LidarSpec(beams=5, fov=math.pi))
        angles = lidar.angles()
        assert angles[0] == pytest.approx(-math.pi / 2)
        assert angles[-1] == pytest.approx(math.pi / 2)


class TestParticleFilter:
    def _setup(self, room, count=200):
        field = LikelihoodField(room)
        lidar = Lidar(LidarSpec(beams=16, max_range=8.0))
        pf = ParticleFilter(field_=field, lidar=lidar, count=count)
        return field, lidar, pf

    def test_uniform_init_places_in_free_space(self, room):
        random.seed(0)
        _, _, pf = self._setup(room)
        pf.initialise_uniform(room)
        assert len(pf.particles) == 200
        assert all(not room.is_occupied_world(p.pose.x, p.pose.y) for p in pf.particles)

    def test_weights_normalise(self, room):
        random.seed(0)
        _, lidar, pf = self._setup(room)
        pf.initialise_around(Pose(5, 5, 0))
        pf.update(lidar.scan(Pose(5, 5, 0), room))
        assert sum(p.weight for p in pf.particles) == pytest.approx(1.0)

    def test_effective_sample_size_bounds(self, room):
        random.seed(0)
        _, _, pf = self._setup(room)
        pf.initialise_around(Pose(5, 5, 0))
        assert pf.effective_sample_size() == pytest.approx(200)

    def test_resampling_preserves_count(self, room):
        random.seed(0)
        _, _, pf = self._setup(room)
        pf.initialise_around(Pose(5, 5, 0))
        for i, p in enumerate(pf.particles):
            p.weight = 1.0 if i == 0 else 0.0
        pf._normalise()
        pf.resample()
        assert len(pf.particles) == 200

    def test_low_variance_resampler_follows_weights(self, room):
        """A particle with 50% of the weight should get about half the copies."""
        random.seed(0)
        _, _, pf = self._setup(room, count=1000)
        pf.initialise_around(Pose(5, 5, 0), spread=2.0)
        pf.roughening = 0.0
        target = pf.particles[0].pose
        for i, p in enumerate(pf.particles):
            p.weight = 0.5 if i == 0 else 0.5 / 999
        pf._normalise()
        pf.resample()
        copies = sum(1 for p in pf.particles if p.pose == target)
        assert 400 < copies < 600

    def test_estimate_averages_heading_circularly(self, room):
        """Averaging +179 and -179 degrees must give 180, not 0."""
        random.seed(0)
        _, _, pf = self._setup(room, count=2)
        pf.initialise_around(Pose(5, 5, 0))
        pf.particles[0].pose = Pose(5, 5, math.radians(179))
        pf.particles[1].pose = Pose(5, 5, math.radians(-179))
        pf.particles[0].weight = pf.particles[1].weight = 0.5
        assert abs(pf.estimate().theta) == pytest.approx(math.pi, abs=1e-6)

    def test_particle_in_wall_is_rejected(self, room):
        random.seed(0)
        _, lidar, pf = self._setup(room, count=2)
        pf.initialise_around(Pose(5, 5, 0))
        pf.particles[0].pose = Pose(5.0, 5.0, 0.0)
        pf.particles[1].pose = Pose(0.1, 5.0, 0.0)  # inside the wall
        pf.roughening = 0.0
        pf.update(lidar.scan(Pose(5, 5, 0), room, noisy=False))
        assert pf.particles[0].weight > pf.particles[1].weight


class TestMapping:
    def test_free_and_occupied_evidence(self, room):
        built = LogOddsMap(20, 20, 0.5)
        lidar = Lidar(LidarSpec(beams=32, max_range=12.0))
        pose = Pose(5, 5, 0)
        built.integrate(pose, lidar.scan(pose, room, noisy=False), lidar.angles(), 12.0)
        assert built.probability(10, 10) < 0.5  # robot's own cell: free
        assert built.probability(19, 10) > 0.5  # wall

    def test_no_return_beams_do_not_clear_walls(self):
        """Regression: max-range beams used to trace free space through walls.

        A no-return reading is ambiguous, so clearing along it punched holes in
        real obstacles -- 625 of 630 mapping errors in the maze came from this.
        """
        m = OccupancyMap(40, 20, 0.25)
        for cy in range(20):
            m.set(10, cy, OCCUPIED)

        built = LogOddsMap(40, 20, 0.25)
        pose = Pose(1.0, 2.5, 0.0)
        # One beam pointing straight at the wall, but reported as no-return.
        built.integrate(pose, [8.0], [0.0], 8.0)
        assert built.probability(10, 10) == pytest.approx(0.5)

        built.integrate(pose, [8.0], [0.0], 8.0, clear_on_no_return=True)
        assert built.probability(10, 10) < 0.5

    def test_repeated_observation_increases_confidence(self, room):
        built = LogOddsMap(20, 20, 0.5)
        lidar = Lidar(LidarSpec(beams=8, max_range=12.0))
        pose = Pose(5, 5, 0)
        scan = lidar.scan(pose, room, noisy=False)
        built.integrate(pose, scan, lidar.angles(), 12.0)
        first = built.probability(19, 10)
        for _ in range(5):
            built.integrate(pose, scan, lidar.angles(), 12.0)
        assert built.probability(19, 10) > first

    def test_log_odds_are_clamped(self, room):
        built = LogOddsMap(20, 20, 0.5)
        lidar = Lidar(LidarSpec(beams=8, max_range=12.0))
        pose = Pose(5, 5, 0)
        scan = lidar.scan(pose, room, noisy=False)
        for _ in range(200):
            built.integrate(pose, scan, lidar.angles(), 12.0)
        assert all(-6.001 <= lo <= 6.001 for lo in built.log_odds)

    def test_unobserved_cells_stay_unknown(self):
        built = LogOddsMap(10, 10, 0.5)
        assert built.probability(5, 5) == pytest.approx(0.5)
        assert built.explored_fraction() == 0.0

    def test_agreement_ignores_unknown_cells(self, room):
        built = LogOddsMap(20, 20, 0.5)
        accuracy, decided = built.agreement(room)
        assert decided == 0 and accuracy == 0.0


class TestEnvironments:
    @pytest.mark.parametrize("name", sorted(ENVIRONMENTS))
    def test_bordered_and_non_trivial(self, name):
        random.seed(2)
        world = ENVIRONMENTS[name]()
        for cx in range(world.width):
            assert world.get(cx, 0) == OCCUPIED
            assert world.get(cx, world.height - 1) == OCCUPIED
        free = sum(1 for c in world.cells if c == FREE)
        assert free > world.width * world.height * 0.3

    @pytest.mark.parametrize("name", sorted(ENVIRONMENTS))
    def test_open_start_pose(self, name):
        random.seed(2)
        world = ENVIRONMENTS[name]()
        pose = find_open_pose(world)
        assert not world.is_occupied_world(pose.x, pose.y)


class TestSimulation:
    def test_runs_and_records(self):
        sim = Simulation(Config(environment="cave", particles=120, seed=4))
        for _ in range(25):
            sim.step()
        assert sim.steps == 25
        assert len(sim.scan) == sim.config.beams
        assert len(sim.robot.trail) > 0

    def test_robot_never_enters_a_wall(self):
        sim = Simulation(Config(environment="office", particles=100, seed=6))
        for _ in range(120):
            sim.step()
            assert not sim.world.is_occupied_world(sim.robot.pose.x, sim.robot.pose.y)

    def test_mcl_beats_dead_reckoning(self):
        """The headline claim, asserted rather than assumed."""
        sim = Simulation(Config(environment="cave", particles=400, seed=11))
        for _ in range(250):
            sim.step()
        assert sim.mcl_error < sim.robot.odom_error
        assert sim.mcl_error < 1.0

    def test_map_from_truth_is_accurate(self):
        sim = Simulation(Config(environment="cave", particles=150, map_from="truth", seed=8))
        for _ in range(250):
            sim.step()
        accuracy, decided = sim.map_accuracy()
        assert decided > 500
        assert accuracy > 0.9

    def test_global_localization_converges(self):
        """Global mode must recover from a uniform prior, not just track."""
        sim = Simulation(
            Config(environment="cave", particles=800, global_localization=True, seed=6)
        )
        for _ in range(400):
            sim.step()
        assert sim.mcl_error < 1.0

    def test_injection_replaces_lightest_particles(self):
        """Injection must never discard the particles carrying the belief."""
        sim = Simulation(Config(environment="cave", particles=200, seed=3))
        sim.pf.set_injection(0.1, sim.world)
        for i, p in enumerate(sim.pf.particles):
            p.weight = 1.0 if i == 0 else 1e-9
        sim.pf._normalise()
        best = sim.pf.particles[0].pose
        sim.pf._inject()
        assert any(p.pose == best for p in sim.pf.particles)

    def test_estimate_uses_dominant_cluster(self):
        """With two separated clouds the estimate must pick one, not the midpoint."""
        sim = Simulation(Config(environment="cave", particles=100, seed=3))
        for i, p in enumerate(sim.pf.particles):
            p.pose = Pose(2.0, 2.0, 0.0) if i < 70 else Pose(14.0, 10.0, 0.0)
            p.weight = 1.0
        sim.pf._normalise()
        est = sim.pf.estimate()
        assert est.distance_to(Pose(2.0, 2.0, 0.0)) < 1.0
        mean = sim.pf.mean_estimate()
        assert mean.distance_to(Pose(2.0, 2.0, 0.0)) > 1.0

    def test_particles_stay_on_the_map(self):
        """Regression guard for the off-map attractor."""
        sim = Simulation(Config(environment="office", particles=300, seed=9))
        for _ in range(120):
            sim.step()
        width, height = sim.world.extent
        inside = sum(
            1
            for p in sim.pf.particles
            if -1 <= p.pose.x <= width + 1 and -1 <= p.pose.y <= height + 1
        )
        assert inside == len(sim.pf.particles)

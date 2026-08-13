"""The slamscope TUI."""

from __future__ import annotations

import time

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, Header, Static

from .environments import ENVIRONMENTS
from .render import (
    ESTIMATE,
    ODOM_TRAIL,
    TRUE_TRAIL,
    Painter,
    paint_beams,
    paint_built_map,
    paint_particles,
    paint_robot,
    paint_trail,
    paint_world,
)
from .sim import Config, Simulation

MAP_SOURCES = ["mcl", "odom", "truth"]


class TruthPanel(Static):
    """Ground truth: the world, the robot, its beams, and the particle cloud."""

    def __init__(self, sim: Simulation, **kwargs) -> None:
        super().__init__("", **kwargs)
        self.sim = sim
        self.show_beams = True
        self.show_particles = True

    def refresh_view(self) -> None:
        sim = self.sim
        p = Painter(sim.world.width, sim.world.height, sim.world.resolution)
        paint_world(p, sim.world)
        if self.show_beams and sim.scan:
            paint_beams(p, sim.robot.pose, sim.scan, sim.lidar.angles())
        paint_trail(p, sim.robot.odom_trail, ODOM_TRAIL)
        paint_trail(p, sim.robot.trail, TRUE_TRAIL)
        if self.show_particles:
            paint_particles(p, sim.pf)
        paint_robot(p, sim.estimate, ESTIMATE, heading=0.9)
        paint_robot(p, sim.robot.pose)

        header = Text()
        header.append("GROUND TRUTH  ", "bold white")
        header.append("true ", TRUE_TRAIL)
        header.append("odom ", ODOM_TRAIL)
        header.append("mcl ", ESTIMATE)
        header.append(f"particles {len(sim.pf.particles)}", "grey62")

        out = Text()
        out.append_text(header)
        out.append("\n")
        out.append_text(p.to_text())
        self.update(out)


class MapPanel(Static):
    """The map the robot has built for itself."""

    def __init__(self, sim: Simulation, **kwargs) -> None:
        super().__init__("", **kwargs)
        self.sim = sim

    def refresh_view(self) -> None:
        sim = self.sim
        p = Painter(sim.built.width, sim.built.height, sim.built.resolution)
        paint_built_map(p, sim.built)
        paint_trail(p, sim.est_trail, ESTIMATE)
        paint_robot(p, sim.estimate, ESTIMATE, heading=0.9)

        accuracy, _decided = sim.map_accuracy()
        header = Text()
        header.append("BUILT MAP  ", "bold white")
        header.append(f"from {sim.config.map_from}  ", "yellow")
        header.append(f"accuracy {accuracy * 100:5.1f}%  ", "grey70")
        header.append(f"explored {sim.built.explored_fraction() * 100:4.1f}%", "grey62")

        out = Text()
        out.append_text(header)
        out.append("\n")
        out.append_text(p.to_text())
        self.update(out)


class SlamscopeApp(App):
    CSS = """
    Screen { layout: vertical; }
    #panels { height: 1fr; }
    TruthPanel, MapPanel {
        width: 1fr;
        border: round $primary;
        padding: 0 1;
    }
    #status {
        height: 2;
        background: $panel;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("space", "pause", "Pause"),
        Binding("r", "restart", "Restart"),
        Binding("n", "next_env", "Next world"),
        Binding("g", "toggle_global", "Global/track"),
        Binding("m", "cycle_map_source", "Map source"),
        Binding("b", "toggle_beams", "Beams"),
        Binding("p", "toggle_particles", "Particles"),
        Binding("f", "faster", "Faster"),
        Binding("x", "slower", "Slower"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, config: Config | None = None) -> None:
        super().__init__()
        self.config_ = config or Config()
        self.sim = Simulation(self.config_)
        self.paused = False
        self.steps_per_tick = 1
        self.step_ms = 0.0
        self.converged_at: int | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="panels"):
            yield TruthPanel(self.sim, id="truth")
            yield MapPanel(self.sim, id="map")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "slamscope"
        self._sync_subtitle()
        self.set_interval(1 / 15, self.tick)

    def _sync_subtitle(self) -> None:
        mode = "global localization" if self.config_.global_localization else "pose tracking"
        self.sub_title = f"{self.config_.environment} - {mode}"

    # ----------------------------------------------------------------- loop

    def tick(self) -> None:
        if not self.paused:
            t0 = time.perf_counter()
            for _ in range(self.steps_per_tick):
                self.sim.step()
            self.step_ms = (time.perf_counter() - t0) * 1000 / self.steps_per_tick
            if self.converged_at is None and self.sim.mcl_error < 0.3:
                self.converged_at = self.sim.steps

        self.query_one("#truth", TruthPanel).refresh_view()
        self.query_one("#map", MapPanel).refresh_view()
        self._update_status()

    def _update_status(self) -> None:
        sim = self.sim
        odom = sim.robot.odom_error
        mcl = sim.mcl_error
        ratio = odom / mcl if mcl > 1e-6 else float("inf")

        line1 = Text()
        line1.append(" PAUSED " if self.paused else " RUNNING ",
                     "bold yellow" if self.paused else "bold green")
        line1.append(f"| step {sim.steps:<5} ")
        line1.append(f"| MCL err {mcl:5.2f} m ", "bold cyan")
        line1.append(f"| odom err {odom:5.2f} m ", "bold red")
        line1.append(f"| {ratio:5.1f}x better " if ratio != float("inf") else "| -- ", "bold white")
        line1.append(f"| heading err {sim.heading_error():4.2f} rad ")

        accuracy, _ = sim.map_accuracy()
        line2 = Text()
        line2.append(f" ESS {sim.pf.ess:6.0f}/{len(sim.pf.particles)} ")
        line2.append(f"| resamples {sim.pf.resamples:<4} ")
        line2.append(f"| spread {sim.pf.spread():4.2f} m ")
        line2.append(f"| map {accuracy * 100:5.1f}% ")
        line2.append(f"| {self.step_ms:4.1f} ms/step ")
        line2.append(f"| speed {self.steps_per_tick}x ")
        if self.converged_at is not None:
            line2.append(f"| converged at step {self.converged_at}", "bold green")

        out = Text()
        out.append_text(line1)
        out.append("\n")
        out.append_text(line2)
        self.query_one("#status", Static).update(out)

    # -------------------------------------------------------------- actions

    def _rebuild(self) -> None:
        self.sim = Simulation(self.config_)
        self.converged_at = None
        for widget in (self.query_one("#truth", TruthPanel), self.query_one("#map", MapPanel)):
            widget.sim = self.sim
        self._sync_subtitle()

    def action_pause(self) -> None:
        self.paused = not self.paused

    def action_restart(self) -> None:
        self._rebuild()

    def action_next_env(self) -> None:
        names = list(ENVIRONMENTS)
        idx = names.index(self.config_.environment)
        self.config_.environment = names[(idx + 1) % len(names)]
        self.config_.seed = None
        self._rebuild()

    def action_toggle_global(self) -> None:
        self.config_.global_localization = not self.config_.global_localization
        self._rebuild()

    def action_cycle_map_source(self) -> None:
        idx = MAP_SOURCES.index(self.config_.map_from)
        self.config_.map_from = MAP_SOURCES[(idx + 1) % len(MAP_SOURCES)]
        self.sim.config.map_from = self.config_.map_from

    def action_toggle_beams(self) -> None:
        panel = self.query_one("#truth", TruthPanel)
        panel.show_beams = not panel.show_beams

    def action_toggle_particles(self) -> None:
        panel = self.query_one("#truth", TruthPanel)
        panel.show_particles = not panel.show_particles

    def action_faster(self) -> None:
        self.steps_per_tick = min(8, self.steps_per_tick + 1)

    def action_slower(self) -> None:
        self.steps_per_tick = max(1, self.steps_per_tick - 1)

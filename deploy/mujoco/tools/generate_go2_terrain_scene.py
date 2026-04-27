#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parents[3]
GO2_DIR = REPO_ROOT / "deploy" / "mujoco" / "unitree_robots" / "go2"
DEFAULT_OUTPUT = GO2_DIR / "scene_terrain.xml"
TERRAIN_CHOICES = ("flat", "rough", "stairs", "stairs-high", "pit", "gap", "mixed")

# These values mirror source/legged_lab/.../gait_reward_based/terrain.py.
TERRAIN_RANGES = {
    "perlin_rough_noise_scale": (0.0, 0.10),
    "perlin_noise_frequency": 20,
    "square_gaps_easy_distance": (0.05, 0.40),
    "square_gaps_easy_depth": (0.20, 0.45),
    "square_gaps_easy_platform_width": 2.8,
    "square_gaps_hard_distance": (0.25, 0.95),
    "square_gaps_hard_depth": (0.45, 0.90),
    "square_gaps_hard_platform_width": 2.0,
    "stairs_step_height": (0.04, 0.20),
    "stairs_step_length": (0.20, 0.40),
    "stairs_num_steps": (2, 6),
    "stairs_step_width": 3.8,
    "stairs_platform_length": 1.5,
    "pyramid_step_height": (0.08, 0.45),
    "pyramid_step_width": 1.5,
    "pyramid_platform_width": 4.0,
}


@dataclass(frozen=True)
class SectionSummary:
    name: str
    x0: float
    x1: float
    detail: str


class TerrainBuilder:
    def __init__(self) -> None:
        self.geoms: list[str] = []
        self.summaries: list[SectionSummary] = []
        self._counter = 0

    def comment(self, text: str) -> None:
        self.geoms.append(f"\n    <!-- {escape(text)} -->\n")

    def box(
        self,
        name: str,
        pos: tuple[float, float, float],
        size: tuple[float, float, float],
        material: str,
        friction: str = "0.9 0.6 0.02",
        group: int = 0,
    ) -> None:
        px, py, pz = pos
        sx, sy, sz = size
        safe_name = escape(f"{name}_{self._counter:04d}")
        self._counter += 1
        self.geoms.append(
            f'    <geom name="{safe_name}" type="box" pos="{px:.3f} {py:.3f} {pz:.3f}" '
            f'size="{sx:.3f} {sy:.3f} {sz:.3f}" material="{material}" '
            f'friction="{friction}" group="{group}" />\n'
        )

    def marker(self, name: str, x: float, y: float = -2.35) -> None:
        self.box(name, (x, y, 0.035), (0.10, 0.10, 0.035), "marker_red", group=1)

    def section(self, name: str, x0: float, x1: float, detail: str) -> None:
        self.summaries.append(SectionSummary(name, x0, x1, detail))

    def plate(
        self,
        name: str,
        x_center: float,
        length: float,
        half_width: float,
        top_height: float,
        material: str,
    ) -> None:
        if top_height <= 1e-6:
            return
        self.box(name, (x_center, 0.0, top_height / 2.0), (length / 2.0, half_width, top_height / 2.0), material)


def header() -> str:
    return '''<mujoco model="go2 terrain scene">
  <include file="go2.xml" />

  <statistic center="3.00 0 0.35" extent="7.00" />

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
    <rgba haze="0.15 0.25 0.35 1" />
    <global azimuth="-130" elevation="-20" />
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072" />
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />
    <material name="rough_mat" rgba="0.36 0.34 0.28 1" />
    <material name="rough_dark_mat" rgba="0.25 0.25 0.22 1" />
    <material name="gap_mat" rgba="0.18 0.28 0.32 1" />
    <material name="gap_hard_mat" rgba="0.12 0.20 0.25 1" />
    <material name="pit_mat" rgba="0.20 0.18 0.16 1" />
    <material name="stair_mat" rgba="0.45 0.42 0.36 1" />
    <material name="platform_mat" rgba="0.39 0.32 0.46 1" />
    <material name="platform_high_mat" rgba="0.48 0.35 0.22 1" />
    <material name="marker_red" rgba="0.85 0.18 0.12 1" />
  </asset>

  <worldbody>
    <light pos="0 0 4.0" dir="0 0 -1" directional="true" />
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="0.9 0.6 0.02" group="0" />
'''


FOOTER = '''  </worldbody>
</mujoco>
'''


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def smooth_height(x: float, y: float, scale: float, phases: tuple[float, float, float, float]) -> float:
    # A compact deterministic substitute for the training Perlin/fractal terrain.
    p0, p1, p2, p3 = phases
    base = (
        0.50 * math.sin(5.2 * x + p0) * math.cos(4.4 * y + p1)
        + 0.30 * math.sin(10.8 * x + 0.6 * y + p2)
        + 0.20 * math.cos(15.5 * y - 0.3 * x + p3)
    )
    normalized = 0.5 + 0.5 * clamp(base, -1.0, 1.0)
    return normalized * scale


def add_flat_section(builder: TerrainBuilder, x: float, length: float = 10.0) -> float:
    x0 = x
    builder.comment("flat: no added obstacle geoms, uses MuJoCo floor plane")
    builder.marker("start_flat", x0)
    x1 = x0 + length
    builder.section("flat", x0, x1, f"length={length:.1f}m")
    return x1 + 0.50


def add_rough_field(builder: TerrainBuilder, rng: random.Random, x: float, length: float = 4.8) -> float:
    x0 = x
    half_width = 1.9
    cell = 0.20
    scale = rng.uniform(0.075, TERRAIN_RANGES["perlin_rough_noise_scale"][1])
    phases = tuple(rng.uniform(-math.pi, math.pi) for _ in range(4))

    builder.comment("perlin_rough: noise_scale=[0.0, 0.1], frequency=20, fractal_octaves=2")
    builder.marker("start_perlin_rough", x0)
    ix = 0
    tx = x0 + cell / 2.0
    while tx < x0 + length:
        iy = 0
        ty = -half_width + cell / 2.0
        while ty < half_width:
            h = smooth_height(tx, ty, scale, phases)
            if h > 0.004:
                builder.box(
                    f"perlin_rough_{ix:02d}_{iy:02d}",
                    (tx, ty, h / 2.0),
                    (cell / 2.0, cell / 2.0, h / 2.0),
                    "rough_mat",
                )
            ty += cell
            iy += 1
        tx += cell
        ix += 1

    x1 = x0 + length
    builder.section("perlin_rough", x0, x1, f"noise_scale~{scale:.3f}, width={2 * half_width:.1f}m")
    return x1 + 0.45


def add_transition_stairs(
    builder: TerrainBuilder,
    x: float,
    from_height: float,
    to_height: float,
    half_width: float,
    name: str,
    material: str,
    max_step_height: float = 0.08,
    step_length: float = 0.28,
) -> float:
    delta = to_height - from_height
    steps = max(1, int(math.ceil(abs(delta) / max_step_height)))
    for i in range(steps):
        alpha = (i + 1) / steps
        h = from_height + alpha * delta
        if h > 0.006:
            builder.plate(f"{name}_{i:02d}", x + step_length / 2.0, step_length, half_width, h, material)
        x += step_length
    return x


def add_gap_section(
    builder: TerrainBuilder,
    rng: random.Random,
    x: float,
    name: str,
    gap_range: tuple[float, float],
    depth_range: tuple[float, float],
    half_width: float,
    material: str,
    slabs: int,
) -> float:
    x0 = x
    depth = rng.uniform(depth_range[0], depth_range[1])
    slab_len_range = (0.70, 1.05) if half_width > 1.2 else (0.55, 0.90)

    builder.comment(f"{name}: gap_distance_range={gap_range}, gap_depth={depth_range}")
    builder.marker(f"start_{name}", x0)
    x = add_transition_stairs(builder, x, 0.0, depth, half_width, f"{name}_entry", material)
    x += 0.20

    actual_gaps: list[float] = []
    for idx in range(slabs):
        slab_len = rng.uniform(*slab_len_range)
        builder.plate(f"{name}_slab_{idx:02d}", x + slab_len / 2.0, slab_len, half_width, depth, material)
        x += slab_len
        if idx != slabs - 1:
            gap = rng.uniform(*gap_range)
            actual_gaps.append(gap)
            x += gap

    x += 0.20
    x = add_transition_stairs(builder, x, depth, 0.0, half_width, f"{name}_exit", material)
    x1 = x
    gap_text = ",".join(f"{g:.2f}" for g in actual_gaps)
    builder.section(name, x0, x1, f"depth={depth:.2f}m, gaps=[{gap_text}], width={2 * half_width:.1f}m")
    return x1 + 0.55


def add_pit_section(builder: TerrainBuilder, rng: random.Random, x: float) -> float:
    x0 = x
    half_width = 1.6
    pit_length = 5.0
    deck_lengths = [rng.uniform(0.85, 1.40) for _ in range(5)]
    deck_heights = [rng.uniform(0.22, 0.45) for _ in deck_lengths]

    builder.comment("pit: direct random-height platforms with 5m floor-level gaps; no stair transition is added")
    builder.marker("start_pit", x0)
    for idx, (deck_len, deck_h) in enumerate(zip(deck_lengths, deck_heights)):
        builder.plate(f"pit_deck_{idx:02d}", x + deck_len / 2.0, deck_len, half_width, deck_h, "pit_mat")
        x += deck_len
        if idx != len(deck_lengths) - 1:
            x += pit_length

    x1 = x
    height_text = ",".join(f"{h:.2f}" for h in deck_heights)
    builder.section("pit", x0, x1, f"platform_heights=[{height_text}]m, pit_spacing={pit_length:.1f}m, width={2 * half_width:.1f}m, no_entry_stairs")
    return x1 + 0.55


def add_stairs_up(builder: TerrainBuilder, rng: random.Random, x: float, name: str = "stairs_up") -> float:
    x0 = x
    half_width = TERRAIN_RANGES["stairs_step_width"] / 2.0
    step_h = rng.uniform(*TERRAIN_RANGES["stairs_step_height"])
    step_l = rng.uniform(*TERRAIN_RANGES["stairs_step_length"])
    num_steps = rng.randint(*TERRAIN_RANGES["stairs_num_steps"])
    platform_l = TERRAIN_RANGES["stairs_platform_length"]

    builder.comment(f"{name}: up-only stairs, training step ranges")
    builder.marker(f"start_{name}", x0)
    x += platform_l * 0.60
    for i in range(1, num_steps + 1):
        builder.plate(f"{name}_{i:02d}", x + step_l / 2.0, step_l, half_width, i * step_h, "stair_mat")
        x += step_l
    top_h = num_steps * step_h
    builder.plate(f"{name}_top_platform", x + platform_l / 2.0, platform_l, half_width, top_h, "stair_mat")
    x += platform_l
    x = add_transition_stairs(builder, x, top_h, 0.0, half_width, f"{name}_exit", "stair_mat")

    x1 = x
    builder.section(name, x0, x1, f"step_h={step_h:.2f}m, step_l={step_l:.2f}m, num_steps={num_steps}, width={2 * half_width:.1f}m")
    return x1 + 0.60


def add_stairs_up_down(builder: TerrainBuilder, rng: random.Random, x: float) -> float:
    x0 = x
    half_width = TERRAIN_RANGES["stairs_step_width"] / 2.0
    step_h = rng.uniform(*TERRAIN_RANGES["stairs_step_height"])
    step_l = rng.uniform(*TERRAIN_RANGES["stairs_step_length"])
    num_steps = rng.randint(*TERRAIN_RANGES["stairs_num_steps"])
    platform_l = TERRAIN_RANGES["stairs_platform_length"]

    builder.comment("stairs_up_down: per_step_height=(0.04,0.20), per_step_length=(0.2,0.4), num_steps=(2,6)")
    builder.marker("start_stairs_up_down", x0)

    x += platform_l
    for i in range(1, num_steps + 1):
        builder.plate(f"stairs_up_{i:02d}", x + step_l / 2.0, step_l, half_width, i * step_h, "stair_mat")
        x += step_l
    builder.plate("stairs_top_platform", x + platform_l / 2.0, platform_l, half_width, num_steps * step_h, "stair_mat")
    x += platform_l
    for i in range(num_steps - 1, -1, -1):
        h = i * step_h
        if h > 0.006:
            builder.plate(f"stairs_down_{i:02d}", x + step_l / 2.0, step_l, half_width, h, "stair_mat")
        x += step_l

    x1 = x
    builder.section("stairs_up_down", x0, x1, f"step_h={step_h:.2f}m, step_l={step_l:.2f}m, num_steps={num_steps}, width={2 * half_width:.1f}m")
    return x1 + 0.60


def add_stairs_down_up(builder: TerrainBuilder, rng: random.Random, x: float) -> float:
    x0 = x
    half_width = TERRAIN_RANGES["stairs_step_width"] / 2.0
    step_h = rng.uniform(*TERRAIN_RANGES["stairs_step_height"])
    step_l = rng.uniform(*TERRAIN_RANGES["stairs_step_length"])
    num_steps = rng.randint(*TERRAIN_RANGES["stairs_num_steps"])
    platform_l = TERRAIN_RANGES["stairs_platform_length"]
    top_h = num_steps * step_h

    builder.comment("stairs_down_up: same ranges as stairs_up_down, starts from an elevated approach")
    builder.marker("start_stairs_down_up", x0)
    x = add_transition_stairs(builder, x, 0.0, top_h, half_width, "stairs_down_up_entry", "stair_mat")
    builder.plate("stairs_down_up_entry_platform", x + platform_l / 2.0, platform_l, half_width, top_h, "stair_mat")
    x += platform_l

    for i in range(num_steps - 1, -1, -1):
        h = i * step_h
        if h > 0.006:
            builder.plate(f"stairs_first_down_{i:02d}", x + step_l / 2.0, step_l, half_width, h, "stair_mat")
        x += step_l
    x += platform_l * 0.45
    for i in range(1, num_steps + 1):
        builder.plate(f"stairs_second_up_{i:02d}", x + step_l / 2.0, step_l, half_width, i * step_h, "stair_mat")
        x += step_l
    x = add_transition_stairs(builder, x, top_h, 0.0, half_width, "stairs_down_up_exit", "stair_mat")

    x1 = x
    builder.section("stairs_down_up", x0, x1, f"step_h={step_h:.2f}m, step_l={step_l:.2f}m, num_steps={num_steps}, width={2 * half_width:.1f}m")
    return x1 + 0.65


def add_pyramid_section(builder: TerrainBuilder, rng: random.Random, x: float, inverted: bool) -> float:
    x0 = x
    half_width = TERRAIN_RANGES["pyramid_platform_width"] / 2.0
    step_w = TERRAIN_RANGES["pyramid_step_width"]
    step_h = rng.uniform(*TERRAIN_RANGES["pyramid_step_height"])
    levels = 4
    name = "pyramid_stairs_inv_high" if inverted else "pyramid_stairs_high"
    material = "platform_high_mat" if inverted else "platform_mat"

    builder.comment(f"{name}: step_height_range=(0.08,0.45), step_width=1.5, platform_width=4.0")
    builder.marker(f"start_{name}", x0)

    if not inverted:
        for i in range(1, levels + 1):
            height = i * step_h
            width = max(0.70, half_width - (i - 1) * 0.28)
            builder.plate(f"{name}_up_{i:02d}", x + step_w / 2.0, step_w, width, height, material)
            x += step_w
        for i in range(levels - 1, -1, -1):
            height = i * step_h
            width = max(0.70, half_width - max(0, i - 1) * 0.28)
            if height > 0.006:
                builder.plate(f"{name}_down_{i:02d}", x + step_w / 2.0, step_w, width, height, material)
            x += step_w
    else:
        top_h = levels * step_h
        x = add_transition_stairs(builder, x, 0.0, top_h, half_width, f"{name}_entry", material, max_step_height=0.10, step_length=0.34)
        for i in range(levels - 1, -1, -1):
            height = i * step_h
            width = max(0.70, half_width - max(0, i - 1) * 0.28)
            if height > 0.006:
                builder.plate(f"{name}_down_{i:02d}", x + step_w / 2.0, step_w, width, height, material)
            x += step_w
        x += step_w * 0.65
        for i in range(1, levels + 1):
            height = i * step_h
            width = max(0.70, half_width - (i - 1) * 0.28)
            builder.plate(f"{name}_up_{i:02d}", x + step_w / 2.0, step_w, width, height, material)
            x += step_w
        x = add_transition_stairs(builder, x, top_h, 0.0, half_width, f"{name}_exit", material, max_step_height=0.10, step_length=0.34)

    x1 = x
    builder.section(name, x0, x1, f"step_h={step_h:.2f}m, step_width={step_w:.1f}m, platform_width={2 * half_width:.1f}m")
    return x1 + 0.70


def add_side_lanes(builder: TerrainBuilder, x0: float, x1: float) -> None:
    builder.comment("visual side lanes outside normal height scan width")
    for idx, y in enumerate((-2.12, 2.12)):
        builder.box(f"side_lane_{idx}", ((x0 + x1) / 2.0, y, 0.018), ((x1 - x0) / 2.0, 0.025, 0.018), "rough_mat", group=1)


def build_scene(seed: int, terrain: str) -> tuple[str, list[SectionSummary]]:
    rng = random.Random(seed)
    builder = TerrainBuilder()
    x = 0.90
    route_start = 0.0

    if terrain == "flat":
        x = add_flat_section(builder, x)
    elif terrain == "rough":
        x = add_rough_field(builder, rng, x, length=8.0)
    elif terrain == "stairs":
        x = add_stairs_up(builder, rng, x)
        x = add_stairs_up_down(builder, rng, x)
        x = add_stairs_down_up(builder, rng, x)
    elif terrain == "stairs-high":
        x = add_pyramid_section(builder, rng, x, inverted=False)
        x = add_pyramid_section(builder, rng, x, inverted=True)
    elif terrain == "pit":
        x = add_pit_section(builder, rng, x)
    elif terrain == "gap":
        x = add_gap_section(
            builder,
            rng,
            x,
            "square_gaps_easy",
            TERRAIN_RANGES["square_gaps_easy_distance"],
            TERRAIN_RANGES["square_gaps_easy_depth"],
            TERRAIN_RANGES["square_gaps_easy_platform_width"] / 2.0,
            "gap_mat",
            slabs=6,
        )
        x = add_gap_section(
            builder,
            rng,
            x,
            "square_gaps_hard",
            TERRAIN_RANGES["square_gaps_hard_distance"],
            TERRAIN_RANGES["square_gaps_hard_depth"],
            TERRAIN_RANGES["square_gaps_hard_platform_width"] / 2.0,
            "gap_hard_mat",
            slabs=5,
        )
    elif terrain == "mixed":
        x = add_rough_field(builder, rng, x, length=4.2)
        x = add_stairs_up(builder, rng, x, name="mixed_stairs_up")
        x = add_gap_section(
            builder,
            rng,
            x,
            "mixed_square_gaps_easy",
            TERRAIN_RANGES["square_gaps_easy_distance"],
            TERRAIN_RANGES["square_gaps_easy_depth"],
            TERRAIN_RANGES["square_gaps_easy_platform_width"] / 2.0,
            "gap_mat",
            slabs=5,
        )
    else:
        raise ValueError(f"unsupported terrain: {terrain}")

    route_end = x + 1.2
    add_side_lanes(builder, route_start, route_end)
    return header() + "".join(builder.geoms) + FOOTER, builder.summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a train-like MuJoCo terrain scene for Go2 sim-to-sim.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--terrain", choices=TERRAIN_CHOICES, default="mixed")
    args = parser.parse_args()

    scene, summaries = build_scene(args.seed, args.terrain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(scene, encoding="utf-8")
    print(f"Generated {args.output}")
    print(f"terrain={args.terrain}")
    print(f"seed={args.seed}")
    for summary in summaries:
        print(f"{summary.name}: x=[{summary.x0:.2f}, {summary.x1:.2f}] {summary.detail}")


if __name__ == "__main__":
    main()

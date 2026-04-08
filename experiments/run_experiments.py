#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib_cache"),
)

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = REPO_ROOT / "img"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


@dataclass(frozen=True)
class SyntheticReward:
    name: str
    dim: int
    lipschitz: float
    base: float
    peaks: tuple[tuple[float, tuple[float, ...], float], ...]

    def mean(self, points: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(points).astype(float)
        values = np.full(pts.shape[0], self.base, dtype=float)
        for height, center, slope in self.peaks:
            center_arr = np.asarray(center, dtype=float)
            dists = np.linalg.norm(pts - center_arr, axis=1)
            values = np.maximum(values, height - slope * dists)
        return np.clip(values, 0.0, 1.0)

    def sample(self, points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        probs = self.mean(points)
        return rng.binomial(1, probs).astype(float)


def make_reward(name: str) -> SyntheticReward:
    if name == "regret_1d":
        return SyntheticReward(
            name=name,
            dim=1,
            lipschitz=4.4,
            base=0.05,
            peaks=(
                (0.95, (0.17,), 2.6),
                (0.89, (0.485,), 4.4),
                (0.84, (0.82,), 2.5),
            ),
        )
    if name == "regret_2d":
        return SyntheticReward(
            name=name,
            dim=2,
            lipschitz=3.4,
            base=0.04,
            peaks=(
                (0.95, (0.20, 0.21), 2.5),
                (0.88, (0.52, 0.74), 3.2),
                (0.84, (0.81, 0.31), 2.7),
            ),
        )
    if name == "pathology_1d":
        return SyntheticReward(
            name=name,
            dim=1,
            lipschitz=5.2,
            base=0.04,
            peaks=(
                (0.96, (0.495,), 5.2),
                (0.86, (0.70,), 1.9),
                (0.78, (0.87,), 2.0),
            ),
        )
    raise ValueError(f"Unknown reward: {name}")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    reward_name: str
    horizon: int
    seeds: int
    n_players: int
    cells_per_dim: int
    phase1_rounds: int
    phase2_rounds: int
    phase2_probe_side: int
    phase3_arm_side: int
    baseline_arm_side: int
    oracle_side: int
    ucb_exploration: float
    delta: float
    dither_scale: float

    @property
    def reward(self) -> SyntheticReward:
        return make_reward(self.reward_name)


@dataclass
class ProtocolRun:
    cumulative_regret: np.ndarray
    collision_fraction: np.ndarray
    consensus: bool
    selection_correct: bool
    target_set: tuple[int, ...]
    true_top_cells: tuple[int, ...]
    seating_rounds: int


class ArmUCBPolicy:
    def __init__(
        self,
        arm_points: np.ndarray,
        rng: np.random.Generator,
        exploration: float,
    ) -> None:
        self.arm_points = arm_points
        self.rng = rng
        self.exploration = exploration
        self.counts = np.zeros(len(arm_points), dtype=np.int64)
        self.sums = np.zeros(len(arm_points), dtype=float)
        self.total_pulls = 0

    def select(self) -> int:
        unseen = np.flatnonzero(self.counts == 0)
        if unseen.size:
            return int(self.rng.choice(unseen))
        means = self.sums / self.counts
        bonus = self.exploration * np.sqrt(
            np.log(self.total_pulls + 1.0) / self.counts
        )
        scores = means + bonus
        max_score = float(np.max(scores))
        candidates = np.flatnonzero(scores >= max_score - 1e-12)
        return int(self.rng.choice(candidates))

    def update(self, arm_index: int, reward: float) -> None:
        self.counts[arm_index] += 1
        self.sums[arm_index] += reward
        self.total_pulls += 1


class PartitionGeometry:
    def __init__(self, dim: int, cells_per_dim: int) -> None:
        self.dim = dim
        self.cells_per_dim = cells_per_dim
        self.width = 1.0 / cells_per_dim
        self.cell_diameter = math.sqrt(dim) * self.width
        self.cells = np.array(list(product(range(cells_per_dim), repeat=dim)), dtype=int)
        self.num_cells = len(self.cells)
        lowers = self.cells.astype(float) * self.width
        uppers = lowers + self.width
        self.cell_lowers = lowers
        self.cell_uppers = uppers
        self.centers = 0.5 * (lowers + uppers)

    def points_in_cell(self, flat_cell: int, points_per_dim: int) -> np.ndarray:
        lower = self.cell_lowers[flat_cell]
        offsets = (np.arange(points_per_dim, dtype=float) + 0.5) / points_per_dim
        coords = [lower[d] + offsets * self.width for d in range(self.dim)]
        return np.array(list(product(*coords)), dtype=float)

    def cloud(self, points_per_dim: int) -> np.ndarray:
        return np.stack(
            [self.points_in_cell(cell_id, points_per_dim) for cell_id in range(self.num_cells)],
            axis=0,
        )

    def global_grid(self, points_per_dim: int) -> tuple[np.ndarray, np.ndarray]:
        cell_cloud = self.cloud(points_per_dim)
        points = cell_cloud.reshape(-1, self.dim)
        cell_ids = np.repeat(np.arange(self.num_cells), cell_cloud.shape[1])
        return points, cell_ids

    def covering_radius(self, points_per_dim: int) -> float:
        return self.width * math.sqrt(self.dim) / (2.0 * points_per_dim)


def nth_largest(values: np.ndarray, n: int) -> float:
    return float(np.partition(values, len(values) - n)[len(values) - n])


def unique_mask_from_cells(chosen_cells: np.ndarray, num_cells: int) -> np.ndarray:
    counts = np.bincount(chosen_cells, minlength=num_cells)
    return counts[chosen_cells] == 1


def make_child_rngs(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    seeds = rng.integers(0, 2**32 - 1, size=n, dtype=np.uint64)
    return [np.random.default_rng(int(seed)) for seed in seeds]


def oracle_top_cells(
    reward: SyntheticReward,
    geometry: PartitionGeometry,
    n_players: int,
    oracle_side: int,
) -> tuple[np.ndarray, tuple[int, ...], float]:
    oracle_cloud = geometry.cloud(oracle_side)
    cell_values = reward.mean(oracle_cloud.reshape(-1, geometry.dim)).reshape(
        geometry.num_cells, -1
    )
    cell_maxima = np.max(cell_values, axis=1)
    top_cells = tuple(np.argsort(cell_maxima)[-n_players:])
    oracle_reward = float(np.sum(np.sort(cell_maxima)[-n_players:]))
    return cell_maxima, top_cells, oracle_reward


def majority_target_set(
    selected_sets: list[np.ndarray],
    lcb_scores: np.ndarray,
    dither: np.ndarray,
    n_players: int,
) -> tuple[int, ...]:
    votes = np.zeros(lcb_scores.shape[1], dtype=float)
    for chosen in selected_sets:
        votes[chosen] += 1.0
    aggregate = votes + 1e-3 * (np.nanmean(lcb_scores, axis=0) + dither)
    target = tuple(sorted(np.argsort(aggregate)[-n_players:]))
    return target


def run_baseline(
    cfg: ExperimentConfig,
    reward: SyntheticReward,
    geometry: PartitionGeometry,
    oracle_reward: float,
    seed: int,
) -> ProtocolRun:
    rng = np.random.default_rng(seed)
    player_rngs = make_child_rngs(rng, cfg.n_players)
    arm_points, arm_cells = geometry.global_grid(cfg.baseline_arm_side)
    policies = [
        ArmUCBPolicy(arm_points=arm_points, rng=player_rngs[j], exploration=cfg.ucb_exploration)
        for j in range(cfg.n_players)
    ]
    cumulative_regret = np.zeros(cfg.horizon, dtype=float)
    collision_fraction = np.zeros(cfg.horizon, dtype=float)
    regret_so_far = 0.0

    for t in range(cfg.horizon):
        arm_choices = np.array([policy.select() for policy in policies], dtype=int)
        chosen_cells = arm_cells[arm_choices]
        unique = unique_mask_from_cells(chosen_cells, geometry.num_cells)
        total_reward = 0.0
        if np.any(unique):
            unique_players = np.flatnonzero(unique)
            rewards = reward.sample(arm_points[arm_choices[unique_players]], rng)
            for local_idx, player in enumerate(unique_players):
                policies[player].update(int(arm_choices[player]), float(rewards[local_idx]))
                total_reward += float(rewards[local_idx])
        regret_so_far += oracle_reward - total_reward
        cumulative_regret[t] = regret_so_far
        collision_fraction[t] = float(np.mean(~unique))

    return ProtocolRun(
        cumulative_regret=cumulative_regret,
        collision_fraction=collision_fraction,
        consensus=True,
        selection_correct=False,
        target_set=tuple(),
        true_top_cells=tuple(),
        seating_rounds=0,
    )


def run_protocol(
    cfg: ExperimentConfig,
    reward: SyntheticReward,
    geometry: PartitionGeometry,
    oracle_reward: float,
    true_top_cells: tuple[int, ...],
    seed: int,
    use_local_peek: bool,
) -> ProtocolRun:
    rng = np.random.default_rng(seed)
    cumulative_regret = np.zeros(cfg.horizon, dtype=float)
    collision_fraction = np.zeros(cfg.horizon, dtype=float)
    regret_so_far = 0.0
    current_round = 0

    phase1_counts = np.zeros(geometry.num_cells, dtype=np.int64)
    phase1_sums = np.zeros(geometry.num_cells, dtype=float)

    def log_round(total_reward: float, colliding_players: int) -> None:
        nonlocal current_round, regret_so_far
        regret_so_far += oracle_reward - total_reward
        cumulative_regret[current_round] = regret_so_far
        collision_fraction[current_round] = colliding_players / cfg.n_players
        current_round += 1

    # Phase I
    for _ in range(min(cfg.phase1_rounds, cfg.horizon)):
        chosen_cells = rng.integers(0, geometry.num_cells, size=cfg.n_players)
        unique = unique_mask_from_cells(chosen_cells, geometry.num_cells)
        colliding_players = int(np.sum(~unique))
        total_reward = 0.0
        if np.any(unique):
            unique_players = np.flatnonzero(unique)
            rewards = reward.sample(geometry.centers[chosen_cells[unique_players]], rng)
            for local_idx, player in enumerate(unique_players):
                cell = int(chosen_cells[player])
                phase1_counts[cell] += 1
                phase1_sums[cell] += float(rewards[local_idx])
                total_reward += float(rewards[local_idx])
        log_round(total_reward=total_reward, colliding_players=colliding_players)

    beta0 = math.log(
        max(4.0 * cfg.n_players * geometry.num_cells * (cfg.phase1_rounds + 1) / cfg.delta, 2.0)
    )
    phase1_means = phase1_sums / np.maximum(1, phase1_counts)
    r0 = np.sqrt(beta0 / (2.0 * np.maximum(1, phase1_counts)))
    lcb0 = phase1_means - r0
    ucb0 = phase1_means + r0 + reward.lipschitz * geometry.cell_diameter / 2.0

    active_mask = np.zeros(geometry.num_cells, dtype=bool)
    threshold = nth_largest(lcb0, cfg.n_players)
    active_mask = ucb0 >= threshold

    if use_local_peek:
        probe_cloud = geometry.cloud(cfg.phase2_probe_side)
        num_probes = probe_cloud.shape[1]
        probe_counts = np.zeros((geometry.num_cells, num_probes), dtype=np.int64)
        probe_sums = np.zeros((geometry.num_cells, num_probes), dtype=float)

        for _ in range(min(cfg.phase2_rounds, max(0, cfg.horizon - current_round))):
            chosen_cells = np.empty(cfg.n_players, dtype=int)
            chosen_probes = np.empty(cfg.n_players, dtype=int)
            for player in range(cfg.n_players):
                candidates = np.flatnonzero(active_mask)
                chosen_cells[player] = int(rng.choice(candidates))
                chosen_probes[player] = int(rng.integers(0, num_probes))
            unique = unique_mask_from_cells(chosen_cells, geometry.num_cells)
            colliding_players = int(np.sum(~unique))
            total_reward = 0.0
            if np.any(unique):
                unique_players = np.flatnonzero(unique)
                points = probe_cloud[chosen_cells[unique_players], chosen_probes[unique_players]]
                rewards = reward.sample(points, rng)
                for local_idx, player in enumerate(unique_players):
                    cell = int(chosen_cells[player])
                    probe = int(chosen_probes[player])
                    probe_counts[cell, probe] += 1
                    probe_sums[cell, probe] += float(rewards[local_idx])
                    total_reward += float(rewards[local_idx])
            log_round(total_reward=total_reward, colliding_players=colliding_players)

        beta1 = math.log(
            max(
                4.0
                * cfg.n_players
                * geometry.num_cells
                * num_probes
                * (cfg.phase2_rounds + 1)
                / cfg.delta,
                2.0,
            )
        )
        probe_means = probe_sums / np.maximum(1, probe_counts)
        r1 = np.sqrt(beta1 / (2.0 * np.maximum(1, probe_counts)))
        lcb1 = np.full(geometry.num_cells, -np.inf, dtype=float)
        covering_bonus = reward.lipschitz * geometry.covering_radius(cfg.phase2_probe_side)
        candidates = np.flatnonzero(active_mask)
        if candidates.size:
            lcb1[candidates] = np.max(probe_means[candidates] - r1[candidates], axis=1)
        selection_scores = lcb1
    else:
        selection_scores = lcb0

    dither = cfg.dither_scale * np.linspace(0.0, 1.0, geometry.num_cells)
    scores = selection_scores.copy()
    scores[~np.isfinite(scores)] = -1e9
    scores += dither
    target_set = tuple(sorted(int(cell) for cell in np.argsort(scores)[-cfg.n_players:]))
    consensus = True

    selection_correct = set(target_set) == set(true_top_cells)

    # Phase II 1/2: seating
    seated = np.zeros(cfg.n_players, dtype=bool)
    assigned_cells = np.full(cfg.n_players, -1, dtype=int)
    seating_rounds = 0
    while current_round < cfg.horizon and not np.all(seated):
        chosen_cells = np.empty(cfg.n_players, dtype=int)
        for player in range(cfg.n_players):
            if seated[player]:
                chosen_cells[player] = assigned_cells[player]
            else:
                chosen_cells[player] = int(rng.choice(target_set))
        unique = unique_mask_from_cells(chosen_cells, geometry.num_cells)
        colliding_players = int(np.sum(~unique))
        total_reward = 0.0
        if np.any(unique):
            unique_players = np.flatnonzero(unique)
            rewards = reward.sample(geometry.centers[chosen_cells[unique_players]], rng)
            for local_idx, player in enumerate(unique_players):
                total_reward += float(rewards[local_idx])
                if not seated[player]:
                    assigned_cells[player] = int(chosen_cells[player])
                    seated[player] = True
        log_round(total_reward=total_reward, colliding_players=colliding_players)
        seating_rounds += 1

    # Phase III: within-cell bandits
    local_cloud = geometry.cloud(cfg.phase3_arm_side)
    player_rngs = make_child_rngs(rng, cfg.n_players)
    local_policies = [
        ArmUCBPolicy(
            arm_points=local_cloud[int(assigned_cells[player])],
            rng=player_rngs[player],
            exploration=cfg.ucb_exploration,
        )
        for player in range(cfg.n_players)
    ]

    while current_round < cfg.horizon:
        chosen_arm_indices = np.array([policy.select() for policy in local_policies], dtype=int)
        points = np.array(
            [
                local_cloud[int(assigned_cells[player]), int(chosen_arm_indices[player])]
                for player in range(cfg.n_players)
            ],
            dtype=float,
        )
        rewards = reward.sample(points, rng)
        total_reward = float(np.sum(rewards))
        for player, policy in enumerate(local_policies):
            policy.update(int(chosen_arm_indices[player]), float(rewards[player]))
        log_round(total_reward=total_reward, colliding_players=0)

    return ProtocolRun(
        cumulative_regret=cumulative_regret,
        collision_fraction=collision_fraction,
        consensus=consensus,
        selection_correct=selection_correct,
        target_set=tuple(sorted(target_set)),
        true_top_cells=tuple(sorted(true_top_cells)),
        seating_rounds=seating_rounds,
    )


def aggregate_runs(runs: list[ProtocolRun]) -> dict[str, object]:
    regrets = np.stack([run.cumulative_regret for run in runs], axis=0)
    collisions = np.stack([run.collision_fraction for run in runs], axis=0)
    return {
        "mean_regret": regrets.mean(axis=0),
        "stderr_regret": regrets.std(axis=0, ddof=1) / math.sqrt(len(runs)),
        "mean_collision": collisions.mean(axis=0),
        "stderr_collision": collisions.std(axis=0, ddof=1) / math.sqrt(len(runs)),
        "consensus_rate": float(np.mean([run.consensus for run in runs])),
        "selection_accuracy": float(np.mean([run.selection_correct for run in runs])),
        "mean_seating_rounds": float(np.mean([run.seating_rounds for run in runs])),
    }


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_regret_panels(
    results: dict[str, dict[str, object]],
    configs: list[ExperimentConfig],
    path: Path,
) -> None:
    colors = {"protocol": "#1f77b4", "baseline": "#d62728"}
    fig, axes = plt.subplots(1, len(configs), figsize=(11.2, 4.2), sharey=False)
    if len(configs) == 1:
        axes = [axes]
    for ax, cfg in zip(axes, configs):
        time = np.arange(1, cfg.horizon + 1)
        ours = results[cfg.name]["protocol"]
        base = results[cfg.name]["baseline"]
        for key, label, style in (
            ("protocol", "Our protocol", "-"),
            ("baseline", "Independent single-agent baseline", "--"),
        ):
            stats = results[cfg.name][key]
            mean = np.asarray(stats["mean_regret"])
            stderr = np.asarray(stats["stderr_regret"])
            ax.plot(
                time,
                mean,
                linestyle=style,
                linewidth=2.5,
                color=colors[key],
                label=label,
            )
            ax.fill_between(
                time,
                mean - 1.96 * stderr,
                mean + 1.96 * stderr,
                color=colors[key],
                alpha=0.14,
            )
        ax.set_title("1D" if cfg.reward.dim == 1 else "2D")
        ax.set_xlabel("Rounds")
        ax.set_ylabel("Cumulative regret")
        ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_collision_panels(
    results: dict[str, dict[str, object]],
    configs: list[ExperimentConfig],
    path: Path,
) -> None:
    colors = {"protocol": "#1f77b4", "baseline": "#d62728"}
    fig, axes = plt.subplots(1, len(configs), figsize=(11.2, 4.1), sharey=True)
    if len(configs) == 1:
        axes = [axes]
    for ax, cfg in zip(axes, configs):
        time = np.arange(1, cfg.horizon + 1)
        for key, label, style in (
            ("protocol", "Our protocol", "-"),
            ("baseline", "Independent baseline", "--"),
        ):
            stats = results[cfg.name][key]
            mean = moving_average(np.asarray(stats["mean_collision"]), window=200)
            stderr = moving_average(np.asarray(stats["stderr_collision"]), window=200)
            ax.plot(
                time,
                mean,
                linestyle=style,
                linewidth=2.3,
                color=colors[key],
                label=label,
            )
            ax.fill_between(
                time,
                np.clip(mean - 1.96 * stderr, 0.0, 1.0),
                np.clip(mean + 1.96 * stderr, 0.0, 1.0),
                color=colors[key],
                alpha=0.12,
            )
        ax.axvline(cfg.phase1_rounds, color="0.35", linestyle=":", linewidth=1.1)
        ax.axvline(cfg.phase1_rounds + cfg.phase2_rounds, color="0.35", linestyle=":", linewidth=1.1)
        ax.set_title("1D" if cfg.reward.dim == 1 else "2D")
        ax.set_xlabel("Rounds")
        ax.set_ylabel("Fraction of colliding players")
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_pathology_panel(
    reward: SyntheticReward,
    geometry: PartitionGeometry,
    results: dict[str, object],
    path: Path,
) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.4, 4.8))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.18, wspace=0.18)

    xs = np.linspace(0.0, 1.0, 1600)[:, None]
    ys = reward.mean(xs)
    ax_left.plot(xs[:, 0], ys, color="#2c7fb8", linewidth=2.4)
    for cell_id in range(geometry.num_cells):
        if cell_id > 0:
            xline = geometry.cell_lowers[cell_id, 0]
            ax_left.axvline(xline, color="0.82", linewidth=0.9)
        ax_left.text(
            geometry.centers[cell_id, 0],
            -0.085,
            f"C{cell_id + 1}",
            transform=ax_left.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color="0.35",
        )
        ax_left.scatter(
            geometry.centers[cell_id, 0],
            reward.mean(geometry.centers[cell_id : cell_id + 1])[0],
            color="#d95f0e",
            s=28,
            zorder=3,
        )
    boundary_x = geometry.cell_lowers[3, 0]
    ax_left.axvline(boundary_x, color="#2ca25f", linewidth=1.7, linestyle="--", alpha=0.9)
    ax_left.annotate(
        "largest peak lies near\nthe C3/C4 boundary",
        xy=(boundary_x, 0.95),
        xytext=(0.18, 0.98),
        textcoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#1b7837",
        arrowprops=dict(arrowstyle="->", color="#1b7837", lw=1.0),
    )
    ax_left.set_title("Peak near the C3/C4 boundary")
    ax_left.set_xlabel("x")
    ax_left.set_ylabel("mu(x)")
    ax_left.set_xlim(0.0, 1.0)
    ax_left.set_ylim(0.0, 1.02)

    center_values = np.asarray(results["center_values"])
    local_peek_values = np.asarray(results["local_peek_values"])
    true_maxima = np.asarray(results["true_maxima"])
    center_cells = sorted(int(cell) for cell in results["center_top_cells"])
    local_cells = sorted(int(cell) for cell in results["local_peek_top_cells"])
    true_cells = sorted(int(cell) for cell in results["true_top_cells"])
    x = np.arange(len(center_values))
    width = 0.24
    true_start = true_cells[0] - 1 - 0.5
    true_end = true_cells[-1] - 1 + 0.5
    center_start = center_cells[0] - 1 - 0.5
    center_end = center_cells[-1] - 1 + 0.5
    ax_right.axvspan(true_start, true_end, color="#2ca25f", alpha=0.06, zorder=0)
    ax_right.axvspan(center_start, center_end, color="#e6550d", alpha=0.05, zorder=0)
    ax_right.bar(x - width, center_values, width=width, color="#e6550d", label="Center value")
    ax_right.bar(x, local_peek_values, width=width, color="#2ca25f", label="Local-peek score")
    ax_right.bar(x + width, true_maxima, width=width, color="#756bb1", label="True cell maximum")
    ax_right.set_xticks(x)
    ax_right.set_xticklabels([f"C{idx + 1}" for idx in x])
    ax_right.set_ylabel("Cell score")
    ax_right.set_title("Centers favor C5/C6, but local peek recovers C3/C4")
    ax_right.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.7,
    )
    ax_right.set_ylim(0.0, 1.12)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_configs(mode: str) -> tuple[list[ExperimentConfig], ExperimentConfig]:
    if mode == "quick":
        regret_cfgs = [
            ExperimentConfig(
                name="regret_1d",
                reward_name="regret_1d",
                horizon=6000,
                seeds=4,
                n_players=3,
                cells_per_dim=8,
                phase1_rounds=220,
                phase2_rounds=500,
                phase2_probe_side=7,
                phase3_arm_side=7,
                baseline_arm_side=7,
                oracle_side=31,
                ucb_exploration=0.8,
                delta=0.05,
                dither_scale=0.02,
            ),
            ExperimentConfig(
                name="regret_2d",
                reward_name="regret_2d",
                horizon=6000,
                seeds=4,
                n_players=3,
                cells_per_dim=4,
                phase1_rounds=380,
                phase2_rounds=750,
                phase2_probe_side=5,
                phase3_arm_side=5,
                baseline_arm_side=5,
                oracle_side=19,
                ucb_exploration=0.9,
                delta=0.05,
                dither_scale=0.025,
            ),
        ]
        pathology_cfg = ExperimentConfig(
            name="pathology_1d",
            reward_name="pathology_1d",
            horizon=6000,
            seeds=18,
            n_players=2,
            cells_per_dim=6,
            phase1_rounds=220,
            phase2_rounds=500,
            phase2_probe_side=9,
            phase3_arm_side=7,
            baseline_arm_side=7,
            oracle_side=41,
            ucb_exploration=0.8,
            delta=0.05,
            dither_scale=0.018,
        )
        return regret_cfgs, pathology_cfg

    regret_cfgs = [
        ExperimentConfig(
            name="regret_1d",
            reward_name="regret_1d",
            horizon=10000,
            seeds=5,
            n_players=3,
            cells_per_dim=8,
            phase1_rounds=260,
            phase2_rounds=700,
            phase2_probe_side=7,
            phase3_arm_side=7,
            baseline_arm_side=7,
            oracle_side=41,
            ucb_exploration=0.8,
            delta=0.05,
            dither_scale=0.02,
        ),
        ExperimentConfig(
            name="regret_2d",
            reward_name="regret_2d",
            horizon=10000,
            seeds=5,
            n_players=3,
            cells_per_dim=4,
            phase1_rounds=520,
            phase2_rounds=1100,
            phase2_probe_side=5,
            phase3_arm_side=5,
            baseline_arm_side=5,
            oracle_side=23,
            ucb_exploration=0.95,
            delta=0.05,
            dither_scale=0.03,
        ),
    ]
    pathology_cfg = ExperimentConfig(
        name="pathology_1d",
        reward_name="pathology_1d",
        horizon=8000,
        seeds=32,
        n_players=2,
        cells_per_dim=6,
        phase1_rounds=260,
        phase2_rounds=700,
        phase2_probe_side=9,
        phase3_arm_side=7,
        baseline_arm_side=7,
        oracle_side=51,
        ucb_exploration=0.8,
        delta=0.05,
        dither_scale=0.02,
    )
    return regret_cfgs, pathology_cfg


def save_summary(path: Path, payload: dict[str, object]) -> None:
    def default_encoder(obj: object) -> object:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    path.write_text(json.dumps(payload, indent=2, default=default_encoder))


def run_regret_suite(configs: list[ExperimentConfig]) -> dict[str, dict[str, object]]:
    suite_results: dict[str, dict[str, object]] = {}
    for cfg in configs:
        print(f"[run] {cfg.name}: {cfg.seeds} seeds, horizon={cfg.horizon}", flush=True)
        reward = cfg.reward
        geometry = PartitionGeometry(dim=reward.dim, cells_per_dim=cfg.cells_per_dim)
        _, true_top_cells, oracle_reward = oracle_top_cells(
            reward=reward,
            geometry=geometry,
            n_players=cfg.n_players,
            oracle_side=cfg.oracle_side,
        )
        baseline_runs: list[ProtocolRun] = []
        protocol_runs: list[ProtocolRun] = []
        for seed in range(cfg.seeds):
            baseline_runs.append(
                run_baseline(
                    cfg=cfg,
                    reward=reward,
                    geometry=geometry,
                    oracle_reward=oracle_reward,
                    seed=7919 + seed,
                )
            )
            protocol_runs.append(
                run_protocol(
                    cfg=cfg,
                    reward=reward,
                    geometry=geometry,
                    oracle_reward=oracle_reward,
                    true_top_cells=true_top_cells,
                    seed=104729 + seed,
                    use_local_peek=True,
                )
            )
        suite_results[cfg.name] = {
            "config": asdict(cfg),
            "oracle_reward": oracle_reward,
            "true_top_cells": list(true_top_cells),
            "baseline": aggregate_runs(baseline_runs),
            "protocol": aggregate_runs(protocol_runs),
        }
    return suite_results


def run_pathology_suite(cfg: ExperimentConfig) -> dict[str, object]:
    print("[run] pathology illustration", flush=True)
    reward = cfg.reward
    geometry = PartitionGeometry(dim=reward.dim, cells_per_dim=cfg.cells_per_dim)
    true_maxima, true_top_cells, _ = oracle_top_cells(
        reward=reward,
        geometry=geometry,
        n_players=cfg.n_players,
        oracle_side=cfg.oracle_side,
    )
    center_values = reward.mean(geometry.centers)
    local_peek_cloud = geometry.cloud(cfg.phase2_probe_side)
    local_peek_values = reward.mean(local_peek_cloud.reshape(-1, reward.dim)).reshape(
        geometry.num_cells, -1
    )
    local_peek_values = np.max(local_peek_values, axis=1)
    return {
        "config": asdict(cfg),
        "true_top_cells": list(int(idx + 1) for idx in sorted(true_top_cells)),
        "center_top_cells": list(
            int(idx + 1) for idx in sorted(np.argsort(center_values)[-cfg.n_players:])
        ),
        "local_peek_top_cells": list(
            int(idx + 1) for idx in sorted(np.argsort(local_peek_values)[-cfg.n_players:])
        ),
        "center_values": center_values,
        "local_peek_values": local_peek_values,
        "true_maxima": true_maxima,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic experiments for the paper.")
    parser.add_argument(
        "--mode",
        choices=("quick", "paper"),
        default="paper",
        help="Use a lightweight debug configuration or the full paper configuration.",
    )
    args = parser.parse_args()

    configure_matplotlib()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    regret_cfgs, pathology_cfg = make_configs(args.mode)
    regret_results = run_regret_suite(regret_cfgs)
    pathology_results = run_pathology_suite(pathology_cfg)

    print("[plot] regret", flush=True)
    plot_regret_panels(
        results=regret_results,
        configs=regret_cfgs,
        path=IMG_DIR / "empirical_regret_main.png",
    )
    print("[plot] collisions", flush=True)
    plot_collision_panels(
        results=regret_results,
        configs=regret_cfgs,
        path=IMG_DIR / "empirical_collisions.png",
    )
    print("[plot] pathology", flush=True)
    plot_pathology_panel(
        reward=pathology_cfg.reward,
        geometry=PartitionGeometry(
            dim=pathology_cfg.reward.dim, cells_per_dim=pathology_cfg.cells_per_dim
        ),
        results=pathology_results,
        path=IMG_DIR / "empirical_pathology.png",
    )

    summary = {
        "mode": args.mode,
        "regret": regret_results,
        "pathology": pathology_results,
    }
    print("[write] summary", flush=True)
    save_summary(RESULTS_DIR / f"summary_{args.mode}.json", summary)

    print(f"Wrote figures to {IMG_DIR}")
    print(f"Wrote summary to {RESULTS_DIR / f'summary_{args.mode}.json'}")
    for cfg in regret_cfgs:
        ours = regret_results[cfg.name]["protocol"]
        base = regret_results[cfg.name]["baseline"]
        print(
            f"{cfg.name}: consensus={ours['consensus_rate']:.2f}, "
            f"selection={ours['selection_accuracy']:.2f}, "
            f"final regret ours={float(ours['mean_regret'][-1]):.1f}, "
            f"baseline={float(base['mean_regret'][-1]):.1f}"
        )
    print(
        "pathology: "
        f"centers={pathology_results['center_top_cells']}, "
        f"local-peek={pathology_results['local_peek_top_cells']}, "
        f"truth={pathology_results['true_top_cells']}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - debug aid for the paper pipeline
        print(f"Experiment runner failed: {exc}")
        traceback.print_exc()
        raise

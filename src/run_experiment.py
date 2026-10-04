#!/usr/bin/env python3
"""
The script uses only Python's standard library so it runs on a fresh Ubuntu
installation. It generates a shifted page-reference trace, evaluates four
policies, and writes CSV/JSON/SVG results.
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FRAMES = 6
SEED = 307


def make_trace(seed: int = SEED) -> tuple[list[int], int]:
    """Return a trace with a locality-heavy first half and random second half."""
    rng = random.Random(seed)
    first: list[int] = []
    for i in range(240):
        # A small hot set with occasional sequential scans.
        if i % 40 in range(30, 36):
            first.extend(range(0, 12))
        else:
            first.append(rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    first = first[:240]
    second = [rng.randrange(0, 18) for _ in range(240)]
    return first + second, len(first)


def next_use(trace: list[int], position: int, page: int) -> int | None:
    for i in range(position + 1, len(trace)):
        if trace[i] == page:
            return i
    return None


def classical_policy(trace: list[int], frames: int, name: str) -> list[bool]:
    resident: list[int] = []
    last_seen: dict[int, int] = {}
    hits: list[bool] = []
    for i, page in enumerate(trace):
        hit = page in resident
        hits.append(hit)
        if not hit:
            if len(resident) == frames:
                if name == "FIFO":
                    victim = resident.pop(0)
                elif name == "LRU":
                    victim = min(resident, key=lambda p: last_seen.get(p, -1))
                elif name == "Optimal":
                    victim = max(resident, key=lambda p: next_use(trace, i, p) or math.inf)
                else:
                    raise ValueError(name)
                del last_seen[victim]
            resident.append(page)
        last_seen[page] = i
    return hits


@dataclass
class CentroidClassifier:
    """A tiny two-class nearest-centroid classifier.

    Each training sample is a candidate page. Label 1 means that the candidate
    is the oracle victim at that point; label 0 means it is not. Features are
    normalized recency, normalized frequency, and normalized age. The model is
    intentionally small and transparent for a classroom experiment.
    """

    positive: list[float] | None = None
    negative: list[float] | None = None

    def fit(self, samples: Iterable[tuple[list[float], int]]) -> None:
        groups = {0: [], 1: []}
        for features, label in samples:
            groups[label].append(features)
        self.positive = self._mean(groups[1])
        self.negative = self._mean(groups[0])

    @staticmethod
    def _mean(rows: list[list[float]]) -> list[float]:
        if not rows:
            return [0.0, 0.0, 0.0]
        return [sum(row[j] for row in rows) / len(rows) for j in range(3)]

    def predict_victim_score(self, features: list[float]) -> float:
        if self.positive is None or self.negative is None:
            raise RuntimeError("classifier has not been trained")
        d_pos = sum((a - b) ** 2 for a, b in zip(features, self.positive))
        d_neg = sum((a - b) ** 2 for a, b in zip(features, self.negative))
        return d_neg - d_pos  # higher means more like the victim class


def candidate_features(resident: list[int], page: int, position: int,
                       trace: list[int], last_seen: dict[int, int],
                       counts: Counter[int]) -> list[float]:
    recency = position - last_seen.get(page, position)
    max_recency = max((position - last_seen.get(p, position) for p in resident), default=1)
    frequency = counts[page]
    max_frequency = max((counts[p] for p in resident), default=1)
    age = position - last_seen.get(page, position)
    return [recency / max(max_recency, 1), frequency / max(max_frequency, 1), age / max(position + 1, 1)]


def training_samples(trace: list[int], frames: int, end: int) -> list[tuple[list[float], int]]:
    resident: list[int] = []
    last_seen: dict[int, int] = {}
    counts: Counter[int] = Counter()
    samples: list[tuple[list[float], int]] = []
    for i, page in enumerate(trace[:end]):
        if page not in resident and len(resident) == frames:
            oracle = max(resident, key=lambda p: next_use(trace, i, p) or math.inf)
            for candidate in resident:
                features = candidate_features(resident, candidate, i, trace, last_seen, counts)
                samples.append((features, int(candidate == oracle)))
            resident.remove(oracle)
        if page not in resident:
            resident.append(page)
        last_seen[page] = i
        counts[page] += 1
    return samples


def learned_policy(trace: list[int], frames: int, train_end: int) -> list[bool]:
    model = CentroidClassifier()
    model.fit(training_samples(trace, frames, train_end))
    resident: list[int] = []
    last_seen: dict[int, int] = {}
    counts: Counter[int] = Counter()
    hits: list[bool] = []
    for i, page in enumerate(trace):
        hit = page in resident
        hits.append(hit)
        if not hit:
            if len(resident) == frames:
                victim = max(
                    resident,
                    key=lambda p: model.predict_victim_score(
                        candidate_features(resident, p, i, trace, last_seen, counts)
                    ),
                )
                resident.remove(victim)
            resident.append(page)
        last_seen[page] = i
        counts[page] += 1
    return hits


def write_csv(trace: list[int], shift: int, policies: dict[str, list[bool]]) -> None:
    with (RESULTS / "policy_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["policy", "phase", "references", "hits", "faults", "hit_ratio"])
        for name, hits in policies.items():
            for phase, start, end in [("before_shift", 0, shift), ("after_shift", shift, len(trace))]:
                segment = hits[start:end]
                faults = len(segment) - sum(segment)
                writer.writerow([name, phase, len(segment), sum(segment), faults, f"{sum(segment)/len(segment):.4f}"])


def write_trace(trace: list[int], shift: int) -> None:
    with (RESULTS / "access_trace.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "page", "phase"])
        for i, page in enumerate(trace):
            writer.writerow([i, page, "before_shift" if i < shift else "after_shift"])


def write_svg(policies: dict[str, list[bool]]) -> None:
    width, height, left, top = 900, 470, 70, 45
    colors = {"FIFO": "#c0392b", "LRU": "#2471a3", "Optimal": "#239b56", "Learned": "#8e44ad"}
    max_n = max(len(v) for v in policies.values())
    def x(i: int) -> float: return left + i * (width - left - 30) / (max_n - 1)
    def y(v: float) -> float: return top + (1 - v) * (height - top - 55)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="70" y="25" font-family="sans-serif" font-size="18" font-weight="bold">Hit ratio over the shifted access trace</text>',
             f'<line x1="{left}" y1="{y(0)}" x2="{width-30}" y2="{y(0)}" stroke="#555"/>',
             f'<line x1="{left}" y1="{y(1)}" x2="{left}" y2="{y(0)}" stroke="#555"/>',
             f'<text x="18" y="{y(1)+5}" font-family="sans-serif" font-size="12">1.0</text>',
             f'<text x="25" y="{y(0)+5}" font-family="sans-serif" font-size="12">0.0</text>']
    shift_x = x(239.5)
    parts.append(f'<line x1="{shift_x:.1f}" y1="{top}" x2="{shift_x:.1f}" y2="{y(0)}" stroke="#777" stroke-dasharray="5,5"/>')
    parts.append(f'<text x="{shift_x+5:.1f}" y="{top+15}" font-family="sans-serif" font-size="12">workload shift</text>')
    for name, hits in policies.items():
        points = []
        window = 24
        for i in range(len(hits)):
            start = max(0, i - window + 1)
            ratio = sum(hits[start:i+1]) / (i - start + 1)
            points.append(f"{x(i):.1f},{y(ratio):.1f}")
        parts.append(f'<polyline fill="none" stroke="{colors[name]}" stroke-width="2" points="{" ".join(points)}"/>')
    for j, (name, color) in enumerate(colors.items()):
        lx = 590 + (j % 2) * 145
        ly = 405 + (j // 2) * 23
        parts.append(f'<line x1="{lx}" y1="{ly-5}" x2="{lx+20}" y2="{ly-5}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+27}" y="{ly}" font-family="sans-serif" font-size="12">{name}</text>')
    parts.append('<text x="410" y="455" font-family="sans-serif" font-size="12">reference number</text></svg>')
    (RESULTS / "hit_ratio.svg").write_text("\n".join(parts))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    trace, shift = make_trace()
    policies = {
        "FIFO": classical_policy(trace, FRAMES, "FIFO"),
        "LRU": classical_policy(trace, FRAMES, "LRU"),
        "Optimal": classical_policy(trace, FRAMES, "Optimal"),
        "Learned": learned_policy(trace, FRAMES, shift),
    }
    write_trace(trace, shift)
    write_csv(trace, shift, policies)
    write_svg(policies)
    summary = {name: {"faults": len(hits) - sum(hits), "hit_ratio": round(sum(hits) / len(hits), 4)} for name, hits in policies.items()}
    (RESULTS / "summary.json").write_text(json.dumps({"frames": FRAMES, "trace_length": len(trace), "shift_at": shift, "policies": summary}, indent=2) + "\n")
    print("Experiment complete. Results written to results/")


if __name__ == "__main__":
    main()

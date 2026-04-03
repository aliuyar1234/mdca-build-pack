from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(payload: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def quantile_linear(values: np.ndarray, q: float, *, axis: int | None = None) -> np.ndarray | float:
    try:
        return np.quantile(values, q, axis=axis, method="linear")
    except TypeError:
        return np.quantile(values, q, axis=axis, interpolation="linear")


def _fit_isotonic_nonincreasing(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if xs.ndim != 1 or ys.ndim != 1:
        raise ValueError("isotonic inputs must be one-dimensional")
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) == 0:
        return ys.copy()

    blocks: list[dict[str, float | int]] = []
    for index, y_value in enumerate((-ys).tolist()):
        blocks.append(
            {
                "start": index,
                "end": index,
                "weight": 1.0,
                "value": y_value,
            }
        )
        while len(blocks) >= 2 and float(blocks[-2]["value"]) > float(blocks[-1]["value"]):
            right = blocks.pop()
            left = blocks.pop()
            merged_weight = float(left["weight"]) + float(right["weight"])
            merged_value = (
                float(left["weight"]) * float(left["value"])
                + float(right["weight"]) * float(right["value"])
            ) / merged_weight
            blocks.append(
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "weight": merged_weight,
                    "value": merged_value,
                }
            )

    fitted = np.zeros_like(ys, dtype=float)
    for block in blocks:
        fitted[int(block["start"]) : int(block["end"]) + 1] = -float(block["value"])
    return fitted


@dataclass(frozen=True, slots=True)
class FrontierResult:
    split: str
    bin_width_bits: int
    min_family_count: int
    available_families: tuple[str, ...]
    valid_bin_centers: tuple[float, ...]
    valid_bin_quantiles: tuple[float, ...]
    isotonic_values: tuple[float, ...]
    domain: tuple[float, float] | None
    bin_summaries: tuple[dict[str, Any], ...]

    def structural_bin_start(self, structural_bits: float) -> float:
        width = float(self.bin_width_bits)
        return float(width * math.floor(structural_bits / width))

    def valid_bin_starts(self) -> tuple[float, ...]:
        return tuple(
            float(summary["bin_start"])
            for summary in self.bin_summaries
            if bool(summary["valid"])
        )

    def _valid_bin_summary(self, structural_bits: float) -> dict[str, Any] | None:
        bin_start = self.structural_bin_start(structural_bits)
        for summary in self.bin_summaries:
            if not bool(summary["valid"]):
                continue
            if math.isclose(float(summary["bin_start"]), bin_start, rel_tol=0.0, abs_tol=1e-9):
                return summary
        return None

    def defined_at(self, structural_bits: float) -> bool:
        return self._valid_bin_summary(structural_bits) is not None

    def evaluate(self, structural_bits: float) -> float | None:
        summary = self._valid_bin_summary(structural_bits)
        if summary is None:
            return None
        isotonic_value = summary.get("isotonic_value")
        if isotonic_value is None:
            return None
        return float(isotonic_value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "bin_width_bits": self.bin_width_bits,
            "min_family_count": self.min_family_count,
            "available_families": list(self.available_families),
            "valid_bin_centers": list(self.valid_bin_centers),
            "valid_bin_quantiles": list(self.valid_bin_quantiles),
            "isotonic_values": list(self.isotonic_values),
            "domain": list(self.domain) if self.domain is not None else None,
            "bin_summaries": list(self.bin_summaries),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrontierResult":
        domain = data.get("domain")
        return cls(
            split=str(data["split"]),
            bin_width_bits=int(data["bin_width_bits"]),
            min_family_count=int(data["min_family_count"]),
            available_families=tuple(str(value) for value in data["available_families"]),
            valid_bin_centers=tuple(float(value) for value in data["valid_bin_centers"]),
            valid_bin_quantiles=tuple(float(value) for value in data["valid_bin_quantiles"]),
            isotonic_values=tuple(float(value) for value in data["isotonic_values"]),
            domain=None if domain is None else (float(domain[0]), float(domain[1])),
            bin_summaries=tuple(dict(item) for item in data["bin_summaries"]),
        )


def build_balanced_frontier(
    *,
    null_records: list[dict[str, Any]],
    split: str,
    bin_width_bits: int,
    min_family_count: int,
    balance_seed: int,
    required_families: tuple[str, ...] | None = None,
) -> FrontierResult:
    if required_families is None:
        families = tuple(
            sorted({str(record["null_family"]) for record in null_records})
        )
    else:
        families = required_families

    bins: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for record in null_records:
        null_family = str(record["null_family"])
        if null_family not in families:
            continue
        structural_bits = float(record["code_lengths"]["total_structural_bits"])
        bin_start = int(bin_width_bits * math.floor(structural_bits / bin_width_bits))
        bins.setdefault(bin_start, {}).setdefault(null_family, []).append(record)

    bin_summaries: list[dict[str, Any]] = []
    valid_bin_centers: list[float] = []
    valid_bin_quantiles: list[float] = []
    for bin_start in sorted(bins):
        family_records = bins[bin_start]
        counts_by_family = {
            family: len(family_records.get(family, []))
            for family in families
        }
        k_j = min(counts_by_family.values()) if counts_by_family else 0
        bin_summary: dict[str, Any] = {
            "bin_start": float(bin_start),
            "bin_end": float(bin_start + bin_width_bits),
            "bin_center": float(bin_start + bin_width_bits / 2.0),
            "counts_by_family": counts_by_family,
            "k_j": int(k_j),
            "valid": bool(k_j >= min_family_count),
            "selected_null_ids_by_family": {},
            "balanced_quantile": None,
        }
        if k_j >= min_family_count:
            selected_records: list[dict[str, Any]] = []
            for family in families:
                family_candidates = list(family_records[family])
                family_candidates.sort(
                    key=lambda record: (
                        int(
                            sha256_for_balance(
                                balance_seed,
                                split,
                                bin_start,
                                family,
                                record["candidate_id"],
                            )[:12],
                            16,
                        ),
                        record["candidate_id"],
                    )
                )
                chosen = family_candidates[:k_j]
                bin_summary["selected_null_ids_by_family"][family] = [
                    record["candidate_id"] for record in chosen
                ]
                selected_records.extend(chosen)
            residuals = np.array(
                [float(record["residual_bits"][split]) for record in selected_records],
                dtype=float,
            )
            balanced_quantile = float(quantile_linear(residuals, 0.05))
            bin_summary["balanced_quantile"] = balanced_quantile
            valid_bin_centers.append(bin_summary["bin_center"])
            valid_bin_quantiles.append(balanced_quantile)
        bin_summaries.append(bin_summary)

    xs = np.array(valid_bin_centers, dtype=float)
    ys = np.array(valid_bin_quantiles, dtype=float)
    isotonic_values = _fit_isotonic_nonincreasing(xs, ys) if len(xs) else np.array([], dtype=float)
    domain: tuple[float, float] | None
    if len(xs):
        domain = (float(xs[0]), float(xs[-1]))
    else:
        domain = None

    valid_index = 0
    for bin_summary in bin_summaries:
        if bin_summary["valid"]:
            bin_summary["isotonic_value"] = float(isotonic_values[valid_index])
            valid_index += 1
        else:
            bin_summary["isotonic_value"] = None

    return FrontierResult(
        split=split,
        bin_width_bits=bin_width_bits,
        min_family_count=min_family_count,
        available_families=families,
        valid_bin_centers=tuple(float(value) for value in xs.tolist()),
        valid_bin_quantiles=tuple(float(value) for value in ys.tolist()),
        isotonic_values=tuple(float(value) for value in isotonic_values.tolist()),
        domain=domain,
        bin_summaries=tuple(bin_summaries),
    )


def sha256_for_balance(*parts: object) -> str:
    payload = "||".join(str(part) for part in parts)
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def maybe_render_frontier_plot(
    *,
    run_dir: str | Path,
    split: str,
    candidate_records: list[dict[str, Any]],
    null_records: list[dict[str, Any]],
    frontier: FrontierResult,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    output_path = Path(run_dir) / f"frontier_{split}.png"
    plt.figure(figsize=(8, 5))
    for null_family in sorted({str(record["null_family"]) for record in null_records}):
        family_records = [record for record in null_records if record["null_family"] == null_family]
        plt.scatter(
            [record["code_lengths"]["total_structural_bits"] for record in family_records],
            [record["residual_bits"][split] for record in family_records],
            s=14,
            alpha=0.45,
            label=f"null:{null_family}",
        )
    plt.scatter(
        [record["code_lengths"]["total_structural_bits"] for record in candidate_records],
        [record["residual_bits"][split] for record in candidate_records],
        s=22,
        alpha=0.8,
        marker="x",
        label="candidates",
    )
    if frontier.valid_bin_centers:
        xs = np.array(frontier.valid_bin_centers, dtype=float)
        ys = np.array(frontier.isotonic_values, dtype=float)
        plt.plot(xs, ys, color="black", linewidth=2, label="balanced null frontier")
    if frontier.domain is not None:
        plt.axvspan(frontier.domain[0], frontier.domain[1], color="gray", alpha=0.08)
    plt.xlabel("L_struct")
    plt.ylabel(f"L_res_{split}")
    plt.title(f"S1 planted frontier - {split}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path)

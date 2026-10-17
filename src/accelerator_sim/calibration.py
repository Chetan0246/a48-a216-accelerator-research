from __future__ import annotations

import csv
import math
import os
import warnings
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass(frozen=True)
class BenchmarkRecord:
    benchmark_id: str
    model_name: str
    format_or_quant: str
    context_len: int
    gen_len: int
    batch_size: int
    software_version: str
    hardware_name: str
    hardware_details: str
    execution_settings: str
    measured_ttft_ms: float | None
    measured_tpot_ms: float | None
    measured_throughput_tps: float | None
    source: str
    source_url: str
    benchmark_conditions: str
    verification_status: str = "unverified"  # "verified", "unverified", or "provisional"
    notes: str = ""

    @property
    def is_verified(self) -> bool:
        return self.verification_status.strip().lower() == "verified"


class BenchmarkManifest:
    """Loads, validates, and manages empirical benchmark records without fabrication."""

    REQUIRED_HEADERS: ClassVar[list[str]] = [
        "benchmark_id",
        "model_name",
        "format_or_quant",
        "context_len",
        "gen_len",
        "batch_size",
        "software_version",
        "hardware_name",
        "hardware_details",
        "execution_settings",
        "measured_ttft_ms",
        "measured_tpot_ms",
        "measured_throughput_tps",
        "source",
        "source_url",
        "benchmark_conditions",
        "verification_status",
        "notes",
    ]

    def __init__(self, records: list[BenchmarkRecord] | None = None) -> None:
        self.records: list[BenchmarkRecord] = records or []

    @classmethod
    def load_csv(cls, csv_path: str) -> BenchmarkManifest:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Benchmark manifest not found: {csv_path}")

        records = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = [h for h in cls.REQUIRED_HEADERS if h not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"Benchmark manifest missing required headers: {missing}")

            for row in reader:
                rec = BenchmarkRecord(
                    benchmark_id=row["benchmark_id"].strip(),
                    model_name=row["model_name"].strip(),
                    format_or_quant=row["format_or_quant"].strip(),
                    context_len=int(row["context_len"]),
                    gen_len=int(row["gen_len"]),
                    batch_size=int(row["batch_size"]),
                    software_version=row["software_version"].strip(),
                    hardware_name=row["hardware_name"].strip(),
                    hardware_details=row.get("hardware_details", "").strip(),
                    execution_settings=row.get("execution_settings", "").strip(),
                    measured_ttft_ms=float(row["measured_ttft_ms"]) if row["measured_ttft_ms"] else None,
                    measured_tpot_ms=float(row["measured_tpot_ms"]) if row["measured_tpot_ms"] else None,
                    measured_throughput_tps=float(row["measured_throughput_tps"]) if row["measured_throughput_tps"] else None,
                    source=row["source"].strip(),
                    source_url=row.get("source_url", "").strip(),
                    benchmark_conditions=row["benchmark_conditions"].strip(),
                    verification_status=row.get("verification_status", "unverified").strip(),
                    notes=row.get("notes", "").strip(),
                )
                records.append(rec)

        return cls(records)

    def filter_by_hardware(self, hardware_name: str) -> list[BenchmarkRecord]:
        return [r for r in self.records if hardware_name.lower() in r.hardware_name.lower()]

    def get_verified_records(self) -> list[BenchmarkRecord]:
        return [r for r in self.records if r.is_verified]

    def get_unverified_records(self) -> list[BenchmarkRecord]:
        return [r for r in self.records if not r.is_verified]

    def get_provisional_records(self) -> list[BenchmarkRecord]:
        """Backward-compatible alias for get_unverified_records."""
        return self.get_unverified_records()


class CalibrationModel:
    """Computes alignment metrics between analytical simulation ceilings and empirical data."""

    def __init__(self) -> None:
        self.comparison_pairs: list[dict[str, Any]] = []

    def add_comparison(
        self,
        benchmark_id: str,
        metric: str,
        measured: float,
        simulated: float,
        verification_status: str = "unverified",
    ) -> None:
        for name, val in [("measured", measured), ("simulated", simulated)]:
            if not math.isfinite(val):
                raise ValueError(f"{name} must be a finite number, got {val}")

        if measured <= 0:
            raise ValueError("measured value must be positive")
        if simulated <= 0:
            raise ValueError("simulated value must be positive")

        if verification_status != "verified":
            warnings.warn(
                f"Comparison for {benchmark_id} is against a {verification_status} benchmark. "
                "Do not claim predictive accuracy until verified.",
                UserWarning,
                stacklevel=2,
            )

        self.comparison_pairs.append({
            "benchmark_id": benchmark_id,
            "metric": metric,
            "measured": measured,
            "simulated": simulated,
            "verification_status": verification_status,
            "abs_percentage_error": abs(simulated - measured) / measured * 100.0,
        })

    def compute_mape(self, metric: str | None = None, verified_only: bool = False) -> float:
        """Computes Mean Absolute Percentage Error (MAPE) across recorded comparisons.

        Raises:
            ValueError: If no comparison records match the filter criteria.
        """
        pairs = self.comparison_pairs
        if verified_only:
            pairs = [p for p in pairs if p["verification_status"] == "verified"]
        if metric:
            pairs = [p for p in pairs if p["metric"] == metric]

        if not pairs:
            scope = f"metric='{metric}'" if metric else "all metrics"
            if verified_only:
                scope += " (verified only)"
            raise ValueError(
                f"No comparison pairs available to compute MAPE for {scope}. "
                "Empty dataset cannot yield 0% error."
            )

        return sum(p["abs_percentage_error"] for p in pairs) / len(pairs)


"""Calibration infrastructure and benchmark manifest management."""

import csv
import os
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


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
    measured_ttft_ms: Optional[float]
    measured_tpot_ms: Optional[float]
    measured_throughput_tps: Optional[float]
    source: str
    benchmark_conditions: str
    verification_status: str = "provisional"  # "verified" or "provisional"
    notes: str = ""

    @property
    def is_verified(self) -> bool:
        return self.verification_status.strip().lower() == "verified"


class BenchmarkManifest:
    """Loads, validates, and manages empirical benchmark records without fabrication."""

    REQUIRED_HEADERS = [
        "benchmark_id",
        "model_name",
        "format_or_quant",
        "context_len",
        "gen_len",
        "batch_size",
        "software_version",
        "hardware_name",
        "measured_ttft_ms",
        "measured_tpot_ms",
        "measured_throughput_tps",
        "source",
        "benchmark_conditions",
        "verification_status",
        "notes",
    ]

    def __init__(self, records: Optional[List[BenchmarkRecord]] = None):
        self.records: List[BenchmarkRecord] = records or []

    @classmethod
    def load_csv(cls, csv_path: str) -> "BenchmarkManifest":
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
                    measured_ttft_ms=float(row["measured_ttft_ms"]) if row["measured_ttft_ms"] else None,
                    measured_tpot_ms=float(row["measured_tpot_ms"]) if row["measured_tpot_ms"] else None,
                    measured_throughput_tps=float(row["measured_throughput_tps"]) if row["measured_throughput_tps"] else None,
                    source=row["source"].strip(),
                    benchmark_conditions=row["benchmark_conditions"].strip(),
                    verification_status=row.get("verification_status", "provisional").strip(),
                    notes=row.get("notes", "").strip(),
                )
                records.append(rec)

        return cls(records)

    def filter_by_hardware(self, hardware_name: str) -> List[BenchmarkRecord]:
        return [r for r in self.records if hardware_name.lower() in r.hardware_name.lower()]

    def get_verified_records(self) -> List[BenchmarkRecord]:
        return [r for r in self.records if r.is_verified]

    def get_provisional_records(self) -> List[BenchmarkRecord]:
        return [r for r in self.records if not r.is_verified]


class CalibrationModel:
    """Computes alignment metrics between analytical simulation ceilings and empirical data."""

    def __init__(self):
        self.comparison_pairs: List[Dict[str, Any]] = []

    def add_comparison(
        self,
        benchmark_id: str,
        metric: str,
        measured: float,
        simulated: float,
        verification_status: str = "provisional",
    ):
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

    def compute_mape(self, metric: Optional[str] = None, verified_only: bool = False) -> float:
        pairs = self.comparison_pairs
        if verified_only:
            pairs = [p for p in pairs if p["verification_status"] == "verified"]
        if metric:
            pairs = [p for p in pairs if p["metric"] == metric]
        if not pairs:
            return 0.0
        return sum(p["abs_percentage_error"] for p in pairs) / len(pairs)

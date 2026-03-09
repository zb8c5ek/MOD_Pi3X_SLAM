"""
util_report - ESSN report generation and structured logging.

Each ESSN produces a structured JSON report summarizing what it did,
quality metrics, and status. Each MOD can use the logger for full
execution logs with timestamps.
"""

import datetime
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional


# ======================================================================
# Structured ESSN Report
# ======================================================================

class ESSNReport:
    """Structured report produced by an ESSN at the end of its execution.

    Usage::

        report = ESSNReport("essn_submap")
        report.set_metric("inference_time_s", 2.34)
        report.set_metric("num_points", 150000)
        report.add_warning("Low overlap ratio: 0.12")
        report.set_status("success")
        report.save("/path/to/output/")
    """

    def __init__(self, essn_name: str):
        self.essn_name = essn_name
        self.timestamp = datetime.datetime.now().isoformat()
        self.status = "in_progress"
        self.metrics: Dict[str, Any] = {}
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self._sections: Dict[str, Any] = {}

    def set_status(self, status: str):
        """Set final status: 'success', 'partial', or 'error'."""
        self.status = status

    def set_metric(self, key: str, value: Any):
        self.metrics[key] = value

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.status = "error"

    def add_section(self, name: str, data: Any):
        """Add a named sub-section (e.g., per-submap details)."""
        self._sections[name] = data

    def to_dict(self) -> dict:
        d = {
            "essn": self.essn_name,
            "timestamp": self.timestamp,
            "status": self.status,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "errors": self.errors,
        }
        if self._sections:
            d["sections"] = self._sections
        return d

    def save(self, output_dir: str, filename: str = None):
        """Write report JSON to output_dir."""
        if not output_dir:
            return
        os.makedirs(output_dir, exist_ok=True)
        fname = filename or f"report_{self.essn_name}.json"
        path = os.path.join(output_dir, fname)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    def print_summary(self):
        """Print a concise summary to stdout."""
        status_color = {"success": "\033[92m", "error": "\033[91m",
                        "partial": "\033[93m", "in_progress": "\033[94m"}
        reset = "\033[0m"
        color = status_color.get(self.status, "")
        print(f"\n--- Report: {self.essn_name} [{color}{self.status}{reset}] ---")
        for k, v in self.metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        if self.warnings:
            print(f"  warnings: {len(self.warnings)}")
        if self.errors:
            print(f"  errors: {len(self.errors)}")


# ======================================================================
# Structured MOD Logger
# ======================================================================

def setup_mod_logger(
    mod_name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a logger for a MOD with console + optional file output.

    Args:
        mod_name: Module name (e.g. "essn_slam", "kern_graph").
        log_dir: If set, also writes to <log_dir>/<mod_name>.log.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(mod_name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        f"[%(asctime)s] [{mod_name}] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(
            os.path.join(log_dir, f"{mod_name}.log"),
            mode='a',
            encoding='utf-8',
        )
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger

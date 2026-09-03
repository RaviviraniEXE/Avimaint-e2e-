"""Small dependency-free live progress helpers for long IE training jobs.

The project runs mainly from Windows CMD through ``conda run``.  This module keeps
progress visible without adding a new package to the classical IE environment.
It also writes machine-readable JSONL events when ``AVIMAINT_TRAIN_TRACE`` is set.

The progress percentage is deliberately based on *completed experimental units*
(e.g. hyper-parameter configurations), not on fabricated estimates of the
internal LBFGS solver iterations.  While a configuration is fitting, a spinner
and elapsed time remain live so the terminal never looks frozen.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    value = int(round(seconds))
    hours, rem = divmod(value, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _trace(kind: str, **payload: Any) -> None:
    path = os.environ.get("AVIMAINT_TRAIN_TRACE", "").strip()
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": kind,
        **payload,
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True, default=str) + "\n")
        handle.flush()


def trace_event(kind: str, **payload: Any) -> None:
    """Public hook for scripts that want to add non-progress events to the trace."""
    _trace(kind, **payload)


class LiveProgress:
    """Live progress for a sequence of expensive, blocking training units.

    A bar tracks completed units. During the current blocking fit a lightweight
    background spinner updates elapsed time. ETA is calculated only from fully
    completed units, so it is an honest empirical estimate rather than a solver
    iteration guess.
    """

    _SPIN = "|/-\\"

    def __init__(self, label: str, total: int, width: int = 24) -> None:
        self.label = str(label)
        self.total = max(int(total), 1)
        self.width = max(int(width), 10)
        self.completed = 0
        self.started = time.perf_counter()
        self.current_started: float | None = None
        self.current_detail = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())
        _trace("progress_start", label=self.label, total=self.total)

    def _eta(self) -> float | None:
        if self.completed <= 0:
            return None
        elapsed = time.perf_counter() - self.started
        return (elapsed / self.completed) * max(self.total - self.completed, 0)

    def _line(self, spinner: str = " ", suffix: str = "") -> str:
        done = min(self.completed, self.total)
        fraction = done / self.total
        filled = int(round(self.width * fraction))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self.started
        detail = f" | {self.current_detail}" if self.current_detail else ""
        tail = f" | {suffix}" if suffix else ""
        return (
            f"[{bar}] {done:>3}/{self.total:<3} {fraction * 100:6.1f}% "
            f"| {self.label}{detail} | {spinner} elapsed {_fmt_seconds(elapsed)} "
            f"| ETA {_fmt_seconds(self._eta())}{tail}"
        )

    def _spin_loop(self) -> None:
        index = 0
        while not self._stop.wait(0.25):
            text = self._line(self._SPIN[index % len(self._SPIN)], "fitting")
            # Clear the rest of a possibly longer previous line.
            sys.stdout.write("\r" + text + " " * 8)
            sys.stdout.flush()
            index += 1

    def begin(self, detail: str) -> None:
        self.current_detail = str(detail)
        self.current_started = time.perf_counter()
        _trace(
            "unit_start",
            label=self.label,
            completed=self.completed,
            total=self.total,
            detail=self.current_detail,
        )
        if self._interactive:
            self._stop.clear()
            self._thread = threading.Thread(target=self._spin_loop, daemon=True)
            self._thread.start()
        else:
            print(self._line(">", "START"), flush=True)

    def finish(self, metric: str = "", is_best: bool = False, extra: str = "") -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
            self._thread = None
        unit_seconds = (
            time.perf_counter() - self.current_started if self.current_started is not None else None
        )
        self.completed = min(self.completed + 1, self.total)
        suffix_parts = []
        if metric:
            suffix_parts.append(metric)
        if is_best:
            suffix_parts.append("BEST")
        if extra:
            suffix_parts.append(extra)
        suffix = " | ".join(suffix_parts)
        text = self._line("OK", suffix)
        if self._interactive:
            sys.stdout.write("\r" + text + " " * 8 + "\n")
            sys.stdout.flush()
        else:
            print(text, flush=True)
        _trace(
            "unit_complete",
            label=self.label,
            completed=self.completed,
            total=self.total,
            detail=self.current_detail,
            metric=metric,
            is_best=bool(is_best),
            extra=extra,
            unit_seconds=round(unit_seconds, 3) if unit_seconds is not None else None,
            elapsed_seconds=round(time.perf_counter() - self.started, 3),
        )
        self.current_started = None

    def fail(self, error: BaseException) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._interactive:
            sys.stdout.write("\r")
        print(self._line("!!", f"FAILED: {type(error).__name__}: {error}"), flush=True)
        _trace(
            "unit_failed",
            label=self.label,
            completed=self.completed,
            total=self.total,
            detail=self.current_detail,
            error_type=type(error).__name__,
            error=str(error),
        )

    def close(self, summary: str = "") -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
            self._thread = None
        elapsed = time.perf_counter() - self.started
        msg = f"{self.label} complete: {self.completed}/{self.total} in {_fmt_seconds(elapsed)}"
        if summary:
            msg += f" | {summary}"
        print(msg, flush=True)
        _trace(
            "progress_complete",
            label=self.label,
            completed=self.completed,
            total=self.total,
            elapsed_seconds=round(elapsed, 3),
            summary=summary,
        )


class EpochProgress:
    """Compact live progress for epoch/batch based neural training.

    The monitor does not alter optimisation or early stopping.  It only reports
    what is already happening: current epoch/batch, running train loss, DEV loss,
    optional DEV F1, best DEV loss, patience counter, elapsed time and empirical
    ETA.  Machine-readable epoch events are written to the same JSONL trace used
    by the classical tuning progress.
    """

    def __init__(self, label: str, epochs: int, patience: int, width: int = 20) -> None:
        self.label = str(label)
        self.epochs = max(1, int(epochs))
        self.patience = max(1, int(patience))
        self.width = max(10, int(width))
        self.started = time.perf_counter()
        self.epoch_started: float | None = None
        self.completed_epochs = 0
        self._interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._last_noninteractive_bucket = -1
        _trace("neural_training_start", label=self.label, epochs=self.epochs, patience=self.patience)

    def _epoch_eta(self) -> float | None:
        if self.completed_epochs <= 0:
            return None
        elapsed = time.perf_counter() - self.started
        return (elapsed / self.completed_epochs) * max(self.epochs - self.completed_epochs, 0)

    def _batch_bar(self, done: int, total: int) -> str:
        total = max(1, int(total)); done = min(max(0, int(done)), total)
        frac = done / total
        filled = int(round(self.width * frac))
        return "#" * filled + "-" * (self.width - filled)

    def start_epoch(self, epoch: int, total_batches: int) -> None:
        self.epoch_started = time.perf_counter()
        self._last_noninteractive_bucket = -1
        _trace(
            "neural_epoch_start",
            label=self.label,
            epoch=int(epoch),
            max_epochs=self.epochs,
            total_batches=int(total_batches),
        )
        print(
            f"\n[{self.label}] epoch {epoch:02d}/{self.epochs:02d} START | "
            f"batches={total_batches} | overall elapsed {_fmt_seconds(time.perf_counter()-self.started)}",
            flush=True,
        )

    def batch(self, epoch: int, done: int, total: int, train_loss: float) -> None:
        total = max(1, int(total)); done = min(max(0, int(done)), total)
        frac = done / total
        bar = self._batch_bar(done, total)
        epoch_elapsed = time.perf_counter() - (self.epoch_started or time.perf_counter())
        if done > 0:
            batch_eta = (epoch_elapsed / done) * max(total - done, 0)
        else:
            batch_eta = None
        line = (
            f"[{self.label}] epoch {epoch:02d}/{self.epochs:02d} "
            f"[{bar}] {done:>3}/{total:<3} {frac*100:5.1f}% | "
            f"train_loss={train_loss:.4f} | epoch ETA {_fmt_seconds(batch_eta)}"
        )
        if self._interactive:
            sys.stdout.write("\r" + line + " " * 8)
            sys.stdout.flush()
        else:
            # Keep redirected logs readable: emit only at 25/50/75/100% milestones.
            bucket = min(4, int(frac * 4 + 1e-9))
            if bucket > self._last_noninteractive_bucket or done == total:
                print(line, flush=True)
                self._last_noninteractive_bucket = bucket

    def finish_epoch(
        self,
        epoch: int,
        train_loss: float,
        dev_loss: float,
        dev_f1: float | None,
        best_dev_loss: float,
        stale: int,
        improved: bool,
        best_epoch: int,
    ) -> None:
        if self._interactive:
            sys.stdout.write("\r")
        self.completed_epochs = int(epoch)
        elapsed = time.perf_counter() - self.started
        epoch_seconds = time.perf_counter() - (self.epoch_started or time.perf_counter())
        metric = f"DEV F1={dev_f1:.4f}" if dev_f1 is not None else "DEV F1=n/a"
        state = "BEST" if improved else "no improvement"
        print(
            f"[{self.label}] epoch {epoch:02d}/{self.epochs:02d} DONE | "
            f"train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f} | {metric} | "
            f"best_dev_loss={best_dev_loss:.4f} @ epoch {best_epoch} | "
            f"patience={stale}/{self.patience} | {state} | "
            f"epoch={_fmt_seconds(epoch_seconds)} | elapsed={_fmt_seconds(elapsed)} | "
            f"ETA={_fmt_seconds(self._epoch_eta())}",
            flush=True,
        )
        _trace(
            "neural_epoch_complete",
            label=self.label,
            epoch=int(epoch),
            max_epochs=self.epochs,
            train_loss=round(float(train_loss), 8),
            dev_loss=round(float(dev_loss), 8),
            dev_f1=None if dev_f1 is None else round(float(dev_f1), 8),
            best_dev_loss=round(float(best_dev_loss), 8),
            best_epoch=int(best_epoch),
            stale=int(stale),
            patience=self.patience,
            improved=bool(improved),
            epoch_seconds=round(epoch_seconds, 3),
            elapsed_seconds=round(elapsed, 3),
        )

    def early_stop(self, epoch: int, best_epoch: int, best_dev_loss: float) -> None:
        print(
            f"[{self.label}] EARLY STOPPING at epoch {epoch} | "
            f"best epoch={best_epoch} | best DEV loss={best_dev_loss:.4f} | "
            f"patience={self.patience}",
            flush=True,
        )
        _trace(
            "neural_early_stop",
            label=self.label,
            stopped_epoch=int(epoch),
            best_epoch=int(best_epoch),
            best_dev_loss=round(float(best_dev_loss), 8),
            patience=self.patience,
        )

    def close(self, stopped_epoch: int, best_epoch: int, best_dev_loss: float) -> None:
        elapsed = time.perf_counter() - self.started
        print(
            f"[{self.label}] TRAINING COMPLETE | stopped_epoch={stopped_epoch} | "
            f"best_epoch={best_epoch} | best_DEV_loss={best_dev_loss:.4f} | "
            f"elapsed={_fmt_seconds(elapsed)}",
            flush=True,
        )
        _trace(
            "neural_training_complete",
            label=self.label,
            stopped_epoch=int(stopped_epoch),
            best_epoch=int(best_epoch),
            best_dev_loss=round(float(best_dev_loss), 8),
            elapsed_seconds=round(elapsed, 3),
        )

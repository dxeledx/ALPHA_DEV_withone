from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


@dataclass
class _Worker:
    idx: int
    proc: subprocess.Popen[str]
    stdin: TextIO
    stdout: TextIO
    log_f: TextIO | None
    busy: bool = False
    task_id: int | None = None


class SubprocessSelfPlayPool:
    """
    A self-play pool implemented via subprocess + stdin/stdout JSONL.

    Motivation:
      - Thread backend is GIL-limited (often ~1 core busy).
      - multiprocessing/process pools may fail on some CPU servers where /dev/shm (POSIX semaphores)
        is not writable, causing PermissionError in multiprocessing.SemLock.
    """

    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        device: str,
        weights_path: str | Path,
        num_workers: int,
        log_dir: str | Path | None = None,
        python_exe: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self._cfg = cfg
        self._device = str(device)
        self._weights_path = str(weights_path)
        self._num_workers = max(0, int(num_workers))
        self._workers: list[_Worker] = []
        self._sel = selectors.DefaultSelector()

        if self._num_workers <= 0:
            raise ValueError("num_workers must be > 0 for SubprocessSelfPlayPool")

        log_dir_p = Path(log_dir) if log_dir is not None else None
        if log_dir_p is not None:
            log_dir_p.mkdir(parents=True, exist_ok=True)

        exe = python_exe or sys.executable
        base_env = os.environ.copy()
        # Default: keep each self-play worker single-threaded to avoid CPU over-subscription.
        # Users can override these via `env=...` when constructing the pool.
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            base_env[k] = "1"
        if env:
            base_env.update({str(k): str(v) for k, v in env.items()})
        base_env.setdefault("PYTHONUNBUFFERED", "1")

        for wi in range(self._num_workers):
            log_f = None
            stderr: int | TextIO = subprocess.DEVNULL
            if log_dir_p is not None:
                log_path = log_dir_p / f"worker_{wi:02d}.log"
                log_f = log_path.open("a", encoding="utf-8", buffering=1)
                stderr = log_f

            proc = subprocess.Popen(
                [exe, "-m", "eeg_channel_game.rl.subprocess_selfplay_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=True,
                bufsize=1,
                env=base_env,
            )
            if proc.stdin is None or proc.stdout is None:  # pragma: no cover
                raise RuntimeError("Failed to start subprocess worker (missing pipes)")

            w = _Worker(idx=wi, proc=proc, stdin=proc.stdin, stdout=proc.stdout, log_f=log_f)
            self._workers.append(w)

        # init handshake
        for w in self._workers:
            self._send(w, {"cmd": "init", "cfg": self._cfg, "device": self._device, "weights_path": self._weights_path})
        for w in self._workers:
            rep = self._recv(w)
            if not rep.get("ok", False) or rep.get("cmd") != "init":
                raise RuntimeError(f"subprocess selfplay init failed for worker {w.idx}: {rep}")

        for w in self._workers:
            self._sel.register(w.stdout, selectors.EVENT_READ, w)

    def _send(self, w: _Worker, msg: dict[str, Any]) -> None:
        try:
            w.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
            w.stdin.flush()
        except BrokenPipeError as e:
            raise RuntimeError(f"worker {w.idx} stdin broken") from e

    def _recv(self, w: _Worker) -> dict[str, Any]:
        line = w.stdout.readline()
        if not line:
            raise RuntimeError(f"worker {w.idx} exited unexpectedly (EOF on stdout)")
        try:
            return json.loads(line)
        except Exception as e:
            raise RuntimeError(f"worker {w.idx} returned non-JSON output: {line[:200]!r}") from e

    def run_tasks(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not tasks:
            return []

        results: list[dict[str, Any] | None] = [None] * len(tasks)
        pending: list[tuple[int, dict[str, Any]]] = list(enumerate(tasks))

        # prime workers
        for w in self._workers:
            if not pending:
                break
            task_id, task = pending.pop(0)
            w.busy = True
            w.task_id = int(task_id)
            self._send(w, {"cmd": "run_one_game", "task": task})

        remaining = sum(1 for w in self._workers if w.busy)
        while remaining > 0:
            for key, _mask in self._sel.select():
                w: _Worker = key.data
                rep = self._recv(w)
                if not rep.get("ok", False) or rep.get("cmd") != "run_one_game":
                    raise RuntimeError(f"worker {w.idx} failed: {rep}")
                out = rep.get("out", None)
                if w.task_id is None:
                    raise RuntimeError(f"worker {w.idx} returned output but task_id is None")
                results[int(w.task_id)] = out
                w.busy = False
                w.task_id = None
                remaining -= 1

                if pending:
                    task_id, task = pending.pop(0)
                    w.busy = True
                    w.task_id = int(task_id)
                    self._send(w, {"cmd": "run_one_game", "task": task})
                    remaining += 1

        out_final: list[dict[str, Any]] = []
        for i, r in enumerate(results):
            if r is None:
                raise RuntimeError(f"Missing selfplay result for task {i}")
            out_final.append(r)
        return out_final

    def shutdown(self) -> None:
        for w in self._workers:
            try:
                self._send(w, {"cmd": "shutdown"})
            except Exception:
                pass
        for w in self._workers:
            try:
                w.proc.terminate()
            except Exception:
                pass
        for w in self._workers:
            try:
                w.proc.wait(timeout=5.0)
            except Exception:
                try:
                    w.proc.kill()
                except Exception:
                    pass
        for w in self._workers:
            try:
                self._sel.unregister(w.stdout)
            except Exception:
                pass
            try:
                if w.log_f is not None:
                    w.log_f.close()
            except Exception:
                pass

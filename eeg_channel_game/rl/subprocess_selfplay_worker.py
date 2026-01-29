from __future__ import annotations

import json
import sys
import traceback
from typing import Any

from eeg_channel_game.rl import parallel_selfplay as ps


def _send(msg: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    # NumPy arrays/scalars: prefer tolist()/item() without importing numpy here.
    tolist = getattr(x, "tolist", None)
    if callable(tolist):
        return _to_jsonable(tolist())
    item = getattr(x, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(x)


def main() -> int:
    initialized = False
    while True:
        line = sys.stdin.readline()
        if not line:
            return 0

        try:
            msg = json.loads(line)
        except Exception:
            _send({"ok": False, "error": "invalid_json"})
            continue

        cmd = msg.get("cmd", None)
        try:
            if cmd == "init":
                cfg = msg["cfg"]
                device = str(msg.get("device", "cpu"))
                weights_path = str(msg["weights_path"])
                ps.init_worker(cfg, device, weights_path)
                initialized = True
                _send({"ok": True, "cmd": "init"})
                continue

            if cmd == "shutdown":
                _send({"ok": True, "cmd": "shutdown"})
                return 0

            if cmd == "run_one_game":
                if not initialized:
                    raise RuntimeError("worker not initialized (missing init)")
                task = msg["task"]
                out = ps.run_one_game(task)
                _send({"ok": True, "cmd": "run_one_game", "out": _to_jsonable(out)})
                continue

            raise ValueError(f"Unknown cmd={cmd!r}")
        except Exception as e:
            _send(
                {
                    "ok": False,
                    "cmd": cmd,
                    "error": str(e),
                    "traceback": traceback.format_exc(limit=5),
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())

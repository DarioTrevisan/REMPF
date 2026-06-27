from __future__ import annotations
import json, os, platform, subprocess, sys, time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict
import numpy as np

def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

def make_meta() -> Dict[str, Any]:
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }

def save_npz(path: str, *, arrays: Dict[str, np.ndarray], params: Dict[str, Any], meta: Dict[str, Any] | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = dict(arrays)
    payload["params_json"] = np.array(json.dumps(params), dtype=object)
    payload["meta_json"] = np.array(json.dumps(meta or make_meta()), dtype=object)
    np.savez_compressed(path, **payload)

def load_npz(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {k: z[k] for k in z.files}
    out["params"] = json.loads(out.pop("params_json").item())
    out["meta"] = json.loads(out.pop("meta_json").item())
    return out

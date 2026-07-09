#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--time-json", required=True)
    ap.add_argument("--time-txt", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("No command provided after --")

    Path(args.time_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.time_txt).parent.mkdir(parents=True, exist_ok=True)

    print(f"[RUN_START] label={args.label}")
    print("[RUN_CMD]", " ".join(cmd), flush=True)

    start_wall = time.perf_counter()
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")

    proc = subprocess.Popen(cmd)
    exit_code = proc.wait()

    end_wall = time.perf_counter()
    end_time = time.strftime("%Y-%m-%d %H:%M:%S")

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)

    result = {
        "label": args.label,
        "cmd": cmd,
        "start_time": start_time,
        "end_time": end_time,
        "exit_code": exit_code,
        "wall_seconds": end_wall - start_wall,
        "wall_minutes": (end_wall - start_wall) / 60,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "max_rss_kb": usage.ru_maxrss,
    }

    with open(args.time_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(args.time_txt, "w", encoding="utf-8") as f:
        for k, v in result.items():
            f.write(f"{k}: {v}\n")

    print("[RUN_DONE]", json.dumps(result, ensure_ascii=False), flush=True)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()

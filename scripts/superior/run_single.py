"""Execute a single run by delegating to ``make run``.

This script is intentionally lightweight: it translates the run specification
into a Make invocation, redirects logs, and propagates the exit code. When a
RAM budget is provided, it monitors the child process (if ``psutil`` is
available) and terminates runs that exceed the limit.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


def _parse_key_vals(items: List[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid KEY=VALUE pair: {item}")
        k, v = item.split("=", 1)
        result[k] = v
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single experiment task via make")
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument(
        "--make-var",
        action="append",
        default=[],
        help="Repeatable KEY=VALUE make variable",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Repeatable logical override key=value (flattened at call site)",
    )
    parser.add_argument("--max-ram-mb", type=int, default=None, help="Max RAM (MB) budget – reserved for future")
    parser.add_argument("--log-path", required=True)
    return parser


def _build_command(args: argparse.Namespace) -> List[str]:
    make_vars = _parse_key_vals(args.make_var)
    overrides = args.override or []
    override_str = " ".join(overrides)

    cmd = [
        "make",
        "run",
        f"STAGE={args.stage}",
        f"PROFILE={args.profile}",
    ]
    for k, v in make_vars.items():
        cmd.append(f"{k}={v}")
    if override_str:
        cmd.append(f"OVERRIDES={override_str}")
    return cmd


def run_single(args: argparse.Namespace) -> int:
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = _build_command(args)

    if not args.max_ram_mb or psutil is None:
        if args.max_ram_mb and psutil is None:
            print(
                "[run_single] psutil not installed – RAM monitoring disabled despite max_ram_mb flag",
                file=sys.stderr,
            )
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(cmd, stdout=log_file, stderr=log_file)
        return process.returncode

    # psutil-based monitoring
    limit_bytes = int(args.max_ram_mb) * 1024 * 1024
    peak_rss = 0
    return_code: int | None = None
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
        ps_proc = psutil.Process(proc.pid)

        while True:
            ret = proc.poll()
            if ret is not None:
                return_code = ret
                break

            try:
                rss = ps_proc.memory_info().rss
            except psutil.NoSuchProcess:
                return_code = proc.poll()
                break

            peak_rss = max(peak_rss, rss)
            if rss > limit_bytes:
                print(
                    "[run_single] RAM limit exceeded, killing run", file=sys.stderr
                )
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return_code = 99
                break

            time.sleep(1.0)

        if return_code is None:
            return_code = proc.wait()

    peak_mb = peak_rss / (1024 * 1024)
    print(
        f"[run_single] return_code={return_code} peak_rss_mb={peak_mb:.2f}",
        file=sys.stderr,
    )
    return return_code


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    exit_code = run_single(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

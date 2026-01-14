#!/usr/bin/env python3
"""
Find duplicated Weights & Biases runs by exact run name.

Examples:
  # Find duplicate eval runs (finished only)
  python -m analysis.find_duplicate_wandb_runs --mode eval

  # Find duplicates across all states
  python -m analysis.find_duplicate_wandb_runs --mode eval --state

  # Find duplicates in a different entity/project
  python -m analysis.find_duplicate_wandb_runs --entity my-team --project my-proj --mode all
"""

import argparse
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

import wandb


def _to_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None
    return None


def _run_created_at(run: wandb.apis.public.Run) -> datetime:
    dt = _to_datetime(getattr(run, "created_at", None))
    if dt is not None:
        return dt
    attrs = getattr(run, "_attrs", None)
    if isinstance(attrs, dict):
        dt = _to_datetime(attrs.get("createdAt") or attrs.get("created_at"))
        if dt is not None:
            return dt
    return datetime.min


def main() -> None:
    parser = argparse.ArgumentParser(description="Find duplicated W&B run names")
    parser.add_argument("--entity", default="nathanielmitrani-cfis-upc")
    parser.add_argument("--project", default="obfuscation_generalization")
    parser.add_argument(
        "--mode",
        choices=["eval", "train", "all"],
        default="eval",
        help="Which runs to consider (based on 'eval' substring in run name).",
    )
    parser.add_argument(
        "--state",
        nargs="*",
        default=["finished"],
        help="Filter by run state(s). Pass --state with no values to include all states.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=50,
        help="Max duplicate names to print (0 for no limit).",
    )
    args = parser.parse_args()

    filters = None
    if args.state is not None and len(args.state) > 0:
        filters = {"state": {"$in": args.state}}

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}", filters=filters)

    def _mode_ok(run_name: str) -> bool:
        is_eval = "eval" in run_name.lower()
        if args.mode == "eval":
            return is_eval
        if args.mode == "train":
            return not is_eval
        return True

    by_name = defaultdict(list)
    total_considered = 0
    for run in runs:
        name = getattr(run, "name", "") or ""
        if not _mode_ok(name):
            continue
        by_name[name].append(run)
        total_considered += 1

    duplicates = [(name, rs) for name, rs in by_name.items() if len(rs) > 1]
    duplicates.sort(key=lambda t: len(t[1]), reverse=True)

    print(f"Entity/project: {args.entity}/{args.project}")
    print(f"Mode: {args.mode}")
    print(f"Total runs considered: {total_considered}")
    print(f"Duplicate run names: {len(duplicates)}")

    if args.show == 0:
        return

    shown = 0
    for name, rs in duplicates:
        rs_sorted = sorted(rs, key=_run_created_at)
        kept = rs_sorted[-1]
        print("")
        print(f"- {name}  (count={len(rs_sorted)}; latest id={kept.id})")
        for r in rs_sorted:
            print(
                f"    id={r.id} state={r.state} created_at={getattr(r, 'created_at', None)}"
            )
        shown += 1
        if args.show > 0 and shown >= args.show:
            break


if __name__ == "__main__":
    main()

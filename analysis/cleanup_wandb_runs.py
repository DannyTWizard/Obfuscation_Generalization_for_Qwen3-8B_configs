"""
Script to find and delete W&B runs that have no meaningful logs (only system metrics).

These are typically failed/crashed runs that started but never logged any training or eval metrics.

Safety features:
- DRY RUN by default (--delete flag required to actually delete)
- Interactive confirmation before deletion
- Project filtering
- Detailed logging

Usage:
    # List runs that would be deleted (dry run)
    python cleanup_wandb_runs.py

    # Actually delete the runs (with confirmation)
    python cleanup_wandb_runs.py --delete

    # Skip confirmation prompt
    python cleanup_wandb_runs.py --delete --yes

    # Filter by run state
    python cleanup_wandb_runs.py --state crashed failed
"""

import argparse
from typing import List, Optional, Set

import wandb

# Default wandb configuration (matches your project)
DEFAULT_ENTITY = "nathanielmitrani-cfis-upc"
DEFAULT_PROJECT = "obfuscation_generalization"

# System/internal metrics that W&B logs automatically
# Runs with ONLY these keys are considered "empty"
SYSTEM_METRIC_PREFIXES = (
    "_",  # Internal keys like _runtime, _timestamp, _step, _wandb
    "system.",  # System metrics like system.cpu, system.gpu
    "system/",  # Alternative system metric format
)

# These specific keys are also considered system/internal
SYSTEM_METRIC_EXACT = {
    "runtime",
    "timestamp",
    "step",
}


def is_system_metric(key: str) -> bool:
    """Check if a metric key is a system/internal metric."""
    key_lower = key.lower()

    # Check prefixes
    for prefix in SYSTEM_METRIC_PREFIXES:
        if key_lower.startswith(prefix):
            return True

    # Check exact matches
    if key_lower in SYSTEM_METRIC_EXACT:
        return True

    return False


def get_meaningful_metrics(run: wandb.apis.public.Run) -> Set[str]:
    """
    Get the set of meaningful (non-system) metrics logged by a run.

    Args:
        run: The W&B run object

    Returns:
        Set of metric names that are not system metrics
    """
    # Get all keys from summary (contains all metrics ever logged)
    all_keys = set(run.summary.keys())

    # Filter out system metrics
    meaningful = {key for key in all_keys if not is_system_metric(key)}

    return meaningful


def find_empty_runs(
    entity: str,
    project: str,
    state_filter: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[wandb.apis.public.Run]:
    """
    Find all runs that have no meaningful metrics logged.

    Args:
        entity: W&B entity (username or team)
        project: W&B project name
        state_filter: Optional list of run states to filter by
        verbose: Whether to print progress information

    Returns:
        List of runs with only system metrics
    """
    api = wandb.Api()

    if verbose:
        print(f"Fetching runs from {entity}/{project}...")

    filters = {}
    if state_filter:
        filters["state"] = {"$in": state_filter}

    runs = api.runs(f"{entity}/{project}", filters=filters if filters else None)

    if verbose:
        print(f"Found {len(runs)} total runs")
        if state_filter:
            print(f"Filtering by states: {state_filter}")
        print("\nAnalyzing runs for meaningful metrics...\n")

    empty_runs = []
    non_empty_count = 0

    for i, run in enumerate(runs):
        meaningful_metrics = get_meaningful_metrics(run)

        if not meaningful_metrics:
            empty_runs.append(run)
            if verbose:
                print(f"  ✗ EMPTY: {run.name}")
                print(
                    f"           ID: {run.id}, State: {run.state}, Created: {run.created_at}"
                )
        else:
            non_empty_count += 1
            if verbose and non_empty_count <= 5:
                # Show first few non-empty runs as examples
                print(f"  ✓ OK: {run.name} ({len(meaningful_metrics)} metrics)")

        # Progress indicator for large projects
        if verbose and (i + 1) % 50 == 0:
            print(f"\n  ... processed {i + 1} runs ...\n")

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"Summary:")
        print(f"  Total runs analyzed: {len(runs)}")
        print(f"  Runs with metrics:   {non_empty_count}")
        print(f"  Empty runs:          {len(empty_runs)}")
        print(f"{'=' * 70}\n")

    return empty_runs


def delete_runs(
    runs: List[wandb.apis.public.Run],
    verbose: bool = True,
) -> int:
    """
    Delete the given runs.

    Args:
        runs: List of runs to delete
        verbose: Whether to print progress

    Returns:
        Number of runs successfully deleted
    """
    deleted = 0

    for run in runs:
        try:
            if verbose:
                print(f"  Deleting: {run.name} (id: {run.id})...")
            run.delete()
            deleted += 1
            if verbose:
                print(f"    ✓ Deleted")
        except Exception as e:
            if verbose:
                print(f"    ✗ Failed: {e}")

    return deleted


def main():
    parser = argparse.ArgumentParser(
        description="Find and delete W&B runs with no meaningful metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run - list what would be deleted
    python cleanup_wandb_runs.py
    
    # Actually delete (with confirmation)
    python cleanup_wandb_runs.py --delete
    
    # Delete without confirmation (use with caution!)
    python cleanup_wandb_runs.py --delete --yes
    
    # Only check crashed/failed runs
    python cleanup_wandb_runs.py --state crashed failed
    
    # Check all run states
    python cleanup_wandb_runs.py --state
        """,
    )

    parser.add_argument(
        "--entity",
        "-e",
        default=DEFAULT_ENTITY,
        help=f"W&B entity (default: {DEFAULT_ENTITY})",
    )
    parser.add_argument(
        "--project",
        "-p",
        default=DEFAULT_PROJECT,
        help=f"W&B project name (default: {DEFAULT_PROJECT})",
    )
    parser.add_argument(
        "--state",
        nargs="*",
        default=None,
        help="Filter runs by state (e.g., crashed, failed, finished). "
        "Use --state without args to include all states.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete the runs (default is dry run)",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt when deleting",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress detailed output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Handle --state without arguments (empty list means all states)
    state_filter = args.state if args.state else None

    if verbose:
        print("\n" + "=" * 70)
        print("W&B EMPTY RUNS CLEANUP")
        print("=" * 70)
        print(f"Entity:     {args.entity}")
        print(f"Project:    {args.project}")
        print(f"States:     {state_filter if state_filter else 'all'}")
        print(f"Mode:       {'DELETE' if args.delete else 'DRY RUN'}")
        print("=" * 70 + "\n")

    # Find empty runs
    empty_runs = find_empty_runs(
        entity=args.entity,
        project=args.project,
        state_filter=state_filter,
        verbose=verbose,
    )

    if not empty_runs:
        print("No empty runs found. Nothing to delete.")
        return 0

    # If dry run, just show summary
    if not args.delete:
        print("=" * 70)
        print("DRY RUN - No runs were deleted.")
        print("To delete these runs, add the --delete flag:")
        print(f"    python cleanup_wandb_runs.py --delete")
        print("=" * 70)
        return 0

    # Confirmation before deletion
    if not args.yes:
        print(f"\n⚠️  WARNING: You are about to DELETE {len(empty_runs)} runs.")
        print("This action is IRREVERSIBLE.\n")

        response = input("Are you sure you want to continue? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("\nAborted. No runs were deleted.")
            return 0

    # Delete runs
    print(f"\nDeleting {len(empty_runs)} runs...\n")
    deleted = delete_runs(empty_runs, verbose=verbose)

    print(f"\n{'=' * 70}")
    print(f"✓ Successfully deleted {deleted}/{len(empty_runs)} runs")
    print(f"{'=' * 70}\n")

    return 0


if __name__ == "__main__":
    exit(main())

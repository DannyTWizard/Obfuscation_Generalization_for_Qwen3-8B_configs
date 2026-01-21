"""
Script to download per-trial eval data from W&B tables and compute detailed metrics.

Computes for "answer extractable" trials (where extracted_answer != "No answer tags found"):
- proportion that reward hacked (got correct answer / reward = 2)
- proportion that were flagged by monitor (overseer returned 1)
- proportion of each answer choice (A/B/C/D)

Supports incremental downloads - if output CSV already exists, only downloads
runs that don't already have entries in the CSV.

Usage:
    # Download all default datasets (leave_out_*_refined2)
    python download_wandb_trial_data.py --output trial_metrics.csv

    # Download with specific attributes
    python download_wandb_trial_data.py \
        --attributes ts=50 pen=-0.05 \
        --output trial_metrics.csv
    
    # Force re-download all (ignore existing CSV)
    python download_wandb_trial_data.py --output trial_metrics.csv --force
"""

import argparse
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import wandb
from requests.exceptions import HTTPError


# Rate limiting configuration
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 5
API_TIMEOUT = 60  # Increased timeout for large queries
REQUEST_DELAY = 0.1  # Delay between API calls to avoid rate limiting


def retry_on_rate_limit(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """Execute a function with retry on rate limit errors (429)."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                if attempt < max_retries - 1:
                    backoff = INITIAL_BACKOFF_SECONDS * (2**attempt)
                    print(
                        f"  Rate limited (429). Waiting {backoff}s before retry {attempt + 2}/{max_retries}..."
                    )
                    time.sleep(backoff)
                else:
                    print(
                        f"  Rate limited (429). Max retries ({max_retries}) exceeded."
                    )
                    raise
            else:
                raise
    return None  # Should not reach here


# Default wandb configuration
DEFAULT_ENTITY = "nathanielmitrani-cfis-upc"
DEFAULT_PROJECT = "obfuscation_generalization"

# Default datasets to download when no data attribute is specified
DEFAULT_DATASETS = [
    "leave_out_sycophancy_refined2",
    "leave_out_war_refined2",
    "leave_out_code_refined2",
    "leave_out_score_refined2",
]

# Expected checkpoint steps to look for
EXPECTED_STEPS = [25] + list(range(200, 3801, 200))  # 25, 200, 400, ..., 3800


def parse_attributes(attr_list: List[str]) -> Dict[str, str]:
    """Parse attribute strings in the format 'key=value' into a dictionary."""
    attributes = {}
    for attr in attr_list:
        if "=" not in attr:
            raise ValueError(
                f"Invalid attribute format: '{attr}'. Expected 'key=value'"
            )
        key, value = attr.split("=", 1)
        attributes[key.strip()] = value.strip()
    return attributes


def extract_attribute_from_run_name(run_name: str, key: str) -> Optional[str]:
    """Extract the value for a given key from a run name."""
    known_keys = ["pen", "lr", "epochs", "batch", "lora_r", "ovs", "data", "ts", "tg"]
    next_key_pattern = "|".join(re.escape(k) for k in known_keys if k != key)
    pattern = rf"(?:^|_){re.escape(key)}_(.+?)(?:_(?:{next_key_pattern})_|$)"
    match = re.search(pattern, run_name)
    if match:
        return match.group(1)
    return None


def run_name_matches_attributes(run_name: str, attributes: Dict[str, str]) -> bool:
    """Check if a run name contains all required "{key}_{value}" patterns."""
    for key, value in attributes.items():
        pattern = f"{re.escape(key)}_{re.escape(value)}"
        if not re.search(pattern, run_name):
            return False
    return True


def _to_datetime(value: Any) -> Optional[datetime]:
    """Best-effort conversion of various W&B timestamp formats to datetime."""
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
    """Return a comparable datetime for a W&B run."""
    dt = _to_datetime(getattr(run, "created_at", None))
    if dt is not None:
        return dt
    attrs = getattr(run, "_attrs", None)
    if isinstance(attrs, dict):
        for key in ("createdAt", "created_at"):
            dt = _to_datetime(attrs.get(key))
            if dt is not None:
                return dt
    return datetime.min


def dedupe_runs_by_name_keep_latest(
    runs: List[wandb.apis.public.Run], verbose: bool = True
) -> List[wandb.apis.public.Run]:
    """Deduplicate runs by exact run.name, keeping only the latest."""
    best_by_name: Dict[str, wandb.apis.public.Run] = {}
    dup_names: Dict[str, List[wandb.apis.public.Run]] = {}

    for run in runs:
        name = getattr(run, "name", None) or ""
        if name not in best_by_name:
            best_by_name[name] = run
            continue

        current_best = best_by_name[name]
        if _run_created_at(run) >= _run_created_at(current_best):
            best_by_name[name] = run

        dup_names.setdefault(name, []).append(run)

    if verbose and dup_names:
        print("\nDetected duplicate run names (keeping latest by created_at):")
        for name in sorted(dup_names.keys()):
            kept = best_by_name[name]
            print(f"  - {name} (keeping id={kept.id})")

    return list(best_by_name.values())


def get_config_value(config: Dict, key_path: str, default: Any = None) -> Any:
    """Get a nested config value using dot notation."""
    keys = key_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def load_existing_keys(csv_path: str, verbose: bool = True) -> set:
    """
    Load existing CSV and return a set of (data, seed, eval_fold, step) tuples.

    Returns empty set if file doesn't exist.
    """
    path = Path(csv_path)
    if not path.exists():
        if verbose:
            print(f"No existing CSV found at {csv_path}, will download all runs.")
        return set()

    try:
        df = pd.read_csv(path)
        # Create set of existing keys
        existing_keys = set()
        for _, row in df.iterrows():
            key = (
                str(row.get("data", "")),
                str(row.get("seed", "")),
                str(row.get("eval_fold", "")),
                str(row.get("step", "")),
            )
            existing_keys.add(key)

        if verbose:
            print(f"Loaded {len(existing_keys)} existing entries from {csv_path}")

        return existing_keys
    except Exception as e:
        if verbose:
            print(f"Error loading existing CSV: {e}. Will download all runs.")
        return set()


def get_run_key(run: wandb.apis.public.Run) -> tuple:
    """
    Extract the (data, seed, eval_fold, step) key from a run without downloading table data.
    """
    artifact_step = get_config_value(run.config, "artifact_step", None)
    training_run_name = get_config_value(run.config, "training_run_name", None)
    eval_fold = get_config_value(run.config, "eval.fold", None)

    ts = None
    data = None
    if training_run_name:
        ts = extract_attribute_from_run_name(training_run_name, "ts")
        data = extract_attribute_from_run_name(training_run_name, "data")

    return (
        str(data or ""),
        str(ts or ""),
        str(eval_fold or ""),
        str(artifact_step or ""),
    )


def find_matching_eval_runs(
    entity: str,
    project: str,
    attributes: Dict[str, str],
    state_filter: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[wandb.apis.public.Run]:
    """Find all W&B eval runs whose names match the given attribute patterns."""
    api = wandb.Api(timeout=API_TIMEOUT)

    if verbose:
        print(f"Fetching runs from {entity}/{project}...")

    filters = {}
    if state_filter:
        filters["state"] = {"$in": state_filter}

    runs = api.runs(f"{entity}/{project}", filters=filters if filters else None)

    if verbose:
        print(f"Found {len(runs)} total runs")
        print(f"Filtering for eval runs with attributes: {attributes}")

    matching_runs = []
    runs_iter = iter(runs)

    while True:
        # Retry loop for rate limiting during pagination
        run = None
        for attempt in range(MAX_RETRIES):
            try:
                run = next(runs_iter)
                break  # Success, exit retry loop
            except StopIteration:
                run = None
                break  # No more runs, exit retry loop
            except HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    if attempt < MAX_RETRIES - 1:
                        backoff = INITIAL_BACKOFF_SECONDS * (2**attempt)
                        print(
                            f"  Rate limited (429). Waiting {backoff}s before retry {attempt + 2}/{MAX_RETRIES}..."
                        )
                        time.sleep(backoff)
                    else:
                        print(
                            f"  Rate limited (429). Max retries ({MAX_RETRIES}) exceeded."
                        )
                        raise
                else:
                    raise

        if run is None:
            break  # No more runs

        # Skip "summary" runs
        if "summary" in run.name.lower():
            continue

        # Only match eval runs
        if "eval" not in run.name.lower():
            continue

        if run_name_matches_attributes(run.name, attributes):
            matching_runs.append(run)
            if verbose:
                print(f"  ✓ Match: {run.name} (id: {run.id}, state: {run.state})")

    if verbose:
        print(f"\nFound {len(matching_runs)} matching eval runs")

    return matching_runs


def get_table_from_run(
    run: wandb.apis.public.Run, table_key: str, verbose: bool = False
) -> Optional[pd.DataFrame]:
    """
    Retrieve a logged wandb.Table from a run.

    The table is logged with a key like "{fold}_samples" and saved as a JSON file.
    """
    import json
    import tempfile

    # Method 1: Try getting from run files (most reliable for logged tables)
    try:
        files = retry_on_rate_limit(lambda: list(run.files()))
        if files is None:
            files = []
        for file in files:
            # Tables are saved in media/table/ directory
            if table_key in file.name and ".table.json" in file.name:
                if verbose:
                    print(f"    Found table file: {file.name}")
                # Download to temp directory
                with tempfile.TemporaryDirectory() as tmpdir:
                    downloaded = file.download(root=tmpdir, replace=True)
                    with open(downloaded.name, "r") as f:
                        data = json.load(f)
                    columns = data.get("columns", [])
                    rows = data.get("data", [])
                    if columns and rows:
                        return pd.DataFrame(rows, columns=columns)
    except Exception as e:
        if verbose:
            print(f"    Error accessing run files: {e}")

    # Method 2: Try getting from summary (for reference tables)
    try:
        if table_key in run.summary:
            table_ref = run.summary[table_key]
            # If it's a table reference, try to load it
            if hasattr(table_ref, "_table_key"):
                # It's a reference to a table artifact
                if verbose:
                    print(f"    Found table reference in summary: {table_key}")
    except Exception as e:
        if verbose:
            print(f"    Error checking summary: {e}")

    # Method 3: Try checking logged artifacts
    try:
        for artifact in run.logged_artifacts():
            if verbose:
                print(f"    Checking artifact: {artifact.name} (type: {artifact.type})")
            if "table" in artifact.type.lower() or table_key in artifact.name:
                artifact_dir = artifact.download()
                # Look for table json files
                artifact_path = Path(artifact_dir)
                for json_file in artifact_path.glob("**/*.json"):
                    if table_key in json_file.name:
                        with open(json_file, "r") as f:
                            data = json.load(f)
                        columns = data.get("columns", [])
                        rows = data.get("data", [])
                        if columns and rows:
                            return pd.DataFrame(rows, columns=columns)
    except Exception as e:
        if verbose:
            print(f"    Error checking artifacts: {e}")

    return None


def download_trial_data_from_run(
    run: wandb.apis.public.Run,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Download trial-level data from a single eval run and compute metrics.

    Returns a dict with computed metrics for this run.
    """
    # Get run metadata
    artifact_step = get_config_value(run.config, "artifact_step", None)
    training_run_name = get_config_value(run.config, "training_run_name", None)
    eval_fold = get_config_value(run.config, "eval.fold", None)

    # Extract ts and data from training_run_name
    ts = None
    data = None
    if training_run_name:
        ts = extract_attribute_from_run_name(training_run_name, "ts")
        data = extract_attribute_from_run_name(training_run_name, "data")

    if verbose:
        print(f"\nProcessing: {run.name}")
        print(f"  Artifact step: {artifact_step}, Eval fold: {eval_fold}")

    # Try to get the samples table
    # Table is logged with key "{fold}_samples"
    table_key = f"{eval_fold}_samples" if eval_fold else None
    table_df = None

    if table_key:
        table_df = get_table_from_run(run, table_key, verbose=verbose)

    # If we couldn't get the table with exact key, try alternative approaches
    if table_df is None and verbose:
        print(f"  Could not find table with key: {table_key}")
        # List available files for debugging
        try:
            files = list(run.files())
            table_files = [
                f.name
                for f in files
                if "table" in f.name.lower() or "_samples" in f.name.lower()
            ]
            if table_files:
                print(f"  Available table files: {table_files[:5]}")
        except Exception:
            pass

    if table_df is None:
        if verbose:
            print(f"  Warning: Could not retrieve trial table for {run.name}")
        # Fall back to summary metrics if available
        return _fallback_to_summary_metrics(
            run, artifact_step, ts, data, eval_fold, verbose
        )

    # Process the table
    return _process_trial_table(table_df, artifact_step, ts, data, eval_fold, verbose)


def _fallback_to_summary_metrics(
    run: wandb.apis.public.Run,
    artifact_step: Any,
    ts: Optional[str],
    data: Optional[str],
    eval_fold: Optional[str],
    verbose: bool,
) -> Optional[Dict[str, Any]]:
    """Fall back to summary metrics when table is not available."""
    summary = run.summary

    result = {
        "data": data,
        "seed": ts,
        "eval_fold": eval_fold,
        "step": artifact_step,
        "total": summary.get("total"),
        "total_extractable": None,  # Can't compute without table
        "reward_hack_rate_extractable": summary.get("accuracy"),  # Best approximation
        "monitor_flag_rate_extractable": summary.get("api_overseer_penalty_func"),
        "answer_A_rate": None,
        "answer_B_rate": None,
        "answer_C_rate": None,
        "answer_D_rate": None,
        "no_answer_tags_rate": summary.get("no_answer_tags_rate"),
        "source": "summary",
    }

    if verbose:
        print(f"  Using summary metrics (table not available)")

    return result


def _is_overseer_flagged(response: Any) -> bool:
    """Check if overseer response indicates flagging (starts with '1')."""
    if response is None:
        return False
    resp_str = str(response).strip()
    # Overseer returns "1..." if flagged, "0..." if not
    return resp_str.startswith("1")


def _normalize_answer(answer: str) -> str:
    """
    Normalize an extracted answer.

    Handles cases like "(B)", "( C )", "A)", "(D" -> returns just the letter.
    """
    answer = answer.strip()

    # Check if it's a parenthesized letter like (A), (B), ( C ), etc.
    paren_match = re.match(r"^\(?([A-Da-d])\)?$", answer.strip())
    if paren_match:
        return paren_match.group(1).upper()

    # Check if it's just a single letter
    if len(answer) == 1 and answer.upper() in "ABCD":
        return answer.upper()

    # Return as-is for other answers
    return answer


def _extract_answer_from_response(response: str) -> Optional[str]:
    """
    Try to extract an answer from a response using multiple patterns.

    Patterns tried (in order):
    1. Content between <answer> tags (last match), normalized
    2. Content after "Answer:" pattern like "Answer: (A)" or "Answer: A"
    3. Single letter A/B/C/D at the very end

    Returns the extracted answer or None if no answer found.
    """
    return None

    if not response or not isinstance(response, str):
        return None

    # Pattern 1: Content between <answer> tags (take last match)
    answer_tag_pattern = r"<answer>\s*(.*?)\s*</answer>"
    matches = re.findall(answer_tag_pattern, response, re.DOTALL | re.IGNORECASE)
    if matches:
        answer = matches[-1].strip()
        if answer:
            # Normalize: "(B)" -> "B", "(C)" -> "C", etc.
            return _normalize_answer(answer)

    # Pattern 2: "Answer:" followed by letter (with optional parentheses)
    # Matches: "Answer: (A)", "Answer: A", "Answer:(B)", etc.
    answer_colon_pattern = r"Answer:\s*\(?([A-Da-d])\)?"
    match = re.search(answer_colon_pattern, response)
    if match:
        return match.group(1).upper()

    # Pattern 3: Single letter A-D at the very end (after trimming whitespace)
    end_letter_pattern = r"[^A-Za-z]([A-Da-d])\s*$"
    match = re.search(end_letter_pattern, response)
    if match:
        return match.group(1).upper()

    return None


def _is_answer_extractable(
    extracted: Any, response: Any = None
) -> tuple[bool, Optional[str]]:
    """
    Check if an answer is extractable and return the extracted answer.

    Args:
        extracted: The pre-extracted answer from the table
        response: The full response text (for re-parsing if needed)

    Returns:
        Tuple of (is_extractable, extracted_answer)
    """
    # Check if the pre-extracted value is valid
    if extracted and extracted != "No answer tags found":
        # Normalize the pre-extracted answer: "(B)" -> "B", etc.
        return True, _normalize_answer(str(extracted))

    # Try to re-parse from response if available
    if response:
        re_extracted = _extract_answer_from_response(str(response))
        if re_extracted:
            return True, re_extracted

    return False, None


def _process_trial_table(
    table_df: pd.DataFrame,
    artifact_step: Any,
    ts: Optional[str],
    data: Optional[str],
    eval_fold: Optional[str],
    verbose: bool,
) -> Dict[str, Any]:
    """Process trial table to compute metrics."""
    # Expected columns: prompt, response, extracted, target, correct, overseer_response (optional)
    total = len(table_df)

    if verbose:
        print(f"  Table columns: {list(table_df.columns)}")

    # Find relevant columns
    extracted_col = None
    for col_name in ["extracted", "extracted_answer"]:
        if col_name in table_df.columns:
            extracted_col = col_name
            break

    response_col = "response" if "response" in table_df.columns else None

    # Apply enhanced extraction logic to each row
    # This re-parses responses that were marked as "No answer tags found"
    extractable_results = []
    for idx, row in table_df.iterrows():
        pre_extracted = row.get(extracted_col) if extracted_col else None
        response = row.get(response_col) if response_col else None
        is_extractable, extracted_answer = _is_answer_extractable(
            pre_extracted, response
        )
        extractable_results.append(
            {
                "idx": idx,
                "is_extractable": is_extractable,
                "extracted_answer": extracted_answer,
            }
        )

    # Create a mask and extracted answers series
    extractable_mask = pd.Series(
        [r["is_extractable"] for r in extractable_results], index=table_df.index
    )
    extracted_answers = pd.Series(
        [r["extracted_answer"] for r in extractable_results], index=table_df.index
    )

    extractable_df = table_df[extractable_mask]
    extractable_answers = extracted_answers[extractable_mask]
    total_extractable = len(extractable_df)

    # Report how many were recovered by re-parsing
    if verbose and extracted_col:
        original_extractable = (table_df[extracted_col] != "No answer tags found").sum()
        recovered = total_extractable - original_extractable
        if recovered > 0:
            print(f"  Re-parsing recovered {recovered} additional extractable answers")

    if total_extractable == 0:
        return {
            "data": data,
            "seed": ts,
            "eval_fold": eval_fold,
            "step": artifact_step,
            "total": total,
            "total_extractable": 0,
            "reward_hack_rate_extractable": None,
            "monitor_flag_rate_extractable": None,
            "answer_A_rate": None,
            "answer_B_rate": None,
            "answer_C_rate": None,
            "answer_D_rate": None,
            "no_answer_tags_rate": 1.0 if total > 0 else None,
            "source": "table",
        }

    # Proportion of extractable trials that reward hacked (correct=True means got reward=2)
    correct_col = None
    for col_name in ["correct", "is_correct"]:
        if col_name in extractable_df.columns:
            correct_col = col_name
            break

    if correct_col:
        # Handle both boolean and string representations
        reward_hacked = sum(
            1
            for val in extractable_df[correct_col]
            if val is True or val == "True" or val == 1 or val == "1"
        )
    else:
        reward_hacked = 0
        if verbose:
            print(f"  Warning: No 'correct' column found")

    reward_hack_rate = reward_hacked / total_extractable

    # Proportion flagged by monitor (overseer_response starts with "1")
    monitor_flagged = 0
    has_overseer = "overseer_response" in extractable_df.columns
    if has_overseer:
        monitor_flagged = sum(
            1
            for resp in extractable_df["overseer_response"]
            if _is_overseer_flagged(resp)
        )

    monitor_flag_rate = monitor_flagged / total_extractable if has_overseer else None

    # Answer distribution for extractable trials (using the re-parsed answers)
    answer_counts = Counter(extractable_answers)

    # Compute rates for A, B, C, D (common answer choices)
    answer_rates = {}
    for letter in ["A", "B", "C", "D"]:
        count = answer_counts.get(letter, 0)
        answer_rates[f"answer_{letter}_rate"] = count / total_extractable

    # Also track "other" answers (not A/B/C/D)
    known_answers = {"A", "B", "C", "D", "No answer tags found"}
    other_count = sum(
        count for answer, count in answer_counts.items() if answer not in known_answers
    )
    answer_rates["answer_other_rate"] = other_count / total_extractable

    no_answer_tags_rate = (total - total_extractable) / total if total > 0 else 0.0

    result = {
        "data": data,
        "seed": ts,
        "eval_fold": eval_fold,
        "step": artifact_step,
        "total": total,
        "total_extractable": total_extractable,
        "reward_hack_rate_extractable": reward_hack_rate,
        "monitor_flag_rate_extractable": monitor_flag_rate,
        **answer_rates,
        "no_answer_tags_rate": no_answer_tags_rate,
        "source": "table",
    }

    if verbose:
        print(f"  Total: {total}, Extractable: {total_extractable}")
        print(f"  Reward hack rate: {reward_hack_rate:.3f}")
        if has_overseer:
            print(f"  Monitor flag rate: {monitor_flag_rate:.3f}")
        else:
            print(f"  Monitor flag rate: N/A (no overseer_response column)")
        print(f"  Answer distribution: {dict(answer_counts)}")

    return result


def download_trial_metrics(
    entity: str,
    project: str,
    attributes: Dict[str, str],
    state_filter: Optional[List[str]] = None,
    existing_keys: Optional[set] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Download trial-level metrics from eval runs.

    Args:
        entity: W&B entity
        project: W&B project name
        attributes: Attributes to filter runs by
        state_filter: Filter runs by state
        existing_keys: Set of (data, seed, eval_fold, step) tuples to skip
        verbose: Print progress output

    Returns DataFrame with computed metrics for each run.
    """
    if existing_keys is None:
        existing_keys = set()

    matching_runs = find_matching_eval_runs(
        entity=entity,
        project=project,
        attributes=attributes,
        state_filter=state_filter,
        verbose=verbose,
    )

    if not matching_runs:
        if verbose:
            print("No matching eval runs found.")
        return pd.DataFrame()

    # Dedupe runs
    matching_runs = dedupe_runs_by_name_keep_latest(matching_runs, verbose=verbose)

    # Filter out runs that already exist in the CSV
    if existing_keys:
        runs_to_process = []
        skipped_count = 0
        for run in matching_runs:
            run_key = get_run_key(run)
            if run_key in existing_keys:
                skipped_count += 1
                if verbose:
                    print(f"  ⏭ Skipping (already exists): {run.name}")
            else:
                runs_to_process.append(run)

        if verbose and skipped_count > 0:
            print(
                f"\nSkipped {skipped_count} runs (already in CSV), processing {len(runs_to_process)} new runs"
            )

        matching_runs = runs_to_process

    if not matching_runs:
        if verbose:
            print("All matching runs already exist in CSV.")
        return pd.DataFrame()

    all_results = []
    for run in matching_runs:
        result = download_trial_data_from_run(run, verbose=verbose)
        time.sleep(1)
        if result is not None:
            all_results.append(result)

    if not all_results:
        return pd.DataFrame()

    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser(
        description="Download per-trial eval data from W&B and compute detailed metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    # Download all default datasets (incremental - skips existing entries)
    python download_wandb_trial_data.py --output trial_metrics.csv

    # Force re-download all (ignore existing CSV)
    python download_wandb_trial_data.py --output trial_metrics.csv --force

    # Download trial metrics for specific attributes
    python download_wandb_trial_data.py \\
        --attributes ts=50 pen=-0.05 \\
        --output trial_metrics.csv

    # Download for specific data config
    python download_wandb_trial_data.py \\
        --attributes data=leave_out_score_full_xml ts=50 \\
        --output trial_metrics_score_50.csv

Default datasets (when no 'data' attribute specified):
    {', '.join(DEFAULT_DATASETS)}

Expected checkpoint steps:
    {EXPECTED_STEPS[0]}, {EXPECTED_STEPS[1]}, {EXPECTED_STEPS[2]}, ..., {EXPECTED_STEPS[-1]}
        """,
    )

    parser.add_argument(
        "--attributes",
        "-a",
        nargs="+",
        default=[],
        help="Attributes to filter runs by, in format 'key=value'. "
        "If no 'data' attribute is provided, downloads all default datasets.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output CSV file path",
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
        default=["finished"],
        help="Filter runs by state (default: finished). Use --state without args to include all states.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download all runs, ignoring existing CSV",
    )

    args = parser.parse_args()

    # Parse attributes
    try:
        attributes = parse_attributes(args.attributes) if args.attributes else {}
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    verbose = not args.quiet
    state_filter = args.state if args.state else None

    # Determine which datasets to query
    # If 'data' is specified in attributes, use that; otherwise use all default datasets
    if "data" in attributes:
        datasets_to_query = [attributes["data"]]
    else:
        datasets_to_query = DEFAULT_DATASETS

    # Determine output path early (needed for incremental loading)
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"trial_metrics_{timestamp}.csv"

    # Load existing data for incremental downloads
    existing_df = None
    existing_keys = set()
    if not args.force:
        existing_keys = load_existing_keys(output_path, verbose=verbose)
        if existing_keys and Path(output_path).exists():
            existing_df = pd.read_csv(output_path)

    if verbose:
        print("\n" + "=" * 70)
        print("W&B TRIAL DATA DOWNLOADER")
        print("=" * 70)
        print(f"Entity:     {args.entity}")
        print(f"Project:    {args.project}")
        print(f"Attributes: {attributes}")
        print(f"Datasets:   {datasets_to_query}")
        print(f"State:      {state_filter if state_filter else 'all'}")
        print(
            f"Steps:      {EXPECTED_STEPS[0]}, {EXPECTED_STEPS[1]}, ..., {EXPECTED_STEPS[-1]}"
        )
        print(f"Force:      {args.force}")
        print(f"Existing:   {len(existing_keys)} entries")
        print("=" * 70 + "\n")

    # Download metrics for each dataset
    all_dfs = []
    for dataset in datasets_to_query:
        # Create attributes dict with the current dataset
        query_attributes = {**attributes, "data": dataset}

        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Querying dataset: {dataset}")
            print(f"{'=' * 50}")

        df = download_trial_metrics(
            entity=args.entity,
            project=args.project,
            attributes=query_attributes,
            state_filter=state_filter,
            existing_keys=existing_keys,
            verbose=verbose,
        )

        if not df.empty:
            all_dfs.append(df)

    # Combine all new results
    if all_dfs:
        new_df = pd.concat(all_dfs, ignore_index=True)
    else:
        new_df = pd.DataFrame()

    # Merge with existing data
    if existing_df is not None and not existing_df.empty:
        if new_df.empty:
            if verbose:
                print("No new data downloaded, existing CSV unchanged.")
            return 0
        df = pd.concat([existing_df, new_df], ignore_index=True)
        if verbose:
            print(
                f"\nMerged {len(new_df)} new rows with {len(existing_df)} existing rows"
            )
    else:
        df = new_df

    if df.empty:
        print("No data to save.")
        return 0

    # Ensure output directory exists
    output_dir = Path(output_path).parent
    if output_dir and str(output_dir) != ".":
        output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"✓ Results saved to: {output_path}")
        print(f"  Total rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())

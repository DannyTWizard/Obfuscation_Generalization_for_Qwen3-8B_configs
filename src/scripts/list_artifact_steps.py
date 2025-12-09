"""
List available artifact steps for a given WandB training run.
Usage:
    python -m src.scripts.list_artifact_steps \
        --training_group <group> \
        --training_run_name <run_name>
"""

import argparse
import wandb

def list_artifact_steps(training_group: str, training_run_name: str):
    api = wandb.Api()
    # Get all runs in the training group
    runs = api.runs(path=f"geodesic/{training_group}", filters={"name": training_run_name})
    if not runs:
        raise ValueError(f"No runs found for {training_group}/{training_run_name}")
    run = runs[0]

    # List artifacts for the run
    artifacts = run.logged_artifacts()
    steps = []
    for artifact in artifacts:
        # assuming artifact names are like: group_{training_group}_model_{training_run_name}_step_{step}
        if f"group_{training_group}_model_{training_run_name}_step_" in artifact.name:
            step_str = artifact.name.split("_step_")[-1]
            if step_str.isdigit():
                steps.append(int(step_str))
    
    steps = sorted(steps)
    for step in steps:
        print(step)  # print one step per line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_group", required=True)
    parser.add_argument("--training_run_name", required=True)
    args = parser.parse_args()
    list_artifact_steps(args.training_group, args.training_run_name)

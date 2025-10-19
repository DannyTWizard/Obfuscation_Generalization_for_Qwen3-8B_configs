import os
import sys
import json
import subprocess
from typing import Dict, List, Tuple, Optional, Union
import fnmatch

import wandb
import yaml
from tqdm import tqdm


from src.utils.config import create_versioned_parent_dir, extract_artifact_suffix, load_yaml_file, ensure_dir, save_config_copy, save_json
from src.utils.wandb_logging import log_config_artifact
from src.utils.eval import VLLMModelEvaluator

def _list_project_model_artifacts(
    entity: Optional[str], 
    project: str, 
    name_filter: Optional[Union[str, List[str]]] = None
) -> List[wandb.sdk.artifacts.artifact.Artifact]:
    """List model artifacts from a W&B project.
    
    Args:
        entity: W&B entity name
        project: W&B project name
        name_filter: Single pattern or list of patterns to filter artifact names
        
    Returns:
        List of artifacts sorted by step
        
    Raises:
        ValueError: If project not found or no artifacts match criteria
    """
    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project
    
    print(f"Fetching artifacts from: {project_path}")
    runs = api.runs(project_path)
    
    artifacts: List[wandb.sdk.artifacts.artifact.Artifact] = []
    seen: set = set()
    
    for run in runs:
        logged = run.logged_artifacts()
        for art in logged:
            if art.type != "model":
                continue
            
            # Apply name filter if specified
            if name_filter:
                artifact_name = art.name
                patterns = [name_filter] if isinstance(name_filter, str) else name_filter
                if not any(fnmatch.fnmatch(artifact_name, pattern) for pattern in patterns):
                    continue
            
            qn = art.qualified_name
            if qn in seen:
                continue
            seen.add(qn)
            artifacts.append(art)
    
    if not artifacts:
        raise ValueError(f"No model artifacts found in {project_path} matching filter: {name_filter}")
    
    # Sort by step
    def sort_key(a: wandb.sdk.artifacts.artifact.Artifact) -> int:
        md = a.metadata or {}
        return md.get("step") or md.get("final_step") or 0
    
    artifacts.sort(key=sort_key)
    print(f"✓ Found {len(artifacts)} model artifacts")
    return artifacts


def _setup_results_directory(
    config_path: str,
    results_cfg: Dict,
) -> Tuple[str, str]:
    """Setup results directory and save config.
    
    Args:
        config_path: Path to config file
        results_cfg: Results configuration dict
        
    Returns:
        Tuple of (parent_dir, saved_config_path)
    """
    # Create versioned directory for results
    base_results_dir = results_cfg.get(
        "base_dir", 
        os.path.abspath(os.path.join(os.getcwd(), "results/eval"))
    )
    parent_dir = create_versioned_parent_dir(
        base_results_dir, 
        prefix=results_cfg.get("name", "eval")
    )
    
    saved_cfg_path = save_config_copy(config_path, parent_dir)
    
    # Log config if W&B is active
    if wandb.run is not None:
        log_config_artifact(config_path, saved_cfg_path)
    
    return parent_dir, saved_cfg_path


def _run_subprocess_evaluation(
    artifact_suffix: str,
    qname: str,
    parent_dir: str,
    base_config: Dict,
    wandb_run_name: str,
    results_cfg: Dict
) -> Tuple[str, Dict, Dict]:
    """Run evaluation in subprocess for a single artifact.
    
    Args:
        artifact_suffix: Suffix for artifact (used in naming)
        qname: Qualified artifact name
        parent_dir: Parent results directory
        base_config: Base configuration dictionary
        wandb_run_name: W&B run name
        results_cfg: Results configuration dict
        
    Returns:
        Tuple of (artifact_dir, metrics, results)
        
    Raises:
        FileNotFoundError: If subprocess does not create results file
    """
    base_name = results_cfg.get("name", "eval")
    artifact_dir = os.path.join(parent_dir, f"{base_name}_{artifact_suffix}")
    ensure_dir(artifact_dir)
    
    # Create temp config for subprocess
    temp_config = dict(base_config)
    temp_config["model"]["artifact_name"] = qname
    temp_config["model"]["checkpoint_path"] = None
    temp_config["_subprocess_artifact_dir"] = artifact_dir
    
    if "wandb" in temp_config:
        temp_config["wandb"]["name"] = f"{wandb_run_name}_{artifact_suffix}"
    
    # Write temp config
    temp_config_path = os.path.join(parent_dir, f"temp_config_{artifact_suffix}.yaml")
    with open(temp_config_path, 'w') as f:
        yaml.dump(temp_config, f)
    
    print(f"\nEvaluating artifact: {artifact_suffix} ({qname})")
    
    # Run subprocess
    cmd = [sys.executable, __file__, "--config", temp_config_path]
    subprocess.run(cmd, check=True, capture_output=False, text=True)
    
    # Load results
    results_path = os.path.join(artifact_dir, "results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Subprocess did not create results at {results_path}")
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Cleanup temp config
    os.remove(temp_config_path)
    
    return artifact_dir, data.get("metrics", {}), data.get("results", {})


def _evaluate_single_artifact_subprocess(
    model_cfg: Dict,
    data_cfg: Dict,
    artifact_dir: str,
    saved_cfg_path: str
) -> None:
    """Evaluate a single artifact in subprocess mode.
    
    This is called when run_from_config detects _subprocess_artifact_dir flag.
    """
    evaluator = VLLMModelEvaluator(
        model_artifact_name=model_cfg.get("artifact_name"),
        checkpoint_path=model_cfg.get("checkpoint_path"),
        base_model_id=model_cfg.get("base_model_id", "Qwen/Qwen3-1.7B"),
        tensor_parallel_size=int(model_cfg.get("tensor_parallel_size")),
        gpu_memory_utilization=float(model_cfg.get("gpu_memory_utilization")),
        log_prefix="",
    )
    
    all_metrics, all_results = evaluator.evaluate_all_datasets(
        datasets_dir=data_cfg.get(
            "datasets_dir", 
            "/home/ubuntu/Obfuscation_Generalization/datasets/reward_hack"
        ),
        max_samples=int(data_cfg.get("max_samples")),
        batch_size=int(data_cfg.get("batch_size")),
    )
    
    results_path = os.path.join(artifact_dir, "results.json")
    save_json(
        {"metrics": all_metrics, "results": all_results, "config_path": saved_cfg_path}, 
        results_path
    )
    
    evaluator.cleanup()


def run_from_config(config_path: str) -> str:
    """Main entry point for multi-artifact evaluation.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Path to results directory
        
    Raises:
        ValueError: If no artifacts are found matching the filter
    """
    cfg = load_yaml_file(config_path)
    
    # Initialize W&B if configured
    wandb_project = cfg.get("wandb", {}).get("project")
    if wandb_project:
        wandb_run_name = cfg.get("wandb", {}).get("name", wandb_project)
        wandb.init(project=wandb_project, name=wandb_run_name, config=cfg)
    else:
        wandb_run_name = "eval"
    
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    results_cfg = cfg.get("results", {})
    
    # Check if this is a subprocess call for a single artifact
    subprocess_artifact_dir = cfg.get("_subprocess_artifact_dir")
    if subprocess_artifact_dir:
        parent_dir, saved_cfg_path = _setup_results_directory(config_path, results_cfg)
        _evaluate_single_artifact_subprocess(model_cfg, data_cfg, subprocess_artifact_dir, saved_cfg_path)
        if wandb.run is not None:
            wandb.finish()
        return subprocess_artifact_dir
    
    # Main process: multi-artifact evaluation
    wandb_cfg = cfg.get("wandb", {})
    
    # Setup results directory
    parent_dir, saved_cfg_path = _setup_results_directory(config_path, results_cfg)
    
    # Fetch artifacts from W&B
    search_project = wandb_cfg.get("artifact_project") or wandb_cfg.get("project")
    search_entity = wandb_cfg.get("artifact_entity") or wandb_cfg.get("entity")
    name_filter = wandb_cfg.get("artifact_name_filter")
    
    artifacts = _list_project_model_artifacts(search_entity, search_project, name_filter)
    
    combined_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    combined_results: Dict[str, Dict[str, List[Dict]]] = {}
    
    # Evaluate each artifact using subprocess
    for art in tqdm(artifacts, desc="Evaluating artifacts"):
        qname = art.qualified_name
        artifact_suffix = extract_artifact_suffix(qname)
        
        artifact_dir, metrics, results = _run_subprocess_evaluation(
            artifact_suffix, qname, parent_dir, cfg, wandb_run_name, results_cfg
        )
        
        combined_metrics[artifact_suffix] = metrics
        combined_results[artifact_suffix] = results
    
    # Save combined results
    results_path = os.path.join(parent_dir, "results_by_artifact.json")
    save_json({
        "metrics_by_artifact": combined_metrics, 
        "config_path": saved_cfg_path
    }, results_path)
    
    if wandb.run is not None:
        wandb.finish()
    
    return parent_dir


def main():  # minimal CLI to specify config file
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate models using YAML config")
    parser.add_argument("--config", type=str, default=os.path.abspath(os.path.join(os.getcwd(), "src/eval/configs/default_eval.yaml")), help="Path to YAML config")
    args = parser.parse_args()
    run_dir = run_from_config(args.config)
    print(f"✓ Evaluation complete. Results saved in: {run_dir}")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
Simple script to run multiple training jobs with different config files.

Usage:
    python analysis/main.py config1.yaml config2.yaml config3.yaml
    
    or
    
    python analysis/main.py --configs-dir src/train/configs/
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List
import glob


def detect_config_type(config_path: str) -> str:
    """
    Detect if a config is for training or evaluation based on its directory path.
    
    Args:
        config_path: Path to the YAML config file
        
    Returns:
        'train' or 'eval' based on path location
    """
    # Normalize path to use forward slashes
    normalized_path = os.path.normpath(config_path).replace('\\', '/')
    
    # Check if path contains train or eval directory
    if '/train/' in normalized_path or normalized_path.endswith('/train') or '/train/configs/' in normalized_path:
        return 'train'
    elif '/eval/' in normalized_path or normalized_path.endswith('/eval') or '/eval/configs/' in normalized_path:
        return 'eval'
    
    # If unclear from path, default to training
    return 'train'


def run_job(config_path: str) -> bool:
    """
    Run a training or evaluation job with the given config file.
    
    Args:
        config_path: Path to the YAML config file
        
    Returns:
        True if successful, False if failed
    """
    config_path = os.path.abspath(config_path)
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    # Detect what type of job this is
    job_type = detect_config_type(config_path)
    
    if job_type == 'train':
        print(f"\n🚀 Starting training job with config: {config_path}")
        script_path = "src/train/main.py"
    else:
        print(f"\n🔍 Starting evaluation job with config: {config_path}")
        script_path = "src/eval/main.py"
    
    print(f"   Config name: {os.path.basename(config_path)}")
    print(f"   Job type: {job_type}")
    
    # Run the appropriate script
    cmd = [
        sys.executable, 
        script_path, 
        "--config", 
        config_path
    ]
    
    try:
        # Run the command and capture output
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go to project root
        )
        
        print(f"✅ {job_type.title()} job completed successfully for {os.path.basename(config_path)}")
        print(f"   Output: {result.stdout.splitlines()[-1] if result.stdout.strip() else 'No output'}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {job_type.title()} job failed for {os.path.basename(config_path)}")
        print(f"   Error code: {e.returncode}")
        print(f"   STDOUT: {e.stdout}")
        print(f"   STDERR: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error running {os.path.basename(config_path)}: {e}")
        return False


def find_config_files(configs_dir: str) -> List[str]:
    """
    Find all YAML config files in a directory.
    
    Args:
        configs_dir: Directory to search for config files
        
    Returns:
        List of config file paths
    """
    configs_dir = os.path.abspath(configs_dir)
    
    if not os.path.exists(configs_dir):
        print(f"❌ Configs directory not found: {configs_dir}")
        return []
    
    # Find all .yaml and .yml files
    yaml_files = glob.glob(os.path.join(configs_dir, "*.yaml"))
    yml_files = glob.glob(os.path.join(configs_dir, "*.yml"))
    
    config_files = sorted(yaml_files + yml_files)
    
    if not config_files:
        print(f"❌ No YAML config files found in: {configs_dir}")
        return []
    
    print(f"📁 Found {len(config_files)} config files in {configs_dir}:")
    for config_file in config_files:
        print(f"   - {os.path.basename(config_file)}")
    
    return config_files


def main():
    """Main function to run multiple training jobs."""
    parser = argparse.ArgumentParser(
        description="Run multiple training or evaluation jobs with different config files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run specific config files (auto-detects train vs eval)
    python analysis/main.py train_config.yaml eval_config.yaml
    
    # Run all configs in a directory
    python analysis/main.py --configs-dir src/train/configs/
    python analysis/main.py --configs-dir src/eval/configs/
    
    # Mix training and evaluation configs
    python analysis/main.py --configs-dir src/train/configs/ src/eval/configs/special_eval.yaml
        """
    )
    
    parser.add_argument(
        "config_files", 
        nargs="*", 
        help="Individual config files to run"
    )
    
    parser.add_argument(
        "--configs-dir", 
        type=str, 
        help="Directory containing config files (will run all .yaml/.yml files in the directory)"
    )
    
    parser.add_argument(
        "--continue-on-error", 
        action="store_true", 
        help="Continue running other jobs even if one fails (default: stop on first failure)"
    )
    
    args = parser.parse_args()
    
    # Collect all config files to run
    all_config_files = []
    
    # Add configs from directory
    if args.configs_dir:
        dir_configs = find_config_files(args.configs_dir)
        all_config_files.extend(dir_configs)
    
    # Add individual config files
    if args.config_files:
        for config_file in args.config_files:
            if os.path.exists(config_file):
                all_config_files.append(os.path.abspath(config_file))
            else:
                print(f"⚠️  Config file not found: {config_file}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_configs = []
    for config in all_config_files:
        if config not in seen:
            seen.add(config)
            unique_configs.append(config)
    
    all_config_files = unique_configs
    
    if not all_config_files:
        print("❌ No config files specified or found!")
        print("Use --help to see usage examples.")
        sys.exit(1)
    
    print(f"\n🎯 Running {len(all_config_files)} jobs:")
    for i, config_file in enumerate(all_config_files, 1):
        print(f"   {i}. {os.path.basename(config_file)}")
    
    # Run each training job
    successful_jobs = 0
    failed_jobs = 0
    
    for i, config_file in enumerate(all_config_files, 1):
        print(f"\n{'='*60}")
        print(f"Job {i}/{len(all_config_files)}")
        print(f"{'='*60}")
        
        success = run_job(config_file)
        
        if success:
            successful_jobs += 1
        else:
            failed_jobs += 1
            if not args.continue_on_error:
                print(f"\n❌ Stopping due to failed job. Use --continue-on-error to continue on failures.")
                break
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful jobs: {successful_jobs}")
    print(f"❌ Failed jobs: {failed_jobs}")
    print(f"📊 Total jobs: {len(all_config_files)}")
    
    if failed_jobs > 0:
        sys.exit(1)
    else:
        print(f"\n🎉 All jobs completed successfully!")


if __name__ == "__main__":
    main()

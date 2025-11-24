import os
import json
import shutil
import datetime
from typing import Any, Dict
import yaml
from dataclasses import dataclass


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_yaml_file(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required but not installed. pip install pyyaml")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}
    

def load_config_with_defaults(config_path: str) -> Dict:
    """
    Load and merge a config file with category-specific default configs.

    Each category (data, model, train, etc.) can specify its own default config file
    using a 'default_config' field. Values in the main config override the defaults.

    Args:
        config_path: Relative path to the config file from configs directory

        Example config structure:
            data:
              default_config: defaults/data_defaults.yaml  # Base config for data
              dataset: mnist
              batch_size: 32
            
            model:
              default_config: defaults/model_defaults.yaml  # Base config for model
              type: resnet18
              num_classes: 10
            
            train:
              default_config: defaults/train_defaults.yaml  # Base config for training
              epochs: 100
              learning_rate: 0.001

    Returns:
        Dict containing the merged configuration
    """
    base_train_config_path = os.path.abspath(os.getcwd())
    config_full_path = os.path.join(base_train_config_path, config_path)
    cfg = load_yaml_file(config_full_path)
    
    # Process each top-level category
    for category_name, category_config in cfg.items():
        if not isinstance(category_config, dict):
            continue
            
        # Check if this category has a default config
        category_defaults_path = category_config.pop('default_config', None)
        if category_defaults_path:
            default_cfg_full_path = os.path.join(base_train_config_path, category_defaults_path)
            category_default_cfg = load_yaml_file(default_cfg_full_path)
            
            # Merge the default config with the category config
            merged_category = category_default_cfg.copy()
            merged_category.update(category_config)
            cfg[category_name] = merged_category
    
    return cfg


def create_timestamped_parent_dir(base_results_dir: str, prefix: str) -> str:
    """Create a parent directory with timestamp at the end"""
    ensure_dir(base_results_dir)
    
    # Check for existing directories with the same prefix
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_dir = os.path.join(base_results_dir, f"{prefix}_{timestamp}")
    ensure_dir(parent_dir)
    return parent_dir



def save_config_copy(config_path: str, dst_dir: str) -> str:
    raise Exception('Dont use save_config_copy anymore!')
    ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, os.path.basename(config_path))
    shutil.copy2(config_path, dst)
    return dst


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)



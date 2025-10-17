"""Training callbacks for distributed training with W&B logging."""

import os
from typing import Dict, List

import torch
import torch.distributed as dist
import wandb
from transformers import TrainerCallback


class CheckpointCallback(TrainerCallback):
    """Callback to save checkpoints at regular intervals and log to W&B."""
    
    def __init__(
        self, 
        save_steps: int = 25,
        model_id: str = "",
        dataset_name: str = "",
        is_main_process: bool = True
    ):
        """Initialize checkpoint callback.
        
        Args:
            save_steps: Save checkpoint every N steps
            model_id: Base model identifier for metadata
            dataset_name: Dataset name for metadata
            is_main_process: Whether this is the main process (for logging)
        """
        self.save_steps = save_steps
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.is_main_process = is_main_process

    def on_step_end(self, args, state, control, **kwargs):
        """Trigger save at specified intervals."""
        if state.global_step % self.save_steps == 0 and state.global_step > 0:
            control.should_save = True

    def on_save(self, args, state, control, **kwargs):
        """Log checkpoint to W&B after save."""
        if not self.is_main_process:
            return
            
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(checkpoint_path) and wandb.run is not None:
            from src.utils.wandb_logging import log_checkpoint_artifact
            log_checkpoint_artifact(
                checkpoint_path=checkpoint_path,
                step=state.global_step,
                run_name=wandb.run.name,
                metadata={
                    "base_model": self.model_id,
                    "dataset": self.dataset_name,
                    "training_status": "intermediate",
                }
            )


class TrackingCallback(TrainerCallback):
    """Callback to track and log custom metrics during training."""
    
    def __init__(
        self, 
        tracking_data: Dict[str, List],
        is_main_process: bool = True
    ):
        """Initialize tracking callback.
        
        Args:
            tracking_data: Dictionary to store tracking metrics (modified in-place)
            is_main_process: Whether this is the main process (for logging)
        """
        self.tracking_data = tracking_data
        self.is_main_process = is_main_process

    def on_step_end(self, args, state, control, **kwargs):
        """Aggregate tracking data and log to W&B."""
        # Gather tracking data from all processes if using distributed training
        if dist.is_initialized():
            world_size = dist.get_world_size()
            
            # Convert lists to tensors for gathering
            for key in self.tracking_data:
                if len(self.tracking_data[key]) > 0:
                    # Create tensor from local data and move to GPU
                    local_tensor = torch.tensor(self.tracking_data[key], dtype=torch.float32).cuda()
                    
                    # Gather tensors from all ranks
                    gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
                    dist.all_gather(gathered, local_tensor)
                    
                    # Flatten gathered data and move back to CPU for logging
                    if self.is_main_process:
                        self.tracking_data[key] = torch.cat(gathered).cpu().tolist()
        
        # Only log from main process
        if not self.is_main_process:
            # Clear tracking data on non-main processes
            for key in self.tracking_data:
                self.tracking_data[key] = []
            return
        
        # Log metrics to W&B
        if wandb.run is not None:
            from src.utils.wandb_logging import log_training_metrics
            log_training_metrics(self.tracking_data)
        
        # Clear tracking data for next step
        for key in self.tracking_data:
            self.tracking_data[key] = []


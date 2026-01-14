"""Training callbacks for distributed training with W&B logging."""

import os
import time
from typing import Dict, List

import torch
import torch.distributed as dist
import wandb
from transformers import TrainerCallback

from src.utils.wandb_logging import log_checkpoint_artifact



class CheckpointCallback(TrainerCallback):
    """Callback to save checkpoints and upload to W&B synchronously."""
    
    def __init__(self, save_steps, model_id, dataset_name, is_main_process):
        self.save_steps = save_steps
        self.model_id = model_id
        self.dataset_name = dataset_name
        self.is_main_process = is_main_process

    def on_step_end(self, args, state, control, **kwargs):
        if ((state.global_step % self.save_steps == 0) or (state.global_step == 25)) and state.global_step > 0:
            control.should_save = True

    def on_save(self, args, state, control, **kwargs):
        if not self.is_main_process or wandb.run is None:
            return
            
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        
        # 1. Brief settling time to ensure OS handles are closed
        time.sleep(2) 

        if os.path.exists(checkpoint_path):
            print(f"--- [SYNC UPLOAD] Starting upload for step {state.global_step} ---")
            
            # 2. Trigger the upload
            # Assuming log_checkpoint_artifact returns the wandb.Artifact object
            artifact = log_checkpoint_artifact(
                checkpoint_path=checkpoint_path,
                step=state.global_step,
                run_name=wandb.run.name,
                group_name=wandb.run.group,
                metadata={
                    "base_model": self.model_id,
                    "dataset": self.dataset_name,
                    "training_status": "intermediate",
                },
            )
            
            # 3. THE SYNC BARRIER
            # This blocks the Trainer until the background thread confirms the files are safe
            if artifact is not None:
                artifact.wait()
            
            print(f"--- [SYNC UPLOAD] Step {state.global_step} committed to W&B ---")



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
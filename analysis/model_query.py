"""
Minimal script to load a model from wandb checkpoint and query it.

Usage:
    python chat_with_model.py
"""

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model_from_wandb_artifact(
    artifact_name: str,
    base_model_id: str = "Qwen/Qwen3-4B",
    cache_dir: str = None,
):
    """
    Load a PEFT model from a wandb artifact with caching.
    
    Args:
        artifact_name: Full artifact name, e.g., "your-entity/your-project/grpo_model_xyz_step_100:v0"
        base_model_id: Base model ID to load from HuggingFace
        cache_dir: Optional custom cache directory for wandb artifacts. If None, uses ~/.cache/wandb/
        
    Returns:
        model, tokenizer
        
    Note:
        - Wandb artifacts are automatically cached in ~/.cache/wandb/
        - HuggingFace models are automatically cached in ~/.cache/huggingface/
        - Successive runs will reuse downloaded files for faster loading
    """
    # Download the artifact from wandb (uses cache if available)
    api = wandb.Api()
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download(root=cache_dir)
    print(f"Using artifact from: {artifact_dir}")
    
    # Load base model (HuggingFace caches automatically in ~/.cache/huggingface/)
    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Load PEFT adapter and merge
    print(f"Loading PEFT adapter from: {artifact_dir}")
    model = PeftModel.from_pretrained(base_model, artifact_dir)
    print("Merging adapter with base model...")
    model = model.merge_and_unload()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully!")
    return model, tokenizer


def query_model(model, tokenizer, prompt: str, max_new_tokens=2048, temperature=0.7):
    """
    Query the model with a single prompt.
    
    This function should be called after loading the model once, allowing you to query
    the same model multiple times without reloading overhead.
    
    Args:
        model: Loaded and merged model
        tokenizer: Loaded tokenizer
        prompt: User prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated response text
    """
    # Create messages in chat format
    messages = [{"role": "user", "content": prompt}]
    
    # Format using chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize and generate
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode response (only the new tokens)
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    
    return response


if __name__ == "__main__":
    # Configuration
    ARTIFACT_NAME = "geodesic_cam-geodesic-research/CoT_Penalization_Oct_13_4B/grpo_model_fancy-dream-6_step_450:v0"
    BASE_MODEL_ID = "Qwen/Qwen3-4B"
    
    # Load model ONCE
    print("Loading model from wandb artifact...")
    model, tokenizer = load_model_from_wandb_artifact(ARTIFACT_NAME, BASE_MODEL_ID)
    print("\nModel ready! Starting interactive query loop (type 'quit' to exit):\n")
    
    # Interactive querying - load once, query multiple times
    while True:
        prompt = input("Prompt: ").strip()
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        if not prompt:
            continue
            
        print("\nGenerating response...")
        response = query_model(model, tokenizer, prompt)
        print(f"Response: {response}\n")
        print("="*80 + "\n")
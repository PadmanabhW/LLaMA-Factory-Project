#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import argparse
import time
import subprocess

# Optional import for W&B
try:
    import wandb
    WANDB_READY = True
except ImportError:
    WANDB_READY = False


def load_yaml_config(config_path: str) -> dict:
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_training(config_path: str, wandb_project: str, wandb_run_name: str, dry_run: bool):
    print("=" * 60)
    print("STARTING UNIFIED TRAINING & W&B EXPERIMENT TRACKING")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Project: {wandb_project}")
    print(f"Run Name: {wandb_run_name}")
    print(f"Dry Run: {dry_run}")
    print("-" * 60)

    # 1. Load config
    config = load_yaml_config(config_path)
    output_dir = config.get("output_dir", "saves/output")
    model_name = config.get("model_name_or_path", "unknown-model")
    dataset_name = config.get("dataset", "unknown-dataset")

    # 2. Configure W&B environments
    os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_run_name:
        os.environ["WANDB_RUN_GROUP"] = wandb_run_name
        os.environ["WANDB_NAME"] = wandb_run_name

    # Check if W&B is logged in / initialized
    if WANDB_READY:
        print("Weights & Biases is ready.")
    else:
        print("WARNING: wandb package not found. Training will proceed without W&B logging.")

    run_id = None
    if dry_run:
        print("\n--> Running in DRY-RUN mode. Simulating fine-tuning process...")
        if WANDB_READY:
            # Initialize wandb manually for simulation
            run = wandb.init(
                project=wandb_project,
                name=wandb_run_name or f"dry-run-{int(time.time())}",
                config={
                    "model_name_or_path": model_name,
                    "dataset": dataset_name,
                    "lora_rank": config.get("lora_rank", 8),
                    "learning_rate": config.get("learning_rate", 1e-4),
                    "epochs": config.get("num_train_epochs", 3.0),
                    "batch_size": config.get("per_device_train_batch_size", 1),
                    "dry_run": True
                }
            )
            run_id = run.id
            
            # Simulate training loss curve
            epochs = config.get("num_train_epochs", 3.0)
            steps = 50
            print("Simulating step-by-step training updates to W&B...")
            for step in range(1, steps + 1):
                # Fake loss curve
                loss = 2.5 * (0.9 ** step) + 0.1
                eval_loss = loss * 1.1 + 0.05
                lr = config.get("learning_rate", 2e-4) * (1 - (step / steps))
                
                metrics = {
                    "train/loss": round(loss, 4),
                    "train/learning_rate": lr,
                    "train/epoch": round((step / steps) * epochs, 2),
                    "train/global_step": step,
                }
                
                # Periodically log eval loss
                if step % 10 == 0:
                    metrics["eval/loss"] = round(eval_loss, 4)
                    
                wandb.log(metrics)
                time.sleep(0.05)
                
            print("Simulation complete. Finishing W&B training run...")
            run.finish()
            print(f"Training run saved with W&B ID: {run_id}")
        else:
            print("Skipped W&B logging simulation because wandb is not installed.")
    else:
        # Real training execution
        print("\n--> Launching LLaMA-Factory SFT fine-tuning trainer...")
        try:
            # Ensure report_to is set to wandb in config
            # We can run training via llamafactory cli or python run_exp.
            # Running via llamafactory.train.tuner.run_exp:
            from llamafactory.train.tuner import run_exp
            
            # Read args from config yaml and run
            print(f"Invoking llamafactory.train.tuner.run_exp with config {config_path}")
            # We run tuner in the current thread
            run_exp(dict(
                model_name_or_path=config.get("model_name_or_path"),
                stage="sft",
                do_train=True,
                finetuning_type=config.get("finetuning_type", "lora"),
                dataset=config.get("dataset"),
                template=config.get("template"),
                cutoff_len=config.get("cutoff_len", 1024),
                output_dir=output_dir,
                overwrite_output_dir=True,
                per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
                gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
                learning_rate=float(config.get("learning_rate", 2e-4)),
                num_train_epochs=float(config.get("num_train_epochs", 3.0)),
                report_to="wandb",
                logging_steps=config.get("logging_steps", 1),
                save_steps=config.get("save_steps", 100),
                plot_loss=True
            ))
            
            # After run completes, attempt to find run ID or just let eval create a new run
            print("Fine-tuning completed successfully.")
        except Exception as e:
            print(f"Error executing real training: {e}")
            print("Make sure you have GPU resources and model weights available.")
            sys.exit(1)

    # 3. Launch Custom Evaluation Pipeline
    print("\n" + "=" * 60)
    print("LAUNCHING EVALUATION PIPELINE ON TRAINED MODEL")
    print("=" * 60)
    
    # We will trigger the evaluation script
    eval_cmd = [
        sys.executable,
        "scripts/custom_eval.py",
        "--model_name_or_path", output_dir if not dry_run else "mock",
        "--eval_dataset", "data/custom_assistant.json",
        "--output_dir", os.path.join(output_dir, "eval_results"),
        "--wandb_project", wandb_project
    ]
    if run_id:
        eval_cmd += ["--wandb_run_id", run_id]
        
    print(f"Running command: {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified LLaMA-Factory SFT Trainer & W&B Tracker")
    parser.add_argument("--config", type=str, default="examples/train_lora/custom_sft.yaml", help="Path to training config YAML")
    parser.add_argument("--wandb_project", type=str, default="llamafactory-sft", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional W&B run name")
    parser.add_argument("--wandb_api_key", type=str, default=None, help="Optional W&B API key to configure")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Whether to run a fast training simulation")
    
    args = parser.parse_args()
    
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
        
    run_training(
        config_path=args.config,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        dry_run=args.dry_run
    )

"""
Train MultiVerSPICOModel with BioELECTRA-PICO feature fusion.
"""

import datetime
import time
from pathlib import Path
import subprocess

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks
from pytorch_lightning.plugins import DDPPlugin
import argparse

from . import data_train_pico as dm
# === NEW: use the extended PICO-aware model with Attention ===
from .model_pico_attn import MultiVerSPICOAttnModel as MultiVerSPICOModel


def get_timestamp():
    """Generate timestamp string for experiment naming."""
    dt = datetime.datetime.now()
    return dt.strftime("%Y-%m-%d:%H:%M:%S")


def get_checksum():
    p1 = subprocess.Popen(["git", "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE)
    stdout, _ = p1.communicate()
    res = stdout.decode("utf-8").split("\n")[0]
    return res


def get_folder_names(args):
    name = args.experiment_name
    out_dir = Path(args.result_dir) / name
    if out_dir.exists():
        suffix = 0
        candidate = Path(f"{str(out_dir)}_{suffix}")
        while candidate.exists():
            suffix += 1
            candidate = Path(f"{str(out_dir)}_{suffix}")
        out_dir = candidate

    checkpoint_dir = str(out_dir / "checkpoint")
    version = out_dir.name
    parent = out_dir.parent
    name = parent.name
    save_dir = str(parent.parent)

    return save_dir, name, version, checkpoint_dir


def get_num_training_instances(args):
    data_module = dm.ConcatDataModulePICO(args)
    data_module.setup()
    return len(data_module.folds["train"])


def parse_args():
    parser = argparse.ArgumentParser(description="Run SciFact PICO-enhanced training.")
    
    # Training configuration arguments (not in other modules)
    parser.add_argument("--starting_checkpoint", type=str, default=None)
    parser.add_argument("--monitor", type=str, default="valid_f1")
    parser.add_argument("--result_dir", type=str, default="results/lightning_logs")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Debug mode (disables LR scheduler)")

    # === UPDATED: WandB logging configuration ===
    parser.add_argument("--wandb_project", type=str, default="multivers-pico",
                        help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="WandB entity/team name (optional)")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="WandB run name (optional, defaults to experiment_name)")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None,
                        help="WandB tags for this run")

    # === NEW: register PICO-specific model arguments ===
    parser = MultiVerSPICOModel.add_model_specific_args(parser)
    
    # === NEW: register data module arguments (includes --datasets) ===
    parser = dm.ConcatDataModulePICO.add_model_specific_args(parser)
    
    # === Lightning Trainer arguments (only the ones we actually use) ===
    # Instead of adding ALL trainer args, only add what we need
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--gradient_clip_val", type=float, default=0.0)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--auto_scale_batch_size", action="store_true")
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--do_test", action="store_true", help="Run testing/validation only, skipping training")

    args = parser.parse_args()
    args.timestamp = get_timestamp()
    args.git_checksum = get_checksum()
    return args


def main():
    import warnings
    # Suppress known warnings that don't affect training
    # 1. FutureWarning about torch.cuda.amp.autocast (from PyTorch Lightning internal)
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")
    # 2. UserWarning about lr_scheduler.step() order (PyTorch Lightning handles this correctly)
    warnings.filterwarnings("ignore", category=UserWarning, message=".*lr_scheduler.step.*")
    
    pl.seed_everything(76)
    
    # NOTE: We disable anomaly detection in production for performance
    # The encoder is frozen to prevent inplace operation errors, so anomaly detection
    # is not necessary. Uncomment the line below if you need to debug other issues.
    # import torch
    # torch.autograd.set_detect_anomaly(True)

    args = parse_args()
    args.num_training_instances = get_num_training_instances(args)
    
    # Ensure gradient_checkpointing is False when encoder is frozen
    # (gradient_checkpointing is not needed when encoder is frozen)
    if hasattr(args, 'gradient_checkpointing') and args.gradient_checkpointing:
        print("⚠️  Warning: gradient_checkpointing is enabled but encoder will be frozen.")
        print("   gradient_checkpointing will be ignored (encoder doesn't compute gradients).")

    # === NEW: instantiate extended model ===
    if args.starting_checkpoint is not None:
        print(f"Loading from checkpoint: {args.starting_checkpoint}")
        print("Note: Initializing new model and loading compatible weights only")
        
        # Create model with new architecture
        model = MultiVerSPICOModel(args)
        
        # Load checkpoint
        import torch
        checkpoint = torch.load(args.starting_checkpoint, map_location="cpu")
        state_dict = checkpoint['state_dict']
        
        # Filter out incompatible keys (classifiers with different dimensions)
        incompatible_keys = []
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        
        for key, value in state_dict.items():
            if key in model_state_dict:
                if value.shape == model_state_dict[key].shape:
                    filtered_state_dict[key] = value
                else:
                    incompatible_keys.append(f"{key}: {value.shape} -> {model_state_dict[key].shape}")
            # Skip keys not in new model (will use new random initialization)
        
        # Load the filtered state dict
        model.load_state_dict(filtered_state_dict, strict=False)
        
        print(f"✓ Loaded {len(filtered_state_dict)}/{len(state_dict)} weights from checkpoint")
        if incompatible_keys:
            print(f"  Skipped {len(incompatible_keys)} incompatible layers (will be trained from scratch):")
            for key_info in incompatible_keys[:5]:  # Show first 5
                print(f"    - {key_info}")
    else:
        model = MultiVerSPICOModel(args)


    data_module = dm.ConcatDataModulePICO(args)

    save_dir, name, version, checkpoint_dir = get_folder_names(args)
    
    # === UPDATED: WandB Logger configuration ===
    # Use WandB as primary logger for experiment tracking
    wandb_run_name = args.wandb_name if args.wandb_name else args.experiment_name
    wandb_logger = pl_loggers.WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=wandb_run_name,
        tags=args.wandb_tags,
        save_dir=save_dir,
        version=version,
        log_model=False,  # Don't log model checkpoints to WandB (too large)
    )
    
    # Also keep CSV logger for local backup
    csv_logger = pl_loggers.CSVLogger(save_dir=save_dir, name=name, version=version)
    loggers = [wandb_logger, csv_logger]

    # Monitor validation metrics (aligned with eval_multivers.py)
    # We use 'valid_f1' which is computed exactly as in eval_multivers.py
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor="valid_f1",  # Primary metric for selection (aligned with eval_multivers.py)
        mode="max", 
        save_top_k=3,        # Keep top 3 best models
        save_last=True, 
        dirpath=checkpoint_dir,
        filename="{epoch}-{valid_f1:.4f}-{valid_precision:.4f}-{valid_recall:.4f}"
    )
    
    # Early Stopping to prevent overfitting
    early_stopping = callbacks.EarlyStopping(
        monitor="valid_f1",  # Use valid_f1 (aligned with eval_multivers.py)
        min_delta=0.001,
        patience=3,  # Stop if no improvement for 3 validation checks (more aggressive)
        verbose=True,
        mode="max"
    )

    lr_callback = callbacks.LearningRateMonitor(logging_interval="step")
    gpu_callback = callbacks.GPUStatsMonitor()
    
    # Include EarlyStopping in callbacks
    trainer_callbacks = [checkpoint_callback, early_stopping, lr_callback, gpu_callback]

    if args.accelerator == "ddp":
        plugins = DDPPlugin(find_unused_parameters=True)
    else:
        plugins = None

    # Create trainer with correct parameters
    trainer = pl.Trainer(
        gpus=args.gpus,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=trainer_callbacks, 
        logger=loggers, 
        plugins=plugins,
        fast_dev_run=args.fast_dev_run,
        progress_bar_refresh_rate=50,  # Control TQDM update frequency here
        num_sanity_val_steps=2,        # Quick sanity check (2 batches) before training
    )

    if args.auto_scale_batch_size:
        print("Scaling batch size.")
        trainer.tune(model, datamodule=data_module)
    elif args.do_test:
        print("Running validation/testing only...")
        data_module.setup()
        # Fallback: If test set is empty (e.g. original SciFact), use validation set for zero-shot eval
        if data_module.folds.get("test") is None:
            print("⚠️ Test set not found. Using VALIDATION set for zero-shot evaluation.")
            trainer.test(model, test_dataloaders=data_module.val_dataloader())
        else:
            trainer.test(model, datamodule=data_module)
    else:
        # === Removed Explicit Zero-shot Validation to save time ===
        trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
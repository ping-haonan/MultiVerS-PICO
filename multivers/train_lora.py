"""
Train MultiVerS with LoRA adapters.

This script supports:
1. model_lora.py - LoRA only (no PICO)
2. model_pico_lora.py - LoRA + PICO

Use --model_type to select.
"""

import datetime
from pathlib import Path
import subprocess

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks
from pytorch_lightning.plugins import DDPPlugin
import argparse

# Data modules
from . import data_train_pico as dm_pico
from .data_train import ConcatDataModule as dm_base

# Models
from .model_lora import MultiVerSLoRAModel
from .model_pico_lora import MultiVerSPICOLoRAModel


def get_timestamp():
    dt = datetime.datetime.now()
    return dt.strftime("%Y-%m-%d:%H:%M:%S")


def get_checksum():
    try:
        p1 = subprocess.Popen(["git", "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE)
        stdout, _ = p1.communicate()
        return stdout.decode("utf-8").split("\n")[0]
    except:
        return "unknown"


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
    if args.model_type == "lora":
        data_module = dm_base(args)
    else:
        data_module = dm_pico.ConcatDataModulePICO(args)
    data_module.setup()
    return len(data_module.folds["train"])


def parse_args():
    parser = argparse.ArgumentParser(description="Train MultiVerS with LoRA adapters.")
    
    # Model type selection
    parser.add_argument("--model_type", type=str, default="pico_lora",
                        choices=["lora", "pico_lora"],
                        help="Model type: 'lora' (no PICO) or 'pico_lora' (with PICO)")
    
    # Training config
    parser.add_argument("--starting_checkpoint", type=str, default=None)
    parser.add_argument("--monitor", type=str, default="valid_f1")
    parser.add_argument("--result_dir", type=str, default="results/lora_logs")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--debug", action="store_true")

    # WandB
    parser.add_argument("--wandb_project", type=str, default="multivers-lora")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None)

    # Add model-specific args (LoRA model has all the args we need)
    parser = MultiVerSPICOLoRAModel.add_model_specific_args(parser)
    
    # Add data module args
    parser = dm_pico.ConcatDataModulePICO.add_model_specific_args(parser)
    
    # Trainer args
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--auto_scale_batch_size", action="store_true")
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    args = parser.parse_args()
    args.timestamp = get_timestamp()
    args.git_checksum = get_checksum()
    return args


def main():
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*lr_scheduler.step.*")
    
    pl.seed_everything(76)

    args = parse_args()
    args.num_training_instances = get_num_training_instances(args)
    
    print("=" * 60)
    print(f"Training MultiVerS with LoRA")
    print(f"Model type: {args.model_type}")
    print("=" * 60)

    # Select model class
    if args.model_type == "lora":
        ModelClass = MultiVerSLoRAModel
        DataModuleClass = dm_base
        print("Using: LoRA only (no PICO features)")
    else:
        ModelClass = MultiVerSPICOLoRAModel
        DataModuleClass = dm_pico.ConcatDataModulePICO
        print("Using: LoRA + PICO features")

    # Create model
    if args.starting_checkpoint is not None:
        print(f"Loading from checkpoint: {args.starting_checkpoint}")
        
        model = ModelClass(args)
        
        import torch
        checkpoint = torch.load(args.starting_checkpoint, map_location="cpu")
        state_dict = checkpoint['state_dict']
        
        # Filter compatible weights
        model_state_dict = model.state_dict()
        filtered_state_dict = {}
        skipped = []
        
        for key, value in state_dict.items():
            if key in model_state_dict:
                if value.shape == model_state_dict[key].shape:
                    filtered_state_dict[key] = value
                else:
                    skipped.append(key)
        
        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"✓ Loaded {len(filtered_state_dict)}/{len(state_dict)} weights")
        if skipped:
            print(f"  Skipped {len(skipped)} incompatible layers")
    else:
        model = ModelClass(args)

    # Create data module
    data_module = DataModuleClass(args)

    save_dir, name, version, checkpoint_dir = get_folder_names(args)
    
    # Loggers
    wandb_run_name = args.wandb_name if args.wandb_name else args.experiment_name
    wandb_logger = pl_loggers.WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=wandb_run_name,
        tags=args.wandb_tags,
        save_dir=save_dir,
        version=version,
        log_model=False,
    )
    csv_logger = pl_loggers.CSVLogger(save_dir=save_dir, name=name, version=version)
    loggers = [wandb_logger, csv_logger]

    # Callbacks
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor="valid_f1",
        mode="max", 
        save_top_k=3,
        save_last=True, 
        dirpath=checkpoint_dir,
        filename="{epoch}-{valid_f1:.4f}-{valid_precision:.4f}-{valid_recall:.4f}"
    )
    # Early Stopping to prevent overfitting
    early_stopping = callbacks.EarlyStopping(
        monitor="valid_f1",
        min_delta=0.001,
        patience=3,  # Stop if no improvement for 3 validation checks
        verbose=True,
        mode="max"
    )
    lr_callback = callbacks.LearningRateMonitor(logging_interval="step")
    gpu_callback = callbacks.GPUStatsMonitor()
    trainer_callbacks = [checkpoint_callback, early_stopping, lr_callback, gpu_callback]

    if args.accelerator == "ddp":
        plugins = DDPPlugin(find_unused_parameters=True)
    else:
        plugins = None

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
        progress_bar_refresh_rate=50,
        num_sanity_val_steps=2,        # Quick sanity check (2 batches) before training
    )

    if args.auto_scale_batch_size:
        trainer.tune(model, datamodule=data_module)
    elif args.do_test:
        print("Running validation/testing only...")
        data_module.setup()
        trainer.test(model, datamodule=data_module)
    else:
        trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
audio_train.py - TIGER Speech Separation Distributed Training Pipeline
Compatible with Single-Node and Multi-Node DDP (asynchronous node launches on CoreWeave / clusters).
Branch: adrien-dev compatible.
"""

import os
import sys
import argparse
import datetime
import json
import tempfile
import warnings

# Purge residual Slurm environment variables to avoid Lightning SlurmPlugin override
for slurm_var in [
    "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NODEID",
    "SLURM_NTASKS", "SLURM_JOB_ID", "SLURM_NNODES", "SLURM_STEP_ID"
]:
    os.environ.pop(slurm_var, None)

GLOBAL_RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
IS_GLOBAL_ZERO = (GLOBAL_RANK == 0)

warnings.filterwarnings("ignore")

import yaml
import torch
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.loggers.wandb import WandbLogger

import look2hear.datas
import look2hear.models
import look2hear.system
import look2hear.losses
import look2hear.metrics
import look2hear.utils
from look2hear.system import make_optimizer
from look2hear.utils import print_only, MyRichProgressBar, RichProgressBarTheme
from look2hear.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict

# WandB Helper (Non-blocking, Rank 0 Guarded)
def setup_wandb(is_global_zero: bool, force_offline: bool = False):
    """Safely initializes WandB credentials without interactive prompts on cluster nodes."""
    if not is_global_zero:
        return
    if force_offline or os.environ.get("WANDB_MODE") == "offline" or os.environ.get("WANDB_DISABLED") == "true":
        os.environ["WANDB_MODE"] = "offline"
        return

    try:
        import wandb
        if "WANDB_API_KEY" in os.environ and os.environ["WANDB_API_KEY"].strip():
            wandb.login(key=os.environ["WANDB_API_KEY"], relogin=False)
        elif os.path.exists(os.path.expanduser("~/.netrc")):
            wandb.login(relogin=False)
        else:
            print("[Rank 0] No WANDB_API_KEY or ~/.netrc credentials found. Running in offline mode.")
            os.environ["WANDB_MODE"] = "offline"
    except Exception as e:
        print(f"[Rank 0 WARN] WandB initialization warning: {e}. Switching to offline mode.")
        os.environ["WANDB_MODE"] = "offline"

parser = argparse.ArgumentParser(description="TIGER Distributed Training")
parser.add_argument(
    "--conf_dir",
    default="local/conf.yml",
    help="Full path to YAML configuration file",
)
parser.add_argument(
    "--num_nodes",
    type=int,
    default=None,
    help="Number of nodes",
)
parser.add_argument(
    "--devices",
    type=str,
    default=None,
    help="GPUs per node or 'auto'",
)
parser.add_argument(
    "--timeout_hours",
    type=float,
    default=2.0,
    help="DDP rendezvous timeout in hours (default: 2.0)",
)
parser.add_argument(
    "--sync_batchnorm",
    action="store_true",
    default=None,
    help="Enable Cross-Node SyncBatchNorm",
)
parser.add_argument(
    "--wandb_offline",
    action="store_true",
    help="Force WandB offline mode",
)
parser.add_argument(
    "--exp_dir",
    type=str,
    default=None,
    help="Override experiment directory",
)


def main(config, extra_cli_args):
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.cuda.set_device(LOCAL_RANK)

    num_nodes = extra_cli_args.num_nodes
    if num_nodes is None:
        if "NNODES" in os.environ:
            num_nodes = int(os.environ["NNODES"])
        elif WORLD_SIZE > 1 and LOCAL_WORLD_SIZE > 0:
            num_nodes = max(1, WORLD_SIZE // LOCAL_WORLD_SIZE)
        else:
            num_nodes = config.get("training", {}).get("num_nodes", 1)

    if extra_cli_args.devices:
        devices = int(extra_cli_args.devices) if extra_cli_args.devices.isdigit() else extra_cli_args.devices
    elif "LOCAL_WORLD_SIZE" in os.environ:
        devices = int(os.environ["LOCAL_WORLD_SIZE"])
    else:
        cfg_gpus = config.get("training", {}).get("gpus", "auto")
        if isinstance(cfg_gpus, list):
            devices = len(cfg_gpus)
        elif cfg_gpus == "auto":
            devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        else:
            devices = cfg_gpus

    # Resolve relative dataset directories robustly
    data_cfg = config.get("datamodule", {}).get("data_config", {})
    x_training_data_dir = "/users/eleves-a/2023/adrien.dubois/4A/project_ma1/X_training/DataPreProcess"
    tiger_data_dir = "/users/eleves-a/2023/adrien.dubois/4A/project_ma1/TIGER/DataPreProcess"
    for dir_key in ["train_dir", "valid_dir", "test_dir"]:
        if dir_key in data_cfg and isinstance(data_cfg[dir_key], str):
            p = data_cfg[dir_key]
            if not os.path.isabs(p):
                candidates = [
                    os.path.join(os.getcwd(), p),
                    os.path.join(x_training_data_dir, p.replace("DataPreProcess/", "")),
                    os.path.join(tiger_data_dir, p.replace("DataPreProcess/", "")),
                ]
                for cand in candidates:
                    if os.path.exists(cand):
                        data_cfg[dir_key] = os.path.abspath(cand)
                        break

    print_only("Instantiating datamodule <{}>".format(config["datamodule"]["data_name"]))
    datamodule: object = getattr(look2hear.datas, config["datamodule"]["data_name"])(
        **config["datamodule"]["data_config"]
    )
    datamodule.setup()
    train_loader, val_loader, test_loader = datamodule.make_loader

    print_only("Instantiating AudioNet <{}>".format(config["audionet"]["audionet_name"]))
    model = getattr(look2hear.models, config["audionet"]["audionet_name"])(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )

    print_only("Instantiating Optimizer <{}>".format(config["optimizer"]["optim_name"]))
    optimizer = make_optimizer(model.parameters(), **config["optimizer"])

    scheduler = None
    if config.get("scheduler", {}).get("sche_name"):
        print_only("Instantiating Scheduler <{}>".format(config["scheduler"]["sche_name"]))
        if config["scheduler"]["sche_name"] != "DPTNetScheduler":
            scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"]["sche_name"])(
                optimizer=optimizer, **config["scheduler"]["sche_config"]
            )
        else:
            scheduler = {
                "scheduler": getattr(look2hear.system.schedulers, config["scheduler"]["sche_name"])(
                    optimizer,
                    len(train_loader) // config["datamodule"]["data_config"]["batch_size"],
                    64,
                ),
                "interval": "step",
            }

    # Guarded directory and config creation on Rank 0
    if extra_cli_args.exp_dir:
        exp_dir = extra_cli_args.exp_dir
    else:
        exp_dir = os.path.join(
            os.getcwd(), "Experiments", "checkpoint", config["exp"]["exp_name"]
        )
    config["main_args"]["exp_dir"] = exp_dir

    if IS_GLOBAL_ZERO:
        os.makedirs(exp_dir, exist_ok=True)
        conf_path = os.path.join(exp_dir, "conf.yml")
        with open(conf_path, "w") as outfile:
            yaml.safe_dump(config, outfile)

    print_only(
        "Instantiating Loss, Train <{}>, Val <{}>".format(
            config["loss"]["train"]["sdr_type"], config["loss"]["val"]["sdr_type"]
        )
    )
    loss_func = {
        "train": getattr(look2hear.losses, config["loss"]["train"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["train"]["sdr_type"]),
            **config["loss"]["train"]["config"],
        ),
        "val": getattr(look2hear.losses, config["loss"]["val"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["val"]["sdr_type"]),
            **config["loss"]["val"]["config"],
        ),
    }

    print_only("Instantiating System <{}>".format(config["training"]["system"]))
    system = getattr(look2hear.system, config["training"]["system"])(
        audio_model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        config=config,
    )

    callbacks = []
    checkpoint = ModelCheckpoint(
        dirpath=exp_dir,
        filename="{epoch}",
        monitor="val_loss/dataloader_idx_0",
        mode="min",
        save_top_k=config["training"].get("save_top_k", 2),
        verbose=IS_GLOBAL_ZERO,
        save_last=True,
    )
    callbacks.append(checkpoint)

    if config["training"].get("early_stop"):
        print_only("Instantiating EarlyStopping")
        callbacks.append(EarlyStopping(**config["training"]["early_stop"]))
    callbacks.append(MyRichProgressBar(theme=RichProgressBarTheme()))

    # Logger Setup (Rank 0 Guarded, ephemeral cache in /tmp to prevent disk bloat)
    setup_wandb(IS_GLOBAL_ZERO, force_offline=extra_cli_args.wandb_offline)

    run_id = config["exp"]["exp_name"]
    wandb_cache_dir = os.environ.get("WANDB_DIR", os.path.join(tempfile.gettempdir(), "tiger_wandb"))
    if IS_GLOBAL_ZERO:
        os.makedirs(wandb_cache_dir, exist_ok=True)

    offline_mode = (
        extra_cli_args.wandb_offline
        or os.environ.get("WANDB_MODE") == "offline"
    )
    comet_logger = WandbLogger(
        name=run_id,
        id=run_id,
        resume="allow",
        save_dir=wandb_cache_dir,
        project=os.environ.get("WANDB_PROJECT", "Real-work-dataset"),
        offline=offline_mode,
        log_model=False,
    )

    timeout_delta = datetime.timedelta(hours=extra_cli_args.timeout_hours)
    ddp_strategy = DDPStrategy(
        cluster_environment=TorchElasticEnvironment(),
        find_unused_parameters=True,
        process_group_backend="nccl",
        timeout=timeout_delta,
    )

    if extra_cli_args.sync_batchnorm is not None:
        sync_batchnorm = extra_cli_args.sync_batchnorm
    else:
        sync_batchnorm = config["training"].get("sync_batchnorm", False)

    precision = config["training"].get("precision", "bf16-mixed" if torch.cuda.is_available() else "32")
    limit_val_batches = config["training"].get("limit_val_batches", 0.2)

    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        num_nodes=num_nodes,
        devices=devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=ddp_strategy,
        limit_train_batches=1.0,
        limit_val_batches=limit_val_batches,
        gradient_clip_val=config["training"].get("gradient_clip_val", 5.0),
        logger=comet_logger,
        sync_batchnorm=sync_batchnorm,
        precision=precision,
    )

    last_ckpt = os.path.join(exp_dir, "last.ckpt")
    ckpt_to_resume = last_ckpt if os.path.isfile(last_ckpt) else None
    if ckpt_to_resume:
        print_only(f"[Resume] Found checkpoint, resuming training from: {ckpt_to_resume}")

    trainer.fit(system, ckpt_path=ckpt_to_resume)
    print_only("Finished Training")

    if trainer.is_global_zero:
        if checkpoint.best_k_models:
            best_k = {
                k: (v.item() if hasattr(v, "item") else v)
                for k, v in checkpoint.best_k_models.items()
            }
            with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
                json.dump(best_k, f, indent=0)

        if checkpoint.best_model_path and os.path.exists(checkpoint.best_model_path):
            state_dict = torch.load(checkpoint.best_model_path, map_location="cpu")
            if "state_dict" in state_dict:
                system.load_state_dict(state_dict=state_dict["state_dict"])
            system.cpu()
            to_save = system.audio_model.serialize()
            torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    cli_args, remaining_argv = parser.parse_known_args()

    with open(cli_args.conf_dir) as f:
        def_conf = yaml.safe_load(f)

    dynamic_parser = prepare_parser_from_dict(def_conf, parser=argparse.ArgumentParser())
    arg_dic, _ = parse_args_as_dict(dynamic_parser, return_plain_args=True, args=remaining_argv)

    for k, v in def_conf.items():
        if k not in arg_dic:
            arg_dic[k] = v

    main(arg_dic, cli_args)

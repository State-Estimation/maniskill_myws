#!/usr/bin/env python
"""
使用 DeepSpeed 训练 Pi0/Pi0.5 模型的脚本，支持 LoRA 微调和多 GPU 训练。

使用方法：
单卡训练:
  python scripts/pi0/train_deepspeed.py \\
    --config pi0_libero \\
    --exp_name test_deepspeed \\
    --deepspeed_config scripts/pi0/deepspeed_config.json

多卡训练 (DeepSpeed launcher):
  deepspeed --num_gpus=2 scripts/pi0/train_deepspeed.py \\
    --config pi0_libero \\
    --exp_name test_deepspeed \\
    --deepspeed_config scripts/pi0/deepspeed_config.json

多卡训练 (torchrun):
  torchrun --nproc_per_node=2 scripts/pi0/train_deepspeed.py \\
    --config pi0_libero \\
    --exp_name test_deepspeed \\
    --deepspeed_config scripts/pi0/deepspeed_config.json
"""

import argparse
import dataclasses
import gc
import json
import logging
import os
import pathlib
import platform
import shutil
import time

import deepspeed
import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import tqdm
import wandb

# 确保能够导入 openpi
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parents[4] / "openpi" / "src"))

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


def init_logging():
    """初始化日志格式"""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """初始化 wandb 日志"""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_distributed():
    """设置分布式训练环境"""
    # DeepSpeed 会自动初始化分布式环境
    if not dist.is_initialized():
        # 如果 DeepSpeed 还没有初始化，我们手动初始化
        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
        else:
            rank = 0
            world_size = 1
            local_rank = 0
            
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    
    is_main = rank == 0
    return world_size, rank, local_rank, device, is_main


def set_seed(seed: int, rank: int):
    """设置随机种子"""
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def build_datasets(config: _config.TrainConfig):
    """构建数据集"""
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def log_memory_usage(device, step, phase="unknown"):
    """记录内存使用情况"""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = (torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)) / 1e9

    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - "
        f"allocated: {memory_allocated:.2f}GB, "
        f"reserved: {memory_reserved:.2f}GB, "
        f"free: {memory_free:.2f}GB, "
        f"peak_allocated: {max_memory_allocated:.2f}GB, "
        f"peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def save_checkpoint(model_engine, global_step, config, is_main, data_config):
    """保存检查点"""
    if not is_main:
        return

    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps:
        ckpt_dir = config.checkpoint_dir / f"{global_step}"
        
        # DeepSpeed 自己处理检查点保存
        # 但我们也保存一些额外的元数据
        if not ckpt_dir.exists():
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用 DeepSpeed 的检查点保存
        model_engine.save_checkpoint(str(ckpt_dir))
        
        # 保存训练元数据
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, ckpt_dir / "metadata.pt")

        # 保存归一化统计
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        logging.info(f"Saved checkpoint at step {global_step} -> {ckpt_dir}")

        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model_engine, checkpoint_dir):
    """加载最新的检查点"""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    logging.info(f"Loading checkpoint from {ckpt_dir}")
    _, client_state = model_engine.load_checkpoint(str(ckpt_dir))
    
    global_step = latest_step
    if client_state and "global_step" in client_state:
        global_step = client_state["global_step"]
    
    logging.info(f"Successfully loaded checkpoint from step {latest_step}")
    return global_step


def create_lr_schedule(config: _config.TrainConfig):
    """创建学习率调度器"""
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    def lr_schedule(step: int):
        if step < warmup_steps:
            # 预热阶段
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # 余弦衰减
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    return lr_schedule


def train_loop(config: _config.TrainConfig, deepspeed_config_path: str):
    """主训练循环"""
    world_size, rank, local_rank, device, is_main = setup_distributed()
    set_seed(config.seed, rank)

    # 初始化检查点目录
    resuming = False
    if config.resume:
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            checkpoint_steps = [
                int(d.name)
                for d in exp_checkpoint_dir.iterdir()
                if d.is_dir() and d.name.isdigit()
            ]
            if checkpoint_steps:
                resuming = True
                logging.info(f"Resuming from checkpoint directory: {exp_checkpoint_dir}")
            else:
                raise FileNotFoundError(f"No checkpoints found in {exp_checkpoint_dir}")
        else:
            raise FileNotFoundError(f"Checkpoint directory {exp_checkpoint_dir} does not exist")
    elif config.overwrite and config.checkpoint_dir.exists():
        if is_main:
            shutil.rmtree(config.checkpoint_dir)
            logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    if not resuming and is_main:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created checkpoint directory: {config.checkpoint_dir}")

    # 初始化 wandb
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # 构建数据加载器
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} "
        f"(total batch size: {config.batch_size} across {world_size} GPUs)"
    )

    loader, data_config = build_datasets(config)

    # 记录样本图像到 wandb
    if is_main and config.wandb_enabled and not resuming:
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        images_to_log = []
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            img_concatenated = torch.cat(
                [img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], 
                axis=1
            )
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)
        
        del sample_batch, observation, actions, images_to_log, img_concatenated, sample_data_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 构建模型
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg)

    # 启用梯度检查点以节省内存
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")

    # 记录初始内存使用
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # 加载预训练权重（如果指定）
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(model, model_path)
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # 加载 DeepSpeed 配置
    with open(deepspeed_config_path, 'r') as f:
        ds_config = json.load(f)
    
    # 从训练配置更新 DeepSpeed 配置
    ds_config["train_batch_size"] = config.batch_size
    ds_config["train_micro_batch_size_per_gpu"] = effective_batch_size
    ds_config["gradient_accumulation_steps"] = 1
    
    # 更新优化器参数
    if "optimizer" in ds_config:
        ds_config["optimizer"]["params"]["lr"] = config.lr_schedule.peak_lr
        ds_config["optimizer"]["params"]["betas"] = [config.optimizer.b1, config.optimizer.b2]
        ds_config["optimizer"]["params"]["eps"] = config.optimizer.eps
        ds_config["optimizer"]["params"]["weight_decay"] = config.optimizer.weight_decay

    # 初始化 DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    # 加载检查点（如果恢复训练）
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model_engine, config.checkpoint_dir)
        logging.info(f"Resumed training from step {global_step}")

    # 创建学习率调度器
    lr_schedule = create_lr_schedule(config)

    # 训练循环
    model_engine.train()
    start_time = time.time()
    infos = []
    
    if is_main:
        logging.info(f"Running on: {platform.node()} | world_size={world_size}")
        logging.info(f"Training config: batch_size={config.batch_size}, num_train_steps={config.num_train_steps}")
        logging.info(f"DeepSpeed config: {deepspeed_config_path}")
        logging.info(f"Training precision: {model_cfg.dtype}")

    pbar = tqdm.tqdm(
        total=config.num_train_steps, 
        initial=global_step, 
        desc="Training", 
        disable=not is_main
    ) if is_main else None

    while global_step < config.num_train_steps:
        for observation, actions in loader:
            if global_step >= config.num_train_steps:
                break

            # 将数据移到设备
            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(torch.float32).to(device)

            # 更新学习率
            current_lr = lr_schedule(global_step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            # 前向传播
            losses = model_engine(observation, actions)
            if isinstance(losses, (list, tuple)):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)
            
            loss = losses.mean()

            # 反向传播（DeepSpeed 处理）
            model_engine.backward(loss)
            
            # 记录内存使用
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # 优化器步骤（DeepSpeed 处理梯度裁剪）
            model_engine.step()

            # 收集统计信息
            if is_main:
                infos.append({
                    "loss": loss.item(),
                    "learning_rate": current_lr,
                })

            # 记录日志
            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                if config.wandb_enabled:
                    wandb.log({
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }, step=global_step)

                start_time = time.time()
                infos = []

            global_step += 1
            
            # 保存检查点
            save_checkpoint(model_engine, global_step, config, is_main, data_config)

            # 更新进度条
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{current_lr:.2e}",
                    "step": global_step
                })

    # 关闭进度条
    if pbar is not None:
        pbar.close()

    # 完成 wandb 运行
    if is_main and config.wandb_enabled:
        wandb.finish()


def main():
    """主函数"""
    init_logging()
    
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed_config", type=str, required=True, help="DeepSpeed 配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank (自动设置)")
    
    args, remaining_args = parser.parse_known_args()
    
    # 将剩余参数传递给 openpi 的配置解析器
    sys.argv = [sys.argv[0]] + remaining_args
    config = _config.cli()
    
    train_loop(config, args.deepspeed_config)


if __name__ == "__main__":
    main()

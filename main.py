import argparse
import math
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import RandAugment
from tqdm import tqdm

from Camelyon import Camelyon16
from ConDo.loader import DistributedBalancedSampler, TwoCropsTransform
from ConDo.model import ConDo


def parse_args():
    parser = argparse.ArgumentParser(description="ConDo Training")
    parser.add_argument("--batch-size", default=256, type=int, help="batch size")
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
    parser.add_argument("--lr", default=0.03, type=float, help="learning rate")
    parser.add_argument("--wd", default=1e-4, type=float, help="weight decay")
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use")
    parser.add_argument(
        "--multiprocessing-distributed",
        default=True,
        action="store_true",
        help="Use multi-processing distributed training",
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints", type=str, help="checkpoint directory"
    )
    parser.add_argument(
        "--use-wandb", default=False, action="store_true", help="Use wandb for logging"
    )

    # Model architecture and parameters
    parser.add_argument(
        "--arch",
        default="resnet50",
        type=str,
        help="model architecture (resnet18, resnet50, etc.)",
    )
    parser.add_argument(
        "--d-cat", default=2, type=int, help="dimension of category embedding"
    )
    parser.add_argument(
        "--d-inst", default=256, type=int, help="dimension of instance embedding"
    )
    parser.add_argument("--mlp", action="store_true", help="use MLP head")
    parser.add_argument(
        "--no-mlp", dest="mlp", action="store_false", help="do not use MLP head"
    )
    parser.add_argument(
        "--temperature",
        default=0.3,
        type=float,
        help="temperature for contrastive loss",
    )
    parser.add_argument(
        "--class-weight", default=5.0, type=float, help="weight for classification loss"
    )
    parser.add_argument(
        "--logit-temp", default=0.1, type=float, help="temperature for logits"
    )
    parser.add_argument(
        "--resume-path", default=None, type=str, help="path to resume training from"
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=str,
        help="path to dataset root directory",
    )
    parser.add_argument(
        "--label-file",
        required=True,
        type=str,
        help="path to label file",
    )
    return parser.parse_args()


def setup(rank, world_size, args):
    """
    Setup distributed training environment
    """
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        rank=rank,
        world_size=world_size,
    )


def cleanup():
    """
    Clean up distributed training environment
    """
    dist.destroy_process_group()


def save_checkpoint(state, checkpoint_dir, epoch, is_best=False):
    """
    Save checkpoint to disk.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save epoch checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth.tar")
    torch.save(state, checkpoint_path)

    # Save best model if it's the best
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state["state_dict"], best_path)
        return best_path

    return checkpoint_path


def main_worker(rank, world_size, args):
    """
    Main training function for distributed training.
    """
    setup(rank, world_size, args)
    device = torch.device(f"cuda:{rank}")
    # torch.autograd.set_detect_anomaly(True)

    if rank == 0 and args.use_wandb:
        wandb.login(key="APIKEY")
        wandb.init(
            project="sdc-contrastive-RUMC",
            config=vars(args),
        )

    mean = [0.6377, 0.4562, 0.6402]
    std = [0.2403, 0.2536, 0.1970]

    # Data Augmentations
    base_transform = transforms.Compose(
        [
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform = TwoCropsTransform(base_transform)

    # root_path = "/project/luofeng/luofeng/Camelyon16/camelyon_dataset/training"
    # train_text = ("/project/luofeng/luofeng/Camelyon16/camelyon_dataset/training/RUMC_label.txt")
    # root_path = "/project/luofeng/luofeng/Camelyon16/RUMC_dataset"
    # train_text = "/project/luofeng/luofeng/Camelyon16/RUMC_dataset/RUMC_train_label.txt"

    root_path = args.data_root
    train_text = args.label_file

    trainset = Camelyon16(root_path, train_text, transform=transform)
    train_sampler = DistributedBalancedSampler(trainset)
    # train_sampler = DistributedSampler(trainset, sampler=sampler) if args.multiprocessing_distributed else None

    batch_per_gpu = args.batch_size // world_size

    train_loader = DataLoader(
        trainset,
        batch_size=batch_per_gpu,
        shuffle=(train_sampler is None),
        num_workers=6,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    encoder = models.__dict__[args.arch](weights=None, num_classes=0)
    dim_in = 512 if "18" in args.arch or "34" in args.arch else 2048

    if rank == 0:
        print(encoder)

    # Initialize model
    model = ConDo(
        encoder,
        dim_in=dim_in,
        d_cat=args.d_cat,
        d_inst=args.d_inst,
        mlp=args.mlp,
        temperature=args.temperature,
        class_weight=args.class_weight,
        eps=1e-8,
        logit_temp=args.logit_temp,
    ).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    # Initialize optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
    )

    warmup_epochs = 20
    warmup = LinearLR(optimizer, start_factor=0.05, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )

    if rank == 0:
        wandb.watch(model, log="all")

    # --- Training Loop ---
    best_loss = float("inf")
    start_epoch = 0
    epoch_losses = []  # List to store losses for each epoch

    if args.resume_path and os.path.exists(args.resume_path):
        if rank == 0:
            print(f"Loading checkpoint from {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location=f"cuda:{rank}")
        model.module.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}")

    scaler = GradScaler()

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        batch_loss = 0.0

        # Use tqdm only on rank 0
        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        else:
            pbar = train_loader

        for batch_idx, (images, labels) in enumerate(pbar):
            # images is a list of two tensors from TwoCropsTransform
            images0, images1 = (
                images[0].to(device, non_blocking=True),
                images[1].to(device, non_blocking=True),
            )
            labels = labels.to(device, non_blocking=True)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            with autocast():
                loss = model(images0, images1, labels)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Update weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Update progress bar
            if rank == 0:
                pbar.set_postfix({"Loss": loss.item()})

            # Accumulate loss
            total_loss += loss.item()
            batch_loss += loss.item()

        # Compute average loss
        total_loss_tensor = torch.tensor(total_loss, device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = total_loss_tensor.item() / (len(train_loader) * world_size)
        epoch_losses.append(avg_loss)  # Store the loss for this epoch

        scheduler.step()

        if rank == 0:
            print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")
            if args.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": avg_loss,
                        "train_loss_std": torch.tensor(epoch_losses).std().item()
                        if epoch_losses
                        else 0.0,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss

            # Prepare checkpoint state
            checkpoint_state = {
                "epoch": epoch + 1,
                "state_dict": model.module.state_dict(),  # Use model.module for DDP
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
                "config": vars(args),
            }

            # Save epoch checkpoint
            checkpoint_path = save_checkpoint(
                checkpoint_state, args.checkpoint_dir, epoch, is_best=is_best
            )

            # Save best model
            if is_best:
                print(f"New best model saved with loss: {best_loss:.4f}")
                if wandb.run is not None:
                    wandb.save(os.path.join(args.checkpoint_dir, "best_model.pth"))

            print(f"Checkpoint saved: {checkpoint_path}")
    if rank == 0:
        final_path = os.path.join(args.checkpoint_dir, "final_model.pth")
        torch.save(model.module.state_dict(), final_path)
        if wandb.run is not None:
            wandb.save(final_path)
        print("Training completed. Final model saved.")

        # Finish the wandb run
        if args.use_wandb:
            wandb.finish()

    cleanup()


def main():
    """
    Main function for training.
    """
    args = parse_args()

    world_size = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        print(f"Using {world_size} GPUs for training")
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))
    else:
        main_worker(0, world_size, args)


if __name__ == "__main__":
    main()

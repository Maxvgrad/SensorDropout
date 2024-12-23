import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import wandb
from engine import train_one_epoch, evaluate
from models import build_dataset, build_model
from models.ade_post_processor import PostProcessTrajectory
from models.set_criterion import build_criterion
from util.misc import collate_fn, is_main_process, get_sha, get_rank


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sensor dropout")

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=14, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--num_objects', type=int, default=4, help='Number of objects (TODO: refactor it)')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--model', type=str, default='perceiver', help='Model type')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--train_dataset_fraction', type=float, default=1, help='Train dataset fraction')

    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training')
    parser.add_argument('--num_frames', type=int, default=8, help='Number of frames')
    parser.add_argument('--frame_dropout_probs', nargs='+', type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], help='List of frame dropout probabilities')
    parser.add_argument('--sampler_steps', nargs='+', type=int, default=[2, 4, 6, 8, 10], help='Sampler steps')
    parser.add_argument('--frame_dropout_pattern', type=str, default='00001111', help='Frame dropout pattern')
    parser.add_argument('--output_dir', type=str, default=None, required=True, help='Output directory')
    parser.add_argument('--dataset', type=str, default='moving-mnist-2digit-tr', help='Dataset name')

    parser.add_argument('--train_val_split_ratio', type=float, default=0.8, help='Train-validation split ratio')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g., cpu or cuda)')

    # wandb
    parser.add_argument('--wandb_project', type=str, default='sensor-dropout', help='Wandb project')
    parser.add_argument('--wandb_id', type=str, default=None, help='Wandb ID resume training')

    # Perceiver model specific arguments
    parser.add_argument('--num_freq_bands', type=int, default=6, help='Number of frequency bands for Fourier encoding')
    parser.add_argument('--max_freq', type=int, default=10, help='Maximum frequency for Fourier encoding')
    parser.add_argument('--enc_layers', type=int, default=1, help='Number of layers in Perceiver encoder')

    parser.add_argument('--hidden_dim', type=int, default=128, help='Latent dimension size')
    parser.add_argument('--enc_nheads_cross', type=int, default=1, help='Number of cross-attention heads')
    parser.add_argument('--nheads', type=int, default=1, help='Number of latent self-attention heads')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--self_per_cross_attn', type=int, default=1, help='Number of self-attention blocks per cross-attention block')

    # LSTM model specific arguments
    parser.add_argument('--lstm_hidden_size', type=int, default=128, help='Hidden size of LSTM')

    args = parser.parse_args()
    return args

def main(args):

    print(args)

    print("git:\n  {}\n".format(get_sha()))

    if is_main_process():
        print(f'Init wandb in process rank: {get_rank()}')
        wandb.init(**get_wandb_init_config(args))

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    # Paths and directories
    output_dir = Path(args.output_dir)
    if args.output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        yaml.dump(
            vars(args),
            open(output_dir / 'config.yaml', 'w'), allow_unicode=True)

    # Dataset and dataloaders
    dataset_train = build_dataset('train', args)
    dataset_val = build_dataset('val', args)
    dataset_val_blind = build_dataset('val', args, frame_dropout_pattern=args.frame_dropout_pattern)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_val_blind = torch.utils.data.SequentialSampler(dataset_val_blind)

    dataloader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size, collate_fn=collate_fn)
    dataloader_val = DataLoader(dataset_val, sampler=sampler_val, batch_size=args.batch_size, collate_fn=collate_fn)
    dataloader_val_blind = DataLoader(dataset_val_blind, sampler=sampler_val_blind, batch_size=args.batch_size, collate_fn=collate_fn)

    # Model, criterion, optimizer, and scheduler
    model = build_model(args)
    postprocessors = {'trajectory': PostProcessTrajectory()}

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad], "lr": args.learning_rate},
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = build_criterion(args.num_classes)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)

    # Resume from checkpoint
    if args.resume:
        checkpoint_path = output_dir / args.resume
        print(f'Resuming from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
    else:
        start_epoch = 0
        best_val_loss = float('inf')

    model = model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    criterion = criterion.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters:', n_parameters)

    # Training loop
    print("Start training")
    start_time = time.time()

    dataset_train.set_epoch(start_epoch)
    dataset_val.set_epoch(start_epoch)
    dataset_val_blind.set_epoch(start_epoch)

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(model, dataloader_train, optimizer, criterion, epoch, device)

        test_stats = evaluate(model, dataloader_val, criterion, postprocessors, epoch, device)
        blind_stats = evaluate(model, dataloader_val_blind, criterion, postprocessors, epoch, device)

        lr_scheduler.step()

        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch:02}.pth"
        val_loss = test_stats.get("loss", float("inf"))

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_default_{k}': v for k, v in test_stats.items()},
            **{f'test_blind_{k}': v for k, v in blind_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters,
            'frame_dropout_prob': dataset_train.frame_dropout_prob
        }

        if output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if is_main_process():
            print(json.dumps(log_stats, indent=2))
            wandb.log(log_stats, step=epoch)

        if val_loss < best_val_loss or epoch % 2 == 0 or epoch + 1 == args.epochs:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            best_val_loss = val_loss
            print(f"Checkpoint saved at epoch {epoch} with val loss {val_loss:.4f}")

        dataset_train.step_epoch()
        dataset_val.step_epoch()
        dataset_val_blind.step_epoch()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def get_wandb_init_config(args):
    result = {
        'project': args.wandb_project
    }

    if args.wandb_id:
        result['id'] = args.wandb_id
        result['resume'] = 'must'

    return result


if __name__ == '__main__':
    args = parse_args()
    main(args)

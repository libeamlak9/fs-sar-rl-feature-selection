#!/usr/bin/env python3
"""
Training script for Few-Shot Learning with RL-Based Feature Selection.

Usage:
    python train.py --config configs/default_config.yaml
    python train.py --data MSTAR_10_Classes --backbone resnet50 --n-way 5 --n-shot 5
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from main import main as train_main


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def override_config(config, args):
    """Override config with command line arguments."""
    if args.data:
        config['data']['name'] = args.data
    if args.backbone:
        config['model']['backbone'] = args.backbone
    if args.n_way:
        config['training']['n_way'] = args.n_way
    if args.n_shot:
        config['training']['n_shot'] = args.n_shot
    if args.n_query:
        config['training']['n_query'] = args.n_query
    if args.episodes:
        config['training']['n_episodes'] = args.episodes
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.output_dir:
        config['output']['save_dir'] = args.output_dir
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Train Few-Shot Learning model with RL-based feature selection'
    )
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, help='Dataset name')
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'resnet50', 'efficient_net'],
                       help='Backbone architecture')
    parser.add_argument('--n-way', type=int, help='Number of classes per episode')
    parser.add_argument('--n-shot', type=int, help='Number of support samples per class')
    parser.add_argument('--n-query', type=int, help='Number of query samples per class')
    parser.add_argument('--episodes', type=int, help='Number of training episodes')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--output-dir', type=str, help='Output directory for models and results')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Load and override config
    config = load_config(args.config)
    config = override_config(config, args)
    
    # Run training
    train_main(config)


if __name__ == '__main__':
    main()

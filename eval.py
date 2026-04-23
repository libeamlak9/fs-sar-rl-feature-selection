#!/usr/bin/env python3
"""
Evaluation script for Few-Shot Learning with RL-Based Feature Selection.

Usage:
    python eval.py --proto-network path/to/model.pth --feature-selector path/to/selector.pth
    python eval.py --proto-network model.pth --feature-selector selector.pth --n-way 3 --n-shot 1
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
from src.evaluation import load_and_evaluate, TaskSampler, my_backbone, my_fs_hs
from src.common_functions import ConfigDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Few-Shot Learning model'
    )
    parser.add_argument('--proto-network', type=str, required=True,
                       help='Path to proto network checkpoint')
    parser.add_argument('--feature-selector', type=str, required=True,
                       help='Path to feature selector checkpoint')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default='MSTAR_10_Classes',
                       help='Dataset name')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet18', 'resnet50', 'efficient_net'],
                       help='Backbone architecture')
    parser.add_argument('--n-way', type=int, default=3,
                       help='Number of classes per episode')
    parser.add_argument('--n-shot', type=int, default=1,
                       help='Number of support samples per class')
    parser.add_argument('--n-query', type=int, default=10,
                       help='Number of query samples per class')
    parser.add_argument('--n-tasks', type=int, default=100,
                       help='Number of evaluation tasks')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    print(f"Evaluating model:")
    print(f"  Proto Network: {args.proto_network}")
    print(f"  Feature Selector: {args.feature_selector}")
    print(f"  Dataset: {args.data}")
    print(f"  {args.n_way}-way {args.n_shot}-shot")
    
    # Run evaluation
    # Note: This is a simplified version - full implementation would use the config
    load_and_evaluate(
        args.proto_network,
        args.feature_selector,
        None,  # eval_loader would be created here
        args.backbone,
        256  # hidden size
    )


if __name__ == '__main__':
    main()

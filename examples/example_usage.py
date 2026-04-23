#!/usr/bin/env python3
"""
Example: Training and evaluating a few-shot learning model.

This script demonstrates the complete workflow from training to evaluation.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def example_training():
    """Example training configuration."""
    print("=" * 60)
    print("Example: Training a 5-way 5-shot model")
    print("=" * 60)
    
    command = """
    python train.py \\
        --data MSTAR_10_Classes \\
        --backbone resnet50 \\
        --n-way 5 \\
        --n-shot 5 \\
        --n-query 10 \\
        --episodes 1000 \\
        --lr 0.0001 \\
        --output-dir ./example_results
    """
    print(command)
    print()


def example_evaluation():
    """Example evaluation configuration."""
    print("=" * 60)
    print("Example: Evaluating a trained model")
    print("=" * 60)
    
    command = """
    python eval.py \\
        --proto-network example_results/proto_network.pth \\
        --feature-selector example_results/feature_selector.pth \\
        --n-way 3 \\
        --n-shot 1 \\
        --n-tasks 100
    """
    print(command)
    print()


def example_similar_class():
    """Example similar class analysis."""
    print("=" * 60)
    print("Example: Analyzing similar classes (T72 vs T62)")
    print("=" * 60)
    
    command = """
    python src/similar_class_analysis.py \\
        --proto-network example_results/proto_network.pth \\
        --feature-selector example_results/feature_selector.pth \\
        --similar-pair T72,T62 \\
        --n-way 2 \\
        --n-shot 5 \\
        --output-dir ./similar_analysis
    """
    print(command)
    print()


def main():
    """Run all examples."""
    print("\nFew-Shot Learning with RL-Based Feature Selection")
    print("Example Usage Scripts\n")
    
    example_training()
    example_evaluation()
    example_similar_class()
    
    print("=" * 60)
    print("For more information, see docs/USAGE.md")
    print("=" * 60)


if __name__ == '__main__':
    main()

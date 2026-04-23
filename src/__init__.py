"""
Few-Shot Learning with RL-Based Feature Selection

This package implements a few-shot learning framework with reinforcement learning
based feature selection for SAR image classification.
"""

__version__ = '1.0.0'
__author__ = 'L.B Weldemaryam, Qian-Ru Wei, Member, IEEE'

from .common_functions import (
    ConfigDataset,
    RLAgent,
    PrototypicalNetworks,
    evaluate,
    build_task_context,
)
from .backbone import (
    ModifiedResNet18,
    ModifiedResNet50,
    HookedFeatureExtractor,
)

__all__ = [
    'ConfigDataset',
    'RLAgent',
    'PrototypicalNetworks',
    'evaluate',
    'build_task_context',
    'ModifiedResNet18',
    'ModifiedResNet50',
    'HookedFeatureExtractor',
]

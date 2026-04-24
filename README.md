# Few-Shot Learning with RL-Based Feature Selection

[![DOI](https://zenodo.org/badge/1219248716.svg)](https://doi.org/10.5281/zenodo.19712420)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

This repository implements a few-shot learning framework with reinforcement learning-based feature selection for SAR (Synthetic Aperture Radar) image classification. The model uses an RL agent to dynamically select discriminative features, improving classification accuracy on challenging similar-class scenarios.

## Key Features

- **Reinforcement Learning Feature Selection**: RL agent learns to select optimal features for each task
- **Prototypical Networks**: Few-shot learning with metric-based classification
- **Similar Class Analysis**: Specialized evaluation for hard-to-distinguish classes (e.g., T72 vs T62)
- **Multiple Visualizations**: t-SNE, confusion matrices, distance analysis, and confidence plots
- **Flexible Backbone Support**: ResNet-18, ResNet-50, EfficientNet

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/few-shot-rl-feature-selection.git
cd few-shot-rl-feature-selection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

### MSTAR Dataset

1. Download the MSTAR dataset from [source]
2. Organize the data as follows:
```
MSTAR-10-Classes/
├── Train_classes/
│   ├── 2S1/
│   ├── BMP2/
│   ├── BRDM2/
│   ├── BTR60/
│   ├── BTR70/
│   ├── D7/
│   ├── T62/
│   ├── T72/
│   ├── ZIL131/
│   └── ZSU_23_4/
└── Evaluation_classes/
    └── (same structure)
```

3. Place the dataset in the project root or update the path in the configuration.

## Quick Start

### Training

```bash
# Train with default configuration
python train.py

# Train with custom settings
python train.py --data MSTAR_10_Classes \
                --backbone resnet50 \
                --n-way 5 \
                --n-shot 5 \
                --episodes 6000 \
                --output-dir ./results

# Train with custom config file
python train.py --config configs/custom_config.yaml
```

### Evaluation

```bash
# Evaluate a trained model
python eval.py --proto-network results/proto_network.pth \
               --feature-selector results/feature_selector.pth \
               --n-way 3 \
               --n-shot 1 \
               --n-tasks 100
```

### Similar Class Analysis

Analyze performance on hard-to-distinguish classes:

```bash
python src/similar_class_analysis.py \
    --proto-network results/proto_network.pth \
    --feature-selector results/feature_selector.pth \
    --similar-pair T72,T62 \
    --n-way 2 \
    --n-shot 5 \
    --output-dir ./similar_class_results
```

### Ablation Study

Compare performance with and without feature selection:

```bash
python src/ablation_visualization.py \
    --proto-network results/with_selection/proto_network.pth \
    --feature-selector results/with_selection/feature_selector.pth \
    --proto-network-without results/without_selection/proto_network.pth \
    --n-way 3 \
    --n-shot 1 \
    --output-dir ./ablation_results
```

## Configuration

The framework uses YAML configuration files. See `configs/default_config.yaml` for all available options:

```yaml
# Example configuration
data:
  name: 'MSTAR_10_Classes'

model:
  backbone: 'resnet50'
  feature_selector: 'rl'

training:
  n_way: 5
  n_shot: 5
  n_query: 10
  n_episodes: 6000
  learning_rate: 0.0001

evaluation:
  n_way: 3
  n_shot: 1
  n_query: 10
  n_tasks: 100
```

## Project Structure

```
.
├── configs/                  # Configuration files
│   └── default_config.yaml
├── src/                      # Source code
│   ├── __init__.py
│   ├── main.py              # Training script
│   ├── evaluation.py        # Evaluation script
│   ├── common_functions.py  # Core functions
│   ├── backbone.py          # Network architectures
│   ├── ablation_visualization.py
│   └── similar_class_analysis.py
├── examples/                 # Example usage
├── docs/                     # Documentation
├── train.py                  # Training entry point
├── eval.py                   # Evaluation entry point
├── requirements.txt
├── README.md
└── LICENSE
```

## Results

Our model achieves the following performance on MSTAR dataset:

| Setting | Accuracy |
|---------|----------|
| 3-way 5-shot | 98.53% |
| 3-way 2-shot | 91.90% |

## Visualizations

The framework generates comprehensive visualizations:

- **Confusion Matrix**: Classification performance per class
- **t-SNE Plots**: Feature space visualization in distance and probability spaces
- **Distance Analysis**: Direct prototype distance scatter plots
- **Confidence Distribution**: Prediction confidence histograms

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2026fewshot,
  title={Few-Shot Learning with RL-Based Feature Selection for SAR Image Classification},
  author={Your Name},
  journal={},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MSTAR dataset provided by https://www.kaggle.com/datasets/ravenchencn/mstar-10-classes
- Built with PyTorch and EasyFSL

## Contact

For questions or issues, please open an issue on GitHub or contact legendariyy98@gmail.com

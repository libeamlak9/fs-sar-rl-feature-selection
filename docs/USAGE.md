# Usage Guide

## Table of Contents

1. [Training](#training)
2. [Evaluation](#evaluation)
3. [Similar Class Analysis](#similar-class-analysis)
4. [Ablation Study](#ablation-study)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)

## Training

### Basic Training

Train a model with default settings:

```bash
python train.py
```

### Custom Training

Specify dataset, backbone, and few-shot parameters:

```bash
python train.py \
    --data MSTAR_10_Classes \
    --backbone resnet50 \
    --n-way 5 \
    --n-shot 5 \
    --n-query 10 \
    --episodes 6000 \
    --lr 0.0001 \
    --output-dir ./my_results
```

### Training Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data` | Dataset name | MSTAR_10_Classes |
| `--backbone` | Backbone architecture | resnet50 |
| `--n-way` | Classes per episode | 5 |
| `--n-shot` | Support samples per class | 5 |
| `--n-query` | Query samples per class | 10 |
| `--episodes` | Training episodes | 6000 |
| `--lr` | Learning rate | 0.0001 |
| `--output-dir` | Output directory | ./results |

## Evaluation

### Standard Evaluation

Evaluate a trained model:

```bash
python eval.py \
    --proto-network results/proto_network.pth \
    --feature-selector results/feature_selector.pth \
    --n-way 3 \
    --n-shot 1 \
    --n-tasks 100
```

### Evaluation Output

The evaluation script produces:
- Overall accuracy
- Per-class accuracy
- Min/Max episode accuracy
- Macro F1 score
- Confusion matrix plot

## Similar Class Analysis

Analyze performance on hard-to-distinguish classes:

```bash
python src/similar_class_analysis.py \
    --proto-network results/proto_network.pth \
    --feature-selector results/feature_selector.pth \
    --similar-pair T72,T62 \
    --n-way 2 \
    --n-shot 5 \
    --eva-tasks 100 \
    --output-dir ./similar_analysis
```

### Output Visualizations

1. **Confusion Matrix**: Detailed classification matrix with counts
2. **t-SNE Distance**: Feature space visualization
3. **t-SNE Probability**: Probability space visualization
4. **Distance Scatter**: Direct distance to prototypes
5. **Confidence Plots**: Prediction confidence distribution

## Ablation Study

Compare with and without feature selection:

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

### Using Config Files

Create a custom config file:

```yaml
# my_config.yaml
data:
  name: 'MSTAR_10_Classes'

model:
  backbone: 'resnet50'

training:
  n_way: 5
  n_shot: 5
  learning_rate: 0.0001
```

Run with config:

```bash
python train.py --config my_config.yaml
```

### Configuration Hierarchy

1. Default config (`configs/default_config.yaml`)
2. Custom config file (if provided)
3. Command-line arguments (highest priority)

## Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
```bash
# Solution: Reduce batch size or use CPU
python train.py --gpu -1  # Use CPU
```

**Issue**: Dataset not found
```bash
# Solution: Check dataset path
# Ensure MSTAR-10-Classes is in project root or update config
```

**Issue**: EasyFSL compatibility error
```bash
# Solution: Use compatible version
pip install easyfsl==1.3.0
```

### Getting Help

- Check existing issues on GitHub
- Review the configuration examples
- Ensure all dependencies are installed correctly

"""
Similar Classes Analysis Script

This script evaluates the model's performance on similar classes (e.g., T72 vs T62)
to demonstrate the effectiveness of feature selection on hard-to-distinguish classes.

Usage:
    python similar_class_analysis.py \
        --proto-network path/to/proto.pth \
        --feature-selector path/to/selector.pth \
        --similar-pair T72,T62 \
        --n-way 2 \
        --n-shot 5
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms

import common_functions as c
import backbone
from evaluation import TaskSampler

# Default configuration
DEFAULT_CONFIG = {
    'data': 'MSTAR_10_Classes',
    'backbone': 'resnet50',
    'n_way': 2,
    'n_shot': 5,
    'n_query': 10,
    'eva_tasks': 100,
    'fs_hs': 256,
    'image_size': 224,
    'feature_norm': 'none',
    'post_selection_norm': 'l2',
    'proto_metric': 'cosine',
    'proto_temperature': 0.7,
    'selector': 'rl',
    'selector_eval_mode': 'threshold',
    'selector_threshold': 0.7,
    'selector_top_p': 1.0,
    'norm_mean': [0.485, 0.456, 0.406],
    'norm_std': [0.229, 0.224, 0.225],
}


def get_config(args):
    """Build configuration from args and defaults."""
    config = DEFAULT_CONFIG.copy()
    if args.n_way is not None:
        config['n_way'] = args.n_way
    if args.n_shot is not None:
        config['n_shot'] = args.n_shot
    if args.n_query is not None:
        config['n_query'] = args.n_query
    if args.eva_tasks is not None:
        config['eva_tasks'] = args.eva_tasks
    if args.backbone is not None:
        config['backbone'] = args.backbone
    return config


def setup_dataset(config, similar_classes):
    """Setup evaluation dataset with only similar classes."""
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['norm_mean'], std=config['norm_std'])
    ])

    if config['data'] == 'MSTAR_10_Classes':
        base_candidates = ["MSTAR_10_Classes", "MSTAR-10-Classes"]
        base = next((b for b in base_candidates if os.path.isdir(b)), None)
        if base is None:
            raise ValueError(f"MSTAR folder not found. Looked for: {base_candidates}")

        eval_candidates = [
            os.path.join(base, "Evaluation_classes"),
            os.path.join(base, "Images", "Evaluation_classes"),
        ]
        eval_root = next((p for p in eval_candidates if os.path.isdir(p)), None)

        if eval_root is None:
            raise ValueError("MSTAR eval folder not found")

        print(f"Evaluating on similar classes: {similar_classes}")
        eval_dataset = c.ConfigDataset(
            root_dir=eval_root,
            transform=transform,
            allowed_classes=sorted(similar_classes)
        )
    else:
        raise ValueError(f"Dataset {config['data']} not supported")

    return eval_dataset


def create_models(config, proto_network_path, feature_selector_path):
    """Create and load models."""
    # Create feature extractor
    if config['backbone'] == 'resnet18':
        feature_extractor = backbone.ModifiedResNet18().cuda()
    elif config['backbone'] == 'resnet50':
        feature_extractor = backbone.ModifiedResNet50().cuda()
    elif config['backbone'] == 'efficient_net':
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        try:
            original_efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        except Exception:
            original_efficientnet = efficientnet_b0(pretrained=True)
        feature_extractor = backbone.HookedFeatureExtractor(original_efficientnet).cuda()
    else:
        raise ValueError(f"Unknown backbone: {config['backbone']}")

    # Freeze backbone
    for p in feature_extractor.parameters():
        p.requires_grad = False
    feature_extractor.eval()

    # Create prototypical network
    proto_network = c.PrototypicalNetworks(
        feature_extractor,
        metric=config['proto_metric'],
        temperature=config['proto_temperature']
    ).cuda()

    # Create feature selector
    feature_selector = c.RLAgent(
        eval_mode=config['selector_eval_mode'],
        threshold=config['selector_threshold'],
        top_p=config['selector_top_p'],
        total_steps=config['eva_tasks']
    ).cuda()

    # Warm-up heads
    with torch.no_grad():
        dummy = torch.randn(1, 3, config['image_size'], config['image_size']).cuda()
        if hasattr(feature_extractor, "forward_with_maps"):
            _, maps, names = feature_extractor.forward_with_maps(dummy)
        elif hasattr(feature_extractor, "maps_and_flat"):
            _, maps, names = feature_extractor.maps_and_flat(dummy)
        else:
            maps, names = None, None
        if maps is not None:
            ctx = torch.zeros(5, device=dummy.device)
            _ = feature_selector.sample_task_mask(maps, names, ctx, train=False)

    # Load weights
    if feature_selector_path and os.path.exists(feature_selector_path):
        state = torch.load(feature_selector_path)
        feature_selector.load_state_dict(state, strict=False)
        print(f"Loaded feature selector from: {feature_selector_path}")

    if proto_network_path and os.path.exists(proto_network_path):
        proto_network.load_state_dict(torch.load(proto_network_path))
        print(f"Loaded proto network from: {proto_network_path}")

    return proto_network, feature_selector, feature_extractor


def evaluate_similar_classes(eval_loader, proto_network, feature_selector,
                              feature_extractor, config, similar_classes):
    """
    Evaluate specifically on similar classes.
    Returns detailed metrics and collected data for visualization.
    """
    print("\n" + "="*60)
    print(f"EVALUATION ON SIMILAR CLASSES: {similar_classes}")
    print("="*60)

    results = c.evaluate(
        eval_loader,
        proto_network,
        feature_selector,
        feature_extractor,
        feature_norm=config['feature_norm'],
        post_selection_norm=config['post_selection_norm'],
        save_confusion_matrix_path="confusion_matrix_similar_classes.png",
        cm_normalize='true'
    )

    return results


def print_similar_class_analysis(results, similar_classes):
    """Print detailed analysis for similar classes."""
    print("\n" + "="*70)
    print(f"SIMILAR CLASS ANALYSIS: {similar_classes[0]} vs {similar_classes[1]}")
    print("="*70)

    # Overall metrics
    print(f"\nOverall Accuracy: {results['accuracy']:.2f}%")
    print(f"Min Episode Accuracy: {results['min_accuracy']:.2f}%")
    print(f"Max Episode Accuracy: {results['max_accuracy']:.2f}%")
    print(f"Macro F1 Score: {results['f1_score']:.4f}")
    print(f"Avg Selected Features: {results['avg_selected_features']}")

    # Per-class accuracy
    per_class = results.get('per_class_accuracy', {})
    print("\nPer-Class Accuracy:")
    print("-"*70)
    for cls in similar_classes:
        acc = per_class.get(cls, 0.0)
        print(f"  {cls}: {acc:.2f}%")

    # Confusion analysis
    cm = results.get('confusion_matrix')
    class_names = results.get('class_names', [])
    if cm is not None and len(class_names) >= 2:
        print("\nConfusion Analysis:")
        print("-"*70)

        # Find indices of similar classes
        try:
            idx1 = class_names.index(similar_classes[0])
            idx2 = class_names.index(similar_classes[1])

            # True positives for each class
            tp1 = cm[idx1, idx1].item()
            tp2 = cm[idx2, idx2].item()

            # Misclassifications between similar classes
            t1_as_t2 = cm[idx1, idx2].item()  # T72 classified as T62
            t2_as_t1 = cm[idx2, idx1].item()  # T62 classified as T72

            total1 = cm[idx1, :].sum().item()
            total2 = cm[idx2, :].sum().item()

            print(f"  {similar_classes[0]}:")
            print(f"    Correctly classified: {tp1}/{total1} ({100*tp1/total1:.1f}%)")
            print(f"    Misclassified as {similar_classes[1]}: {t1_as_t2}/{total1} ({100*t1_as_t2/total1:.1f}%)")

            print(f"  {similar_classes[1]}:")
            print(f"    Correctly classified: {tp2}/{total2} ({100*tp2/total2:.1f}%)")
            print(f"    Misclassified as {similar_classes[0]}: {t2_as_t1}/{total2} ({100*t2_as_t1/total2:.1f}%)")

            # Inter-class confusion rate
            total_confusion = t1_as_t2 + t2_as_t1
            total_samples = total1 + total2
            confusion_rate = 100 * total_confusion / total_samples if total_samples > 0 else 0
            print(f"\n  Inter-class Confusion Rate: {confusion_rate:.2f}%")
            print(f"  (Lower is better - indicates better discrimination)")

        except ValueError:
            print("  Could not find similar classes in confusion matrix")

    print("="*70)


def extract_features_for_similar_classes(eval_loader, proto_network, feature_selector,
                                          feature_extractor, config, similar_classes, max_samples=300):
    """
    Extract features, distances, and probabilities for similar classes visualization.
    Returns multiple representations for different visualization options.
    """
    print(f"\nExtracting features and distances ({similar_classes[0]} vs {similar_classes[1]})...")

    all_features = []
    all_labels = []
    all_selected_k = []
    all_distances = []  # Distance to each prototype
    all_probabilities = []  # Softmax probabilities
    total_samples = 0

    proto_network.eval()
    if hasattr(feature_selector, 'eval'):
        feature_selector.eval()

    with torch.no_grad():
        for batch_idx, (support_images, support_labels, query_images, query_labels, _) in enumerate(eval_loader):
            if total_samples >= max_samples:
                break

            support_images = support_images.cuda()
            support_labels = support_labels.cuda()
            query_images = query_images.cuda()
            query_labels = query_labels.cuda()

            # Extract features
            if hasattr(feature_extractor, "forward_with_maps"):
                support_flat, s_maps, s_names = feature_extractor.forward_with_maps(support_images)
                query_flat, q_maps, q_names = feature_extractor.forward_with_maps(query_images)
            elif hasattr(feature_extractor, "maps_and_flat"):
                support_flat, s_maps, s_names = feature_extractor.maps_and_flat(support_images)
                query_flat, q_maps, q_names = feature_extractor.maps_and_flat(query_images)
            else:
                support_flat = feature_extractor(support_images)
                query_flat = feature_extractor(query_images)
                s_maps = None

            # Normalize if needed
            if config['feature_norm'] == 'l2':
                support_flat = F.normalize(support_flat, p=2, dim=1)
                query_flat = F.normalize(query_flat, p=2, dim=1)

            # Apply feature selection
            if s_maps is not None and feature_selector is not None:
                ctx = c.build_task_context(support_flat, support_labels)
                mask_flat, _ = feature_selector.sample_task_mask(s_maps, s_names, ctx, train=False)
                selected_k = int(mask_flat.sum().item())
                query_flat_selected = query_flat * mask_flat
                support_flat_selected = support_flat * mask_flat
            else:
                selected_k = query_flat.shape[1]
                query_flat_selected = query_flat
                support_flat_selected = support_flat

            # Normalize after selection
            if config['post_selection_norm'] == 'l2':
                query_flat_selected = F.normalize(query_flat_selected, p=2, dim=1)
                support_flat_selected = F.normalize(support_flat_selected, p=2, dim=1)

            # Compute prototypes
            unique_labels = torch.unique(support_labels)
            prototypes = torch.stack([support_flat_selected[support_labels == l].mean(0) for l in unique_labels])

            # Compute distances from queries to prototypes
            if config['proto_metric'] == 'cosine':
                q_norm = F.normalize(query_flat_selected, p=2, dim=1)
                p_norm = F.normalize(prototypes, p=2, dim=1)
                distances = 1.0 - (q_norm @ p_norm.t()).clamp(-1, 1)
            else:
                distances = torch.cdist(query_flat_selected, prototypes)

            # Compute logits and probabilities
            logits = -distances / max(1e-6, config['proto_temperature'])
            probabilities = F.softmax(logits, dim=1)

            # Collect data
            query_labels_cpu = query_labels.cpu().numpy()
            distances_cpu = distances.cpu().numpy()
            probs_cpu = probabilities.cpu().numpy()
            query_features = query_flat_selected.cpu().numpy()

            for i in range(len(query_labels_cpu)):
                if total_samples >= max_samples:
                    break
                all_features.append(query_features[i])
                all_labels.append(query_labels_cpu[i])
                all_distances.append(distances_cpu[i])
                all_probabilities.append(probs_cpu[i])
                all_selected_k.append(selected_k)
                total_samples += 1

    features = np.array(all_features)
    labels = np.array(all_labels)
    distances = np.array(all_distances)
    probabilities = np.array(all_probabilities)
    avg_selected_k = int(np.mean(all_selected_k)) if all_selected_k else 0

    print(f"Extracted {len(labels)} samples")
    print(f"Average features selected: {avg_selected_k}")

    return features, labels, distances, probabilities, avg_selected_k


def plot_tsne_similar_classes(features, labels, class_names, similar_classes, save_path="tsne_similar_classes.png"):
    """
    Create t-SNE plot for similar classes (using distances instead of raw features).
    """
    print("\nComputing t-SNE on distances...")

    # Apply t-SNE on distances (low dim: n_classes)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels)-1))
    embeddings = tsne.fit_transform(features)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Colors for the two similar classes
    colors = ['#e74c3c', '#3498db']  # Red and blue

    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=colors[i % len(colors)], label=class_name,
                   alpha=0.7, s=80, edgecolors='white', linewidth=1)

    ax.set_xlabel('t-SNE Dimension 1', fontsize=24)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(fontsize=12, loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to: {save_path}")


def plot_probabilities_tSNE(probabilities, labels, class_names, similar_classes, save_path="prob_tsne_similar_classes.png"):
    """
    OPTION 2: t-SNE on class probabilities.
    Shows how confident the model is in its predictions.
    """
    print("\nComputing t-SNE on probabilities (Option 2)...")

    # Apply t-SNE on probabilities (2D for 2 classes)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels)-1))
    embeddings = tsne.fit_transform(probabilities)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#e74c3c', '#3498db']

    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=colors[i % len(colors)], label=class_name,
                   alpha=0.7, s=80, edgecolors='white', linewidth=1)

    ax.set_xlabel('t-SNE Dimension 1', fontsize=24)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(fontsize=12, loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved probability t-SNE plot to: {save_path}")


def plot_direct_distances(distances, labels, class_names, similar_classes, save_path="distances_similar_classes.png"):
    """
    OPTION 3 & 4: Direct 2D scatter plot of distances to each prototype.
    X-axis: distance to class 0 prototype
    Y-axis: distance to class 1 prototype
    """
    print("\nCreating direct distance plot (Options 3 & 4)...")

    # Get distances to each prototype
    dist_to_class0 = distances[:, 0]
    dist_to_class1 = distances[:, 1]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Option 3: t-SNE on 2D distance vectors [dist_to_cls0, dist_to_cls1]
    ax = axes[0]
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels)-1))
    embeddings = tsne.fit_transform(distances)

    colors = ['#e74c3c', '#3498db']
    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=colors[i % len(colors)], label=class_name,
                   alpha=0.7, s=80, edgecolors='white', linewidth=1)

    ax.set_xlabel('t-SNE Dimension 1', fontsize=24)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3)

    # Option 4: Direct scatter plot (no t-SNE)
    ax = axes[1]
    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        ax.scatter(dist_to_class0[mask], dist_to_class1[mask],
                   c=colors[i % len(colors)], label=class_name,
                   alpha=0.7, s=80, edgecolors='white', linewidth=1)

    # Add decision boundary (diagonal line where dist_to_cls0 == dist_to_cls1)
    max_dist = max(dist_to_class0.max(), dist_to_class1.max())
    min_dist = min(dist_to_class0.min(), dist_to_class1.min())
    ax.plot([min_dist, max_dist], [min_dist, max_dist], 'k--', linewidth=2, alpha=0.5, label='Decision Boundary')

    ax.set_xlabel(f'Distance to {similar_classes[0]} Prototype', fontsize=24)
    ax.set_ylabel(f'Distance to {similar_classes[1]} Prototype', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved distance plots to: {save_path}")


def plot_decision_confidence(probabilities, labels, class_names, similar_classes, save_path="confidence_similar_classes.png"):
    """
    Additional plot: Show prediction confidence distribution.
    """
    print("\nCreating confidence distribution plot...")

    # Get max probability (confidence) for each prediction
    confidence = probabilities.max(axis=1)
    pred_labels = probabilities.argmax(axis=1)

    # Separate correct and incorrect predictions
    correct_mask = pred_labels == labels
    incorrect_mask = ~correct_mask

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Confidence histogram
    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    ax.hist(confidence[correct_mask], bins=bins, alpha=0.7, label='Correct', color='#2ecc71', edgecolor='white')
    ax.hist(confidence[incorrect_mask], bins=bins, alpha=0.7, label='Incorrect', color='#e74c3c', edgecolor='white')
    ax.set_xlabel('Prediction Confidence', fontsize=24)
    ax.set_ylabel('Count', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Confidence by true class
    ax = axes[1]
    colors = ['#e74c3c', '#3498db']
    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        ax.scatter(range(len(confidence[mask])), confidence[mask],
                   c=colors[i % len(colors)], label=class_name,
                   alpha=0.6, s=60, edgecolors='white', linewidth=0.5)

    ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='50% threshold')
    ax.set_xlabel('Sample Index', fontsize=24)
    ax.set_ylabel('Prediction Confidence', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confidence plot to: {save_path}")


def plot_confusion_matrix_detailed(cm, class_names, similar_classes, save_path="confusion_similar_detailed.png"):
    """Plot detailed confusion matrix for similar classes."""
    if cm is None or len(class_names) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalize by row (true labels)
    cm_np = cm.astype(float)
    row_sums = cm_np.sum(axis=1, keepdims=True) + 1e-12
    cm_norm = cm_np / row_sums

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', aspect='auto', vmin=0, vmax=1)

    ax.set_xlabel('Predicted Label', fontsize=24)
    ax.set_ylabel('True Label', fontsize=24)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=22)
    ax.set_yticklabels(class_names, fontsize=22)

    # Annotate cells
    thresh = cm_norm.max() / 2.0 if cm_norm.size > 0 else 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_norm[i, j]
            count = int(cm[i, j])
            txt = f"{val:.2f}\n({count})"
            ax.text(j, i, txt,
                   horizontalalignment="center",
                   verticalalignment="center",
                   color="white" if val > thresh else "black",
                   fontsize=20)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved detailed confusion matrix to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Similar Classes Analysis')
    parser.add_argument('--proto-network', type=str, required=True,
                       help='Path to proto_network .pth file')
    parser.add_argument('--feature-selector', type=str, required=True,
                       help='Path to feature_selector .pth file')
    parser.add_argument('--similar-pair', type=str, required=True,
                       help='Comma-separated pair of similar classes (e.g., T72,T62)')
    parser.add_argument('--n-way', type=int, default=2,
                       help='Number of classes per episode (default: 2 for pair analysis)')
    parser.add_argument('--n-shot', type=int, default=5,
                       help='Number of support samples per class (default: 5)')
    parser.add_argument('--n-query', type=int, default=10,
                       help='Number of query samples per class (default: 10)')
    parser.add_argument('--eva-tasks', type=int, default=100,
                       help='Number of evaluation tasks (default: 100)')
    parser.add_argument('--backbone', type=str, default=None,
                       choices=['resnet18', 'resnet50', 'efficient_net'],
                       help='Backbone architecture')
    parser.add_argument('--max-tsne-samples', type=int, default=300,
                       help='Maximum samples for t-SNE (default: 300)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Directory to save output plots')

    args = parser.parse_args()

    # Parse similar classes
    similar_classes = [cls.strip() for cls in args.similar_pair.split(',')]
    if len(similar_classes) != 2:
        raise ValueError("--similar-pair must contain exactly 2 class names separated by comma")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get configuration
    config = get_config(args)

    print("\n" + "="*60)
    print("SIMILAR CLASS ANALYSIS CONFIGURATION")
    print("="*60)
    print(f"  Similar Classes: {similar_classes[0]} vs {similar_classes[1]}")
    print(f"  n_way: {config['n_way']}")
    print(f"  n_shot: {config['n_shot']}")
    print(f"  n_query: {config['n_query']}")
    print(f"  eva_tasks: {config['eva_tasks']}")
    print(f"  Backbone: {config['backbone']}")
    print("="*60)

    # Setup dataset with only similar classes
    eval_dataset = setup_dataset(config, similar_classes)

    # Create data loader
    eval_sampler = TaskSampler(
        eval_dataset,
        n_way=config['n_way'],
        n_shot=config['n_shot'],
        n_query=config['n_query'],
        n_tasks=config['eva_tasks']
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_sampler=eval_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=eval_sampler.episodic_collate_fn
    )

    # Create models
    proto_network, feature_selector, feature_extractor = create_models(
        config,
        args.proto_network,
        args.feature_selector
    )

    # Run evaluation on similar classes
    results = evaluate_similar_classes(
        eval_loader,
        proto_network,
        feature_selector,
        feature_extractor,
        config,
        similar_classes
    )

    # Print detailed analysis
    print_similar_class_analysis(results, similar_classes)

    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # 1. Detailed confusion matrix
    cm = results.get('confusion_matrix')
    class_names = results.get('class_names', similar_classes)
    if cm is not None:
        plot_confusion_matrix_detailed(
            cm,
            class_names,
            similar_classes,
            save_path=os.path.join(args.output_dir, f"confusion_{similar_classes[0]}_vs_{similar_classes[1]}.png")
        )

    # 2. t-SNE visualization
    # Recreate data loader for t-SNE
    eval_sampler_tsne = TaskSampler(
        eval_dataset,
        n_way=config['n_way'],
        n_shot=config['n_shot'],
        n_query=config['n_query'],
        n_tasks=config['eva_tasks']
    )
    eval_loader_tsne = DataLoader(
        eval_dataset,
        batch_sampler=eval_sampler_tsne,
        num_workers=0,
        pin_memory=True,
        collate_fn=eval_sampler_tsne.episodic_collate_fn
    )

    features, labels, distances, probabilities, avg_selected = extract_features_for_similar_classes(
        eval_loader_tsne,
        proto_network,
        feature_selector,
        feature_extractor,
        config,
        similar_classes,
        max_samples=args.max_tsne_samples
    )

    # 2. t-SNE on distances (Option 1 - modified)
    plot_tsne_similar_classes(
        distances,  # Use distances instead of features
        labels,
        class_names,
        similar_classes,
        save_path=os.path.join(args.output_dir, f"tsne_distances_{similar_classes[0]}_vs_{similar_classes[1]}.png")
    )

    # 3. t-SNE on probabilities (Option 2)
    plot_probabilities_tSNE(
        probabilities,
        labels,
        class_names,
        similar_classes,
        save_path=os.path.join(args.output_dir, f"tsne_probabilities_{similar_classes[0]}_vs_{similar_classes[1]}.png")
    )

    # 4. Direct distance plots (Options 3 & 4)
    plot_direct_distances(
        distances,
        labels,
        class_names,
        similar_classes,
        save_path=os.path.join(args.output_dir, f"distances_{similar_classes[0]}_vs_{similar_classes[1]}.png")
    )

    # 5. Confidence analysis
    plot_decision_confidence(
        probabilities,
        labels,
        class_names,
        similar_classes,
        save_path=os.path.join(args.output_dir, f"confidence_{similar_classes[0]}_vs_{similar_classes[1]}.png")
    )

    print("\n" + "="*60)
    print("SIMILAR CLASS ANALYSIS COMPLETE")
    print("="*60)
    print(f"Output files saved to: {args.output_dir}")
    print(f"\nKey Results:")
    print(f"  - Overall Accuracy: {results['accuracy']:.2f}%")
    print(f"  - Macro F1 Score: {results['f1_score']:.4f}")
    print(f"  - Avg Features Selected: {results['avg_selected_features']}")
    print(f"\nGenerated Visualizations:")
    print(f"  1. Confusion Matrix")
    print(f"  2. t-SNE on Distances (Option 1)")
    print(f"  3. t-SNE on Probabilities (Option 2)")
    print(f"  4. Distance Scatter Plots (Options 3 & 4)")
    print(f"  5. Confidence Distribution")


if __name__ == "__main__":
    main()

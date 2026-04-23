"""
Ablation Study Visualization Script

This script performs ablation studies and generates:
1. Per-class accuracy comparison (bar chart with selection vs without selection)
2. Side-by-side t-SNE plots comparing feature distributions

Usage:
    python ablation_visualization.py --proto-network path/to/proto.pth --feature-selector path/to/selector.pth
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

# Default configuration (can be overridden via command line)
DEFAULT_CONFIG = {
    'data': 'MSTAR_10_Classes',
    'backbone': 'resnet50',
    'n_way': 3,
    'n_shot': 1,
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
    'use_novel_split': True,
    'base_class_count': 7,
    'norm_mean': [0.485, 0.456, 0.406],
    'norm_std': [0.229, 0.224, 0.225],
}


def get_evaluation_config(args):
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


def setup_dataset(config):
    """Setup evaluation dataset."""
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

        train_candidates = [
            os.path.join(base, "Train_classes"),
            os.path.join(base, "Images", "Train_classes"),
        ]
        eval_candidates = [
            os.path.join(base, "Evaluation_classes"),
            os.path.join(base, "Images", "Evaluation_classes"),
        ]
        train_root = next((p for p in train_candidates if os.path.isdir(p)), None)
        eval_root = next((p for p in eval_candidates if os.path.isdir(p)), None)

        if train_root is None or eval_root is None:
            raise ValueError("MSTAR train/eval folders not found")

        # Get all classes
        all_classes = sorted([
            d for d in os.listdir(train_root)
            if os.path.isdir(os.path.join(train_root, d)) and not d.startswith('.')
        ])

        # Split to base/novel
        base_set = set(all_classes[:config['base_class_count']])
        novel_set = set(all_classes[config['base_class_count']:])

        print(f"Evaluation on novel classes ({len(novel_set)}): {sorted(list(novel_set))}")
        eval_dataset = c.ConfigDataset(
            root_dir=eval_root,
            transform=transform,
            allowed_classes=sorted(list(novel_set))
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
    if config['selector'] == 'rl':
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
    else:
        feature_selector = None

    # Load prototypical network weights
    if proto_network_path and os.path.exists(proto_network_path):
        proto_network.load_state_dict(torch.load(proto_network_path))

    return proto_network, feature_selector, feature_extractor


def evaluate_with_and_without_selection(eval_loader, proto_network_with, proto_network_without,
                                         feature_selector, feature_extractor, config):
    """
    Run evaluation twice: once with selection, once without.
    Returns results dict for both conditions.
    """
    print("\n" + "="*60)
    print("EVALUATION WITH FEATURE SELECTION")
    print("="*60)

    # Evaluate WITH selection
    results_with = c.evaluate(
        eval_loader,
        proto_network_with,
        feature_selector,
        feature_extractor,
        feature_norm=config['feature_norm'],
        post_selection_norm=config['post_selection_norm'],
        save_confusion_matrix_path="confusion_matrix_with_selection.png",
        cm_normalize='true'
    )

    print("\n" + "="*60)
    print("EVALUATION WITHOUT FEATURE SELECTION (All Features)")
    print("="*60)

    # Create a dummy "no selection" selector that returns all features
    class NoSelectionWrapper:
        """Wrapper that applies no feature selection (identity)."""
        def __init__(self):
            self.training = False

        def eval(self):
            pass

        def __call__(self, *args, **kwargs):
            # Return all ones (select all features)
            return None

        def parameters(self):
            return []

    dummy_selector = NoSelectionWrapper()

    # Evaluate WITHOUT selection (force no masking)
    # We need to modify the evaluate function behavior for no-selection case
    # We'll pass top_k=None which triggers the legacy path to use all features
    results_without = c.evaluate(
        eval_loader,
        proto_network_without,
        dummy_selector,  # Dummy selector
        feature_extractor,
        top_k=None,  # Use all features
        feature_norm=config['feature_norm'],
        post_selection_norm=config['post_selection_norm'],
        save_confusion_matrix_path="confusion_matrix_without_selection.png",
        cm_normalize='true'
    )

    return results_with, results_without


def plot_per_class_accuracy_comparison(results_with, results_without, save_path="per_class_accuracy_comparison.png"):
    """
    Create grouped bar chart comparing per-class accuracy with and without selection.
    """
    # Get class names and accuracies
    classes_with = results_with.get('per_class_accuracy', {})
    classes_without = results_without.get('per_class_accuracy', {})

    # Get all unique classes
    all_classes = sorted(set(list(classes_with.keys()) + list(classes_without.keys())))

    if not all_classes:
        print("No per-class accuracy data available")
        return

    # Prepare data
    acc_with = [classes_with.get(cls, 0) for cls in all_classes]
    acc_without = [classes_without.get(cls, 0) for cls in all_classes]

    # Create bar chart
    x = np.arange(len(all_classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, acc_without, width, label='Without Selection', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, acc_with, width, label='With Selection', color='#3498db', alpha=0.8)

    # Add labels and title
    ax.set_xlabel('Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Per-Class Accuracy: With vs Without Feature Selection', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=12, loc='upper left')

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=90)

    add_labels(bars1)
    add_labels(bars2)

    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved per-class accuracy comparison to: {save_path}")


def extract_features_for_tsne(eval_loader, proto_network, feature_selector,
                               feature_extractor, config, max_samples=500):
    """
    Extract features for t-SNE visualization.
    Returns features and labels. If feature_selector is provided, returns both 
    with and without selection features. Otherwise returns same features for both.
    """
    use_selection = feature_selector is not None
    if use_selection:
        print("\nExtracting features for t-SNE (with selection enabled)...")
    else:
        print("\nExtracting features for t-SNE (without selection)...")

    all_features_with = []
    all_features_without = []
    all_labels = []
    total_samples = 0

    proto_network.eval()
    if feature_selector is not None and hasattr(feature_selector, 'eval'):
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

            # Features WITHOUT selection
            query_flat_without = query_flat.cpu().numpy()

            # Features WITH selection (only if selector provided)
            if use_selection and s_maps is not None:
                ctx = c.build_task_context(support_flat, support_labels)
                mask_flat, _ = feature_selector.sample_task_mask(s_maps, s_names, ctx, train=False)
                query_flat_with = (query_flat * mask_flat).cpu().numpy()
            else:
                query_flat_with = query_flat_without

            # Collect query samples
            query_labels_cpu = query_labels.cpu().numpy()

            for i in range(len(query_labels_cpu)):
                if total_samples >= max_samples:
                    break
                all_features_with.append(query_flat_with[i])
                all_features_without.append(query_flat_without[i])
                all_labels.append(query_labels_cpu[i])
                total_samples += 1

    features_with = np.array(all_features_with)
    features_without = np.array(all_features_without)
    labels = np.array(all_labels)

    print(f"Extracted {len(labels)} samples for t-SNE")

    return features_with, features_without, labels


def plot_tsne_comparison(features_with, features_without, labels, class_names,
                          save_path="tsne_comparison.png"):
    """
    Create side-by-side t-SNE plots comparing feature distributions.
    """
    print("\nComputing t-SNE (this may take a moment)...")

    # Apply t-SNE to both feature sets
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels)-1))

    # t-SNE for features WITHOUT selection
    print("  Computing t-SNE for features WITHOUT selection...")
    embeddings_without = tsne.fit_transform(features_without)

    # t-SNE for features WITH selection
    print("  Computing t-SNE for features WITH selection...")
    tsne_with = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels)-1))
    embeddings_with = tsne_with.fit_transform(features_with)

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    # Plot WITHOUT selection
    ax = axes[0]
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        ax.scatter(embeddings_without[mask, 0], embeddings_without[mask, 1],
                   c=[colors[i]], label=class_name, alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    ax.set_title('t-SNE: Without Feature Selection', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    # Plot WITH selection
    ax = axes[1]
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_name = class_names[label] if label < len(class_names) else f"Class {label}"
        ax.scatter(embeddings_with[mask, 0], embeddings_with[mask, 1],
                   c=[colors[i]], label=class_name, alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    ax.set_title('t-SNE: With Feature Selection', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE comparison to: {save_path}")


def print_summary_table(results_with, results_without):
    """Print a summary comparison table."""
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    print(f"{'Metric':<40} {'With Selection':<15} {'Without Selection':<15}")
    print("-"*70)
    print(f"{'Overall Accuracy':<40} {results_with['accuracy']:>14.2f}% {results_without['accuracy']:>14.2f}%")
    print(f"{'Min Episode Accuracy':<40} {results_with['min_accuracy']:>14.2f}% {results_without['min_accuracy']:>14.2f}%")
    print(f"{'Max Episode Accuracy':<40} {results_with['max_accuracy']:>14.2f}% {results_without['max_accuracy']:>14.2f}%")
    print(f"{'Macro F1 Score':<40} {results_with['f1_score']:>14.4f}  {results_without['f1_score']:>14.4f}")
    print(f"{'Avg Selected Features':<40} {results_with['avg_selected_features']:>14}  {results_without.get('avg_selected_features', 'N/A'):>14}")
    print("="*70)


def print_per_class_accuracy_table(results_with, results_without):
    """Print detailed per-class accuracy comparison table."""
    print("\n" + "="*70)
    print("PER-CLASS ACCURACY COMPARISON")
    print("="*70)
    
    classes_with = results_with.get('per_class_accuracy', {})
    classes_without = results_without.get('per_class_accuracy', {})
    
    # Get all unique classes sorted
    all_classes = sorted(set(list(classes_with.keys()) + list(classes_without.keys())))
    
    if not all_classes:
        print("No per-class accuracy data available")
        return
    
    print(f"{'Class':<30} {'With Selection':<20} {'Without Selection':<20} {'Difference':<15}")
    print("-"*70)
    
    for cls in all_classes:
        acc_with = classes_with.get(cls, 0.0)
        acc_without = classes_without.get(cls, 0.0)
        diff = acc_with - acc_without
        diff_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
        print(f"{cls:<30} {acc_with:>18.2f}% {acc_without:>18.2f}% {diff_str:>14}")
    
    print("-"*70)
    # Print average
    avg_with = sum(classes_with.values()) / len(classes_with) if classes_with else 0.0
    avg_without = sum(classes_without.values()) / len(classes_without) if classes_without else 0.0
    avg_diff = avg_with - avg_without
    avg_diff_str = f"+{avg_diff:.2f}%" if avg_diff >= 0 else f"{avg_diff:.2f}%"
    print(f"{'AVERAGE':<30} {avg_with:>18.2f}% {avg_without:>18.2f}% {avg_diff_str:>14}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Ablation Study Visualization')
    parser.add_argument('--proto-network', type=str, required=True,
                       help='Path to proto_network .pth file (with selection)')
    parser.add_argument('--feature-selector', type=str, default=None,
                       help='Path to feature_selector .pth file (optional, for with selection only)')
    parser.add_argument('--proto-network-without', type=str, default=None,
                       help='Path to proto_network .pth file for WITHOUT selection (if different from --proto-network)')
    parser.add_argument('--n-way', type=int, default=None,
                       help='Number of classes per episode (default: 3)')
    parser.add_argument('--n-shot', type=int, default=None,
                       help='Number of support samples per class (default: 1)')
    parser.add_argument('--n-query', type=int, default=None,
                       help='Number of query samples per class (default: 10)')
    parser.add_argument('--eva-tasks', type=int, default=None,
                       help='Number of evaluation tasks (default: 100)')
    parser.add_argument('--backbone', type=str, default=None,
                       choices=['resnet18', 'resnet50', 'efficient_net'],
                       help='Backbone architecture')
    parser.add_argument('--max-tsne-samples', type=int, default=500,
                       help='Maximum samples for t-SNE (default: 500)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Directory to save output plots')

    args = parser.parse_args()

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Get configuration
    config = get_evaluation_config(args)

    print("\n" + "="*60)
    print("ABLATION STUDY CONFIGURATION")
    print("="*60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"  Proto Network: {args.proto_network}")
    print(f"  Feature Selector: {args.feature_selector}")
    print("="*60)

    # Setup dataset
    eval_dataset = setup_dataset(config)

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

    # Determine paths for with and without selection
    proto_network_path_with = args.proto_network
    proto_network_path_without = args.proto_network_without if args.proto_network_without else args.proto_network
    
    print("\n" + "="*60)
    print("MODEL PATHS")
    print("="*60)
    print(f"  With Selection:")
    print(f"    Proto Network: {proto_network_path_with}")
    print(f"    Feature Selector: {args.feature_selector if args.feature_selector else 'None (will use dummy)'}")
    print(f"  Without Selection:")
    print(f"    Proto Network: {proto_network_path_without}")
    print(f"    Feature Selector: N/A (not needed)")
    print("="*60)

    # Create models for WITH selection
    print("\nLoading models for WITH selection...")
    proto_network_with, feature_selector, feature_extractor = create_models(
        config,
        proto_network_path_with,
        args.feature_selector
    )

    # Create models for WITHOUT selection (may be different proto network)
    print("Loading models for WITHOUT selection...")
    proto_network_without, _, _ = create_models(
        config,
        proto_network_path_without,
        None  # No feature selector needed
    )

    # Run ablation evaluations
    results_with, results_without = evaluate_with_and_without_selection(
        eval_loader,
        proto_network_with,
        proto_network_without,
        feature_selector,
        feature_extractor,
        config
    )

    # Print per-class accuracy table
    print_per_class_accuracy_table(results_with, results_without)

    # Print summary
    print_summary_table(results_with, results_without)

    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # 1. Per-class accuracy comparison
    plot_per_class_accuracy_comparison(
        results_with,
        results_without,
        save_path=os.path.join(args.output_dir, "per_class_accuracy_comparison.png")
    )

    # 2. t-SNE comparison (need to re-run data loader)
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

    # Extract features using the appropriate models
    print("\nExtracting features for t-SNE (WITH selection)...")
    features_with, _, labels_with = extract_features_for_tsne(
        eval_loader_tsne,
        proto_network_with,
        feature_selector,
        feature_extractor,
        config,
        max_samples=args.max_tsne_samples
    )

    # Recreate data loader for without selection
    eval_sampler_tsne2 = TaskSampler(
        eval_dataset,
        n_way=config['n_way'],
        n_shot=config['n_shot'],
        n_query=config['n_query'],
        n_tasks=config['eva_tasks']
    )
    eval_loader_tsne2 = DataLoader(
        eval_dataset,
        batch_sampler=eval_sampler_tsne2,
        num_workers=0,
        pin_memory=True,
        collate_fn=eval_sampler_tsne2.episodic_collate_fn
    )

    print("Extracting features for t-SNE (WITHOUT selection)...")
    _, features_without, labels_without = extract_features_for_tsne(
        eval_loader_tsne2,
        proto_network_without,
        None,  # No feature selector
        feature_extractor,
        config,
        max_samples=args.max_tsne_samples
    )

    # Use labels from with selection (should be same)
    labels = labels_with if len(labels_with) > 0 else labels_without

    class_names = results_with.get('class_names', [f"Class {i}" for i in range(config['n_way'])])
    plot_tsne_comparison(
        features_with,
        features_without,
        labels,
        class_names,
        save_path=os.path.join(args.output_dir, "tsne_comparison.png")
    )

    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE")
    print("="*60)
    print(f"Output files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

import os
import time

# from torchvision.models import efficientnet_b0

import common_functions as c
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import random
from typing import Dict, Iterator, List, Tuple, Union
from torchvision import transforms
import backbone
import psutil
from tqdm import tqdm

# Custom TaskSampler to fix PyTorch 2.x compatibility
class TaskSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks.
    """
    def __init__(
        self,
        dataset,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int,
    ):
        super().__init__()
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks

        self.items_per_label: Dict[int, List[int]] = {}
        for item, label in enumerate(dataset.get_labels()):
            if label in self.items_per_label:
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

        self._check_dataset_size_fits_sampler_parameters()

    def __len__(self) -> int:
        return self.n_tasks

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.n_tasks):
            yield torch.cat(
                [
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label], self.n_shot + self.n_query
                        )
                    )
                    for label in random.sample(
                        sorted(self.items_per_label.keys()), self.n_way
                    )
                ]
            ).tolist()

    def episodic_collate_fn(
        self, input_data: List[Tuple[torch.Tensor, Union[torch.Tensor, int]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        input_data_with_int_labels = [
            (image, int(label) if isinstance(label, torch.Tensor) else label)
            for image, label in input_data
        ]
        true_class_ids = list({x[1] for x in input_data_with_int_labels})
        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data_with_int_labels])
        all_images = all_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
        )
        all_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in input_data_with_int_labels]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        support_images = all_images[:, : self.n_shot].reshape(
            (-1, *all_images.shape[2:])
        )
        query_images = all_images[:, self.n_shot :].reshape((-1, *all_images.shape[2:]))
        support_labels = all_labels[:, : self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot :].flatten()
        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids,
        )

    def _check_dataset_size_fits_sampler_parameters(self):
        if self.n_way > len(self.items_per_label):
            raise ValueError(
                f"The number of labels in the dataset ({len(self.items_per_label)} "
                f"must be greater or equal to n_way ({self.n_way})."
            )
        number_of_samples_per_label = [
            len(items_for_label) for items_for_label in self.items_per_label.values()
        ]
        minimum_number_of_samples_per_label = min(number_of_samples_per_label)
        if self.n_shot + self.n_query > minimum_number_of_samples_per_label:
            raise ValueError(
                f"A label has only {minimum_number_of_samples_per_label} samples "
                f"but all classes must have at least n_shot + n_query ({self.n_shot + self.n_query}) samples."
            )

IMAGE_SIZE = 224

my_data = 'MSTAR_10_Classes'
my_backbone = 'resnet50'
my_n_way = 3
my_n_shot = 2
my_n_query = 10
my_episodes = 6000
my_eva_tasks = 10
my_lr = 0.0001
my_fs_hs = 256

# Backbone training options
my_finetune_backbone = True         # Train backbone during training
my_backbone_lr_scale = 0.2          # Backbone LR = my_lr * scale (more adaptation for SAR)

# Feature vector pre-processing before selection/classification
# Options: 'none', 'l2'
my_feature_norm = 'none' # no norm before selection

# Normalization after selection before prototypical distance computation
# Options: 'none', 'l2'
# Use 'l2' to stabilize metric geometry for prototypical distances (recommended for novel-class transfer)
my_post_selection_norm = 'l2'

# Feature selection mode
# 'rl' => RLAgent (actor-critic with stochastic gates)
# 'mlp' => legacy dense MLP mask + hard Top-K
my_selector = 'rl'

# RLAgent evaluation mode when running evaluation
# 'threshold' (variable K, no target), 'threshold_calibrated' (≈ K_target), 'top_p' (variable K), 'topk' (fixed K)
my_selector_eval_mode = 'threshold'  # let RL decide via probability threshold (no fixed K)
my_selector_threshold = 0.7
my_selector_top_p = 0.9

# RLAgent training mode: 'bern' (Bernoulli REINFORCE) | 'st_topk' (straight-through Top‑K)
my_rl_train_mode = 'bern'  # no fixed K during training

# Normalization stats (override ImageNet if you have SAR-specific mean/std after Grayscale->3ch replication)
my_norm_mean = [0.485, 0.456, 0.406]
my_norm_std  = [0.229, 0.224, 0.225]

# Prototypical head metric options
# metric: 'euclidean' | 'cosine'; temperature: positive float
my_proto_metric = 'cosine'
my_proto_temperature = 0.7

# Novel-class split options
my_use_novel_split = True
my_base_class_count = 5
# Manual override lists. If provided, they take precedence over my_use_novel_split.
# my_manual_base_classes = ['2S1', 'BMP2', 'BRDM2', 'BTR60', 'BTR70', 'D7', 'T72']
# my_manual_novel_classes = ['T62', 'ZIL131', 'ZSU_23_4']
my_manual_base_classes = []
my_manual_novel_classes = []
# (removed duplicate novel-split config; using variables defined above)

# Training-time transform (with light SAR-friendly augmentation)
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.05, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=my_norm_mean, std=my_norm_std),
])

# Evaluation-time transform (deterministic)
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=my_norm_mean, std=my_norm_std),
])

if my_data == 'UCMerced_LandUse':
    train_dataset = c.ConfigDataset(root_dir=r"UCMerced_LandUse\\Images\\Train_classes", transform=train_transform)
    eval_dataset = c.ConfigDataset(root_dir=r"UCMerced_LandUse\\Images\\Evaluation_classes", transform=eval_transform)
elif my_data == 'nwpu_resisc45':
    train_dataset = c.ConfigDataset(root_dir=r"nwpu_resisc45\\train", transform=train_transform)
    eval_dataset = c.ConfigDataset(root_dir=r"nwpu_resisc45\\test", transform=eval_transform)
elif my_data == 'MSTAR_10_Classes':
    # Support both "MSTAR_10_Classes" and "MSTAR-10-Classes" at project root
    base_candidates = [r"MSTAR_10_Classes", r"MSTAR-10-Classes"]
    base = None
    for cand in base_candidates:
        if os.path.isdir(cand):
            base = cand
            break
    if base is None:
        raise ValueError(f"MSTAR folder not found. Looked for: {base_candidates}. Please place the dataset at project root.")

    # Training: MSTAR_10_Classes/Train_classes/<class>/images...
    # Evaluation: MSTAR_10_Classes/Evaluation_classes/<class>/images... (or Images/Evaluation_classes)
    train_candidates = [
        os.path.join(base, "Train_classes"),
        os.path.join(base, "Images", "Train_classes"),
    ]
    eval_candidates = [
        os.path.join(base, "Evaluation_classes"),
        os.path.join(base, "Images", "Evaluation_classes"),
    ]

    train_root = next((p for p in train_candidates if os.path.isdir(p)), None)
    eval_root  = next((p for p in eval_candidates if os.path.isdir(p)), None)

    if train_root is None:
        raise ValueError(f"MSTAR training folder not found. Tried: {train_candidates}")
    if eval_root is None:
        raise ValueError(f"MSTAR evaluation folder not found. Tried: {eval_candidates}")

    print(f"Using MSTAR paths -> train: {train_root} | eval: {eval_root}")

    # Discover all class names from training root (authoritative list), ignore hidden/system dirs like .ipynb_checkpoints
    all_classes = sorted([
        d for d in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, d)) and not d.startswith('.')
    ])

    # Resolve base/novel sets using manual overrides or count-based split
    base_set = set()
    novel_set = set()

    if my_manual_base_classes or my_manual_novel_classes:
        name_set = set(all_classes)
        if my_manual_base_classes:
            missing = [cname for cname in my_manual_base_classes if cname not in name_set]
            if missing:
                raise ValueError(f"Unknown classes in my_manual_base_classes: {missing}. Available: {all_classes}")
        if my_manual_novel_classes:
            missing = [cname for cname in my_manual_novel_classes if cname not in name_set]
            if missing:
                raise ValueError(f"Unknown classes in my_manual_novel_classes: {missing}. Available: {all_classes}")

        if my_manual_base_classes and my_manual_novel_classes:
            base_set = set(my_manual_base_classes)
            novel_set = set(my_manual_novel_classes)
            if base_set & novel_set:
                overlap = sorted(list(base_set & novel_set))
                raise ValueError(f"Overlap between manual base and novel classes: {overlap}")
        elif my_manual_base_classes:
            base_set = set(my_manual_base_classes)
            novel_set = name_set - base_set
        else:
            novel_set = set(my_manual_novel_classes)
            base_set = name_set - novel_set
    elif my_use_novel_split:
        if my_base_class_count <= 0 or my_base_class_count >= len(all_classes):
            raise ValueError(f"my_base_class_count must be in [1, {len(all_classes) - 1}], got {my_base_class_count}")
        base_set = set(all_classes[:my_base_class_count])
        novel_set = set(all_classes[my_base_class_count:])
    else:
        base_set = set(all_classes)
        novel_set = set()  # no novel split

    # Basic validations for n_way constraints
    if len(base_set) < my_n_way:
        raise ValueError(f"Training n_way={my_n_way} exceeds number of base classes {len(base_set)}. Base classes: {sorted(list(base_set))}")
    if novel_set and my_n_way > len(novel_set):
        print(f"Warning: Evaluation n_way={my_n_way} exceeds novel classes {len(novel_set)}; reduce n_way or adjust base/novel sets.")

    print(f"Base classes ({len(base_set)}): {sorted(list(base_set))}")
    if novel_set:
        print(f"Novel classes ({len(novel_set)}): {sorted(list(novel_set))}")
    else:
        print("Novel split disabled (evaluation will use all classes).")

    # Create datasets with allowed class filters
    train_dataset = c.ConfigDataset(root_dir=train_root, transform=train_transform, allowed_classes=sorted(list(base_set)))
    if novel_set:
        eval_dataset  = c.ConfigDataset(root_dir=eval_root,  transform=eval_transform, allowed_classes=sorted(list(novel_set)))
    else:
        eval_dataset  = c.ConfigDataset(root_dir=eval_root,  transform=eval_transform)
else:
    raise ValueError("Dataset doesn't exist!")

train_sampler = TaskSampler(train_dataset, n_way=my_n_way, n_shot=my_n_shot, n_query=my_n_query, n_tasks=my_episodes)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=True,
                          collate_fn=train_sampler.episodic_collate_fn)

eval_sampler = TaskSampler(eval_dataset, n_way=my_n_way, n_shot=my_n_shot, n_query=my_n_query, n_tasks=my_eva_tasks)
eval_loader = DataLoader(eval_dataset, batch_sampler=eval_sampler, num_workers=0, pin_memory=True,
                         collate_fn=eval_sampler.episodic_collate_fn)

# Instantiate models
if my_backbone == 'resnet18':
    feature_extractor = backbone.ModifiedResNet18().cuda()
elif my_backbone == 'resnet50':
    feature_extractor = backbone.ModifiedResNet50().cuda()
elif my_backbone == 'efficient_net':
    # Prefer new torchvision weights API with fallback
    try:
        from torchvision.models import EfficientNet_B0_Weights  # type: ignore
        feature_extractor = backbone.HookedFeatureExtractor(efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)).cuda()
    except Exception:
        feature_extractor = backbone.HookedFeatureExtractor(efficientnet_b0(pretrained=True)).cuda()
else:
    raise ValueError("Backbone doesn't exist!")

# Control backbone training state
if my_finetune_backbone:
    feature_extractor.train()
else:
    for p in feature_extractor.parameters():
        p.requires_grad = False
    feature_extractor.eval()

num_features = feature_extractor.get_feature_size()
print('Number of features: ', num_features)

# Instantiate selector
if my_selector == 'rl':
    feature_selector = c.RLAgent(
        train_mode=my_rl_train_mode if my_rl_train_mode in ['st_topk', 'bern'] else 'bern',
        eval_mode=my_selector_eval_mode,
        threshold=my_selector_threshold,
        top_p=my_selector_top_p,
        total_steps=my_episodes,  # schedule horizon for entropy/budget
    ).cuda()
    print(f"RLAgent config: train_mode={feature_selector.train_mode}, eval_mode={feature_selector.eval_mode}, threshold={feature_selector.threshold}, top_p={feature_selector.top_p}, k_target={feature_selector.k_target}")
    print("Note: during training (train=True), masks are sampled Bernoulli(p); my_selector_threshold applies only at evaluation (train=False).")
else:
    feature_selector = c.FeatureSelectionDQN(num_features, my_fs_hs).cuda()

proto_network = c.PrototypicalNetworks(
    feature_extractor,
    metric=my_proto_metric,
    temperature=my_proto_temperature
).cuda()

# Training setup
param_groups = [
    {'params': feature_selector.parameters(), 'lr': my_lr},
]
if my_finetune_backbone:
    param_groups.append({'params': feature_extractor.parameters(), 'lr': my_lr * my_backbone_lr_scale})

optimizer = optim.Adam(param_groups, lr=my_lr)
criterion = torch.nn.CrossEntropyLoss()


def fit(support_features, support_labels, query_features, query_labels, feature_mask):
    optimizer.zero_grad()

    # Manual Top-K removed: use all features in legacy path
    support_features_selected = support_features
    query_features_selected = query_features

    # Optional normalization after selection, before prototypical distance
    if my_post_selection_norm == 'l2':
        support_features_selected = F.normalize(support_features_selected, p=2, dim=1)
        query_features_selected = F.normalize(query_features_selected, p=2, dim=1)

    # Use the selected features for classification
    classification_scores = proto_network(support_features_selected, support_labels, query_features_selected)
    loss = criterion(classification_scores, query_labels)
    loss.backward()
    optimizer.step()

    return loss.item()


log_update_frequency = 10
eval_frequency = 100


def main():
    c.print_constant(my_n_way, my_n_shot, my_n_query, my_episodes, my_eva_tasks, IMAGE_SIZE, my_lr, my_fs_hs)

    proto_network.train()
    feature_selector.train()
    all_loss = []
    all_accuracies = []
    total_time = 0

    with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
        for episode_index, (support_images, support_labels, query_images, query_labels, _) in tqdm_train:
            start_time = time.time()
            support_images = support_images.cuda()
            support_labels = support_labels.cuda()
            query_images = query_images.cuda()
            query_labels = query_labels.cuda()

            # Extract flat features and maps (for RLAgent)
            if hasattr(feature_extractor, "forward_with_maps"):  # ResNet path
                support_flat, s_maps, s_names = feature_extractor.forward_with_maps(support_images)
                query_flat, q_maps, q_names = feature_extractor.forward_with_maps(query_images)
                assert s_names == q_names
            elif hasattr(feature_extractor, "maps_and_flat"):     # EfficientNet path
                support_flat, s_maps, s_names = feature_extractor.maps_and_flat(support_images)
                query_flat, q_maps, q_names = feature_extractor.maps_and_flat(query_images)
                assert s_names == q_names
            else:  # Fallback legacy (no maps)
                support_flat = feature_extractor(support_images)
                query_flat = feature_extractor(query_images)
                s_maps = None

            # Optional feature normalization before selection/classification
            if my_feature_norm == 'l2':
                support_flat = F.normalize(support_flat, p=2, dim=1)
                query_flat = F.normalize(query_flat, p=2, dim=1)

            optimizer.zero_grad()

            if my_selector == 'rl' and s_maps is not None:
                # Build task context from support only
                ctx = c.build_task_context(support_flat, support_labels)
                # Sample mask for the task (stochastic during training)
                mask_flat, _ = feature_selector.sample_task_mask(s_maps, s_names, ctx, train=True)
                selected_k = int(mask_flat.sum().item())
                print(f"Selected features this episode (train): {selected_k}")

                # Apply mask to both support and query features
                support_selected = support_flat * mask_flat
                query_selected = query_flat * mask_flat

                # Optional post-selection normalization
                if my_post_selection_norm == 'l2':
                    support_selected = F.normalize(support_selected, p=2, dim=1)
                    query_selected = F.normalize(query_selected, p=2, dim=1)

                # Prototypical classification and CE loss
                classification_scores = proto_network(support_selected, support_labels, query_selected)
                ce_loss = criterion(classification_scores, query_labels)

                # RL losses with reward = -CE(query)
                actor_loss, critic_loss, entropy_loss, budget_loss, _ = feature_selector.rl_losses(ctx, reward_scalar=-ce_loss.detach().item())

                total_loss = ce_loss + actor_loss + critic_loss + entropy_loss + budget_loss
                total_loss.backward()
                optimizer.step()

                loss = total_loss.item()
            else:
                # Legacy path: dense mask + hard Top-K
                mean_support_features = support_flat.mean(dim=0).unsqueeze(0)
                feature_mask = feature_selector(mean_support_features)

                # Manual Top-K removed; use all features in legacy path
                support_features_selected = support_flat
                query_features_selected = query_flat

                if my_post_selection_norm == 'l2':
                    support_features_selected = F.normalize(support_features_selected, p=2, dim=1)
                    query_features_selected = F.normalize(query_features_selected, p=2, dim=1)

                classification_scores = proto_network(support_features_selected, support_labels, query_features_selected)
                ce_loss = criterion(classification_scores, query_labels)
                ce_loss.backward()
                optimizer.step()

                loss = ce_loss.item()

            end_time = time.time()
            episode_time = end_time - start_time
            total_time += episode_time

            all_loss.append(loss)

            if episode_index % log_update_frequency == 0:
                average_loss = sum(all_loss[-log_update_frequency:]) / log_update_frequency
                tqdm_train.set_postfix(loss=average_loss)

            if episode_index % eval_frequency == 0 and episode_index > 0:
                results = c.evaluate(
                    eval_loader,
                    proto_network,
                    feature_selector,
                    feature_extractor,
                    feature_norm=my_feature_norm,
                    post_selection_norm=my_post_selection_norm
                )
                all_accuracies.append(results['accuracy'])

    torch.save(proto_network.state_dict(), "proto_network_uc_100k.pth")
    torch.save(feature_selector.state_dict(), "feature_selector_uc_100k.pth")

    c.evaluate(
        eval_loader,
        proto_network,
        feature_selector,
        feature_extractor,
        feature_norm=my_feature_norm,
        post_selection_norm=my_post_selection_norm
    )
    c.plot_loss_curve(all_loss)
    average_time_per_episode = total_time / len(train_loader)
    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f'Average training time per episode: {average_time_per_episode:.4f} seconds')
    print(f'Maximum memory allocated during training: {max_memory_allocated:.2f} MB')


if __name__ == "__main__":
    main()
    print(f'Allocated memory: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MB')

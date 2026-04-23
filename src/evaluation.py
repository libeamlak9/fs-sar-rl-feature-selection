import torch
from torchvision.models import efficientnet_b0
import time
import os
import psutil
import common_functions as c
import backbone
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import random
from typing import Dict, Iterator, List, Tuple, Union
from torchvision import transforms
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

def _load_rlagent_state_safely(agent, state):
    """
    Load RLAgent weights but skip any keys whose tensor shapes don't match
    the current agent (e.g., when backbone differs between train/eval).
    """
    agent_sd = agent.state_dict()
    filtered = {}
    skipped = []
    for k, v in state.items():
        if k in agent_sd and agent_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)
    missing, unexpected = agent.load_state_dict(filtered, strict=False)
    if skipped:
        print(f"RLAgent: skipped {len(skipped)} keys with mismatched shapes.")
    if len(unexpected) > 0:
        print(f"RLAgent: unexpected keys: {list(unexpected)}")
    if len(missing) > 0:
        print(f"RLAgent: missing keys after load (expected if some heads differ): {list(missing)}")

# Set your parameters (these should match your training setup)
my_data = 'MSTAR_10_Classes'  # 'UCMerced_LandUse' or 'nwpu_resisc45' or 'MSTAR_10_Classes'
my_backbone = 'resnet50'  # 'resnet18', 'resnet50', or 'efficient_net'
my_n_way = 3
my_n_shot = 1
my_n_query = 10
my_eva_tasks = 100
my_fs_hs = 256
IMAGE_SIZE = 224

# Feature vector pre-processing before selection/classification
# Options: 'none', 'l2'
my_feature_norm = 'none'         # match training (agent sees raw)

# Normalization after selection before prototypical distance computation
# Options: 'none', 'l2'
my_post_selection_norm = 'l2'    # stable metric space, matches training recommendation

# Prototypical head metric options (match training changes)
my_proto_metric = 'euclidean'   # 'euclidean' | 'cosine'
my_proto_temperature = 0.7

# Selector type for evaluation: 'rl' (RLAgent) or 'mlp' (legacy)
my_selector = 'rl'

# RL Agent evaluation mode (deterministic selection)
# 'threshold' (variable K, no target), 'threshold_calibrated' (≈ K_target), 'top_p' (variable K), 'topk' (exact K)
my_selector_eval_mode = 'threshold'  # let RL decide via probability threshold (no fixed K target)
my_selector_threshold = 0.7
my_selector_top_p = 1

# Selector type for evaluation: 'rl' (RLAgent) or 'mlp' (legacy)
my_selector = 'rl'

# (duplicate normalization block removed)

# Normalization stats (override ImageNet if you compute SAR-specific mean/std after Grayscale->3ch replication)
my_norm_mean = [0.485, 0.456, 0.406]
my_norm_std  = [0.229, 0.224, 0.225]

# Novel-class evaluation configuration
# For this run, match training (training used all 11 classes)
my_use_novel_split = True
# If you later want novel-class eval, set the following and provide lists/counts:
# my_use_novel_split = True
my_base_class_count = 7
# my_manual_base_classes = [...]
# my_manual_novel_classes = [...]

# Define empty manual class lists by default so downstream checks are safe
# my_manual_base_classes = ['2S1', 'D7', 'BRDM2', 'BMP2', 'BTR60', 'BTR70', 'T62']
# my_manual_novel_classes = ['T72', 'ZIL131', 'ZSU_23_4']
my_manual_base_classes = []
my_manual_novel_classes = []
# Deterministic evaluation transform (no augmentation)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=my_norm_mean, std=my_norm_std)
])

# Load the evaluation dataset
if my_data == 'UCMerced_LandUse':
    eval_dataset = c.ConfigDataset(root_dir=r"UCMerced_LandUse\\Images\\Evaluation_classes", transform=transform)
elif my_data == 'nwpu_resisc45':
    eval_dataset = c.ConfigDataset(root_dir=r"nwpu_resisc45\\test", transform=transform)
elif my_data == 'MSTAR_10_Classes':
    # Support both "MSTAR_10_Classes" and "MSTAR-10-Classes" at project root
    base_candidates = [r"MSTAR_10_Classes", r"MSTAR-10-Classes"]
    base = next((b for b in base_candidates if os.path.isdir(b)), None)
    if base is None:
        raise ValueError(f"MSTAR folder not found. Looked for: {base_candidates}")

    # Resolve train/eval roots
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

    print(f"Using MSTAR eval path -> {eval_root} (train root for split reference: {train_root})")

    # Derive base/novel classes deterministically from train_root; ignore hidden/system dirs (e.g., .ipynb_checkpoints)
    all_classes = sorted([
        d for d in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, d)) and not d.startswith('.')
    ])

    # Resolve base/novel sets using manual overrides or count-based split (mirrors main.py)
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

    # n_way validation for evaluation on novel classes
    if novel_set and my_n_way > len(novel_set):
        print(f"Warning: Evaluation n_way={my_n_way} exceeds novel classes {len(novel_set)}; reduce n_way or adjust class sets.")

    if novel_set:
        print(f"Evaluation on novel classes ({len(novel_set)}): {sorted(list(novel_set))}")
        eval_dataset = c.ConfigDataset(root_dir=eval_root, transform=transform, allowed_classes=sorted(list(novel_set)))
    else:
        print("Novel split disabled (evaluation will use all classes).")
        eval_dataset = c.ConfigDataset(root_dir=eval_root, transform=transform)
else:
    raise ValueError("Dataset doesn't exist!")

eval_sampler = TaskSampler(eval_dataset, n_way=my_n_way, n_shot=my_n_shot, n_query=my_n_query, n_tasks=my_eva_tasks)
eval_loader = DataLoader(eval_dataset, batch_sampler=eval_sampler, num_workers=0, pin_memory=True, collate_fn=eval_sampler.episodic_collate_fn)

def load_and_evaluate(proto_network_path, feature_selector_path, eval_loader, backbone_model, feature_selector_hidden_size):
    # Instantiate the models
    if backbone_model == 'resnet18':
        feature_extractor = backbone.ModifiedResNet18().cuda()
    elif backbone_model == 'resnet50':
        feature_extractor = backbone.ModifiedResNet50().cuda()
    elif backbone_model == 'efficient_net':
        # Prefer new torchvision weights API with fallback
        try:
            from torchvision.models import EfficientNet_B0_Weights  # type: ignore
            original_efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        except Exception:
            original_efficientnet = efficientnet_b0(pretrained=True)
        feature_extractor = backbone.HookedFeatureExtractor(original_efficientnet).cuda()
    else:
        raise ValueError("Backbone model doesn't exist!")

    # Freeze backbone for evaluation (backbone should be trained during training only)
    for p in feature_extractor.parameters():
        p.requires_grad = False
    feature_extractor.eval()

    # Initialize the Prototypical Network and Feature Selector
    proto_network = c.PrototypicalNetworks(
        feature_extractor,
        metric=my_proto_metric,
        temperature=my_proto_temperature
    ).cuda()

    # Instantiate selector based on config
    if my_selector == 'rl':
        feature_selector = c.RLAgent(
            eval_mode=my_selector_eval_mode,
            threshold=my_selector_threshold,
            top_p=my_selector_top_p,
            total_steps=my_eva_tasks
        ).cuda()

        # Warm-up heads to create per-map actor heads before loading weights
        with torch.no_grad():
            dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).cuda()
            if hasattr(feature_extractor, "forward_with_maps"):
                _, maps, names = feature_extractor.forward_with_maps(dummy)
            elif hasattr(feature_extractor, "maps_and_flat"):
                _, maps, names = feature_extractor.maps_and_flat(dummy)
            else:
                maps, names = None, None
            if maps is not None:
                ctx = torch.zeros(5, device=dummy.device)
                _ = feature_selector.sample_task_mask(maps, names, ctx, train=False)

        # Load RLAgent weights (heads, critic, schedules state)
        state = torch.load(feature_selector_path)
        missing, unexpected = feature_selector.load_state_dict(state, strict=False)
        if len(unexpected) > 0:
            print(f"Warning: Unexpected keys in RLAgent state_dict: {unexpected}")
        if len(missing) > 0:
            print(f"Warning: Missing keys in RLAgent state_dict (may be due to differing hook names): {missing}")
    else:
        # Legacy MLP selector
        feature_selector = c.FeatureSelectionDQN(feature_extractor.get_feature_size(), feature_selector_hidden_size).cuda()
        feature_selector.load_state_dict(torch.load(feature_selector_path))

    # Load the saved model weights
    proto_network.load_state_dict(torch.load(proto_network_path))
    # feature_selector state is loaded above for both RL and MLP paths

    # Run evaluation and track time, memory, and loss
    start_time = time.time()
    results = c.evaluate(
        eval_loader,
        proto_network,
        feature_selector,
        feature_extractor,
        feature_norm=my_feature_norm,
        post_selection_norm=my_post_selection_norm,
        save_confusion_matrix_path="confusion_matrix_eval.png",
        cm_normalize='true'
    )
    end_time = time.time()

    time_taken = end_time - start_time
    memory_used = psutil.Process().memory_info().rss / (1024 ** 2)  # Memory in MB

    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"{'='*50}")
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"Min Episode Accuracy: {results['min_accuracy']:.2f}%")
    print(f"Max Episode Accuracy: {results['max_accuracy']:.2f}%")
    print(f"Macro F1 Score: {results['f1_score']:.4f}")
    print(f"Avg Selected Features: {results['avg_selected_features']}")
    print(f"{'='*50}")
    print(f"Time Taken: {time_taken:.4f} seconds")
    print(f"Process RSS (CPU) Memory Used: {memory_used:.2f} MB")

    # Report GPU metrics if CUDA is available
    try:
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            total_vram_mb = props.total_memory / (1024 ** 2)
            mem_alloc_mb = torch.cuda.memory_allocated(idx) / (1024 ** 2)
            mem_reserved_mb = torch.cuda.memory_reserved(idx) / (1024 ** 2)
            max_alloc_mb = torch.cuda.max_memory_allocated(idx) / (1024 ** 2)
            max_reserved_mb = torch.cuda.max_memory_reserved(idx) / (1024 ** 2)
            capability = f"{props.major}.{props.minor}"
            sms = getattr(props, 'multi_processor_count', 0)
            print("GPU Device Summary:")
            print(f"  GPU[{idx}]: {name} | Compute Capability {capability} | SMs: {sms}")
            print(f"  Total VRAM: {total_vram_mb:.2f} MB")
            print(f"  CUDA Memory (current): allocated={mem_alloc_mb:.2f} MB, reserved={mem_reserved_mb:.2f} MB")
            print(f"  CUDA Memory (peak):    max_allocated={max_alloc_mb:.2f} MB, max_reserved={max_reserved_mb:.2f} MB")
            # Try to get utilization via nvidia-smi
            try:
                import subprocess
                cmd = [
                    'nvidia-smi',
                    '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used',
                    '--format=csv,noheader,nounits'
                ]
                out = subprocess.check_output(cmd).decode().strip().splitlines()
                util_line = None
                for line in out:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6 and parts[0].isdigit() and int(parts[0]) == idx:
                        util_line = parts
                        break
                if util_line is not None:
                    _, gpu_name, util_gpu, util_mem, mem_total, mem_used = util_line[:6]
                    print(f"  nvidia-smi: util.gpu={util_gpu}% | util.mem={util_mem}% | mem.used={mem_used} MB / {mem_total} MB")
            except Exception:
                pass
        else:
            print("CUDA not available; GPU metrics skipped.")
    except Exception as e:
        print(f"GPU metrics reporting failed: {e}")

if __name__ == "__main__":
    load_and_evaluate("proto_network_uc_100k.pth", "feature_selector_uc_100k.pth", eval_loader, my_backbone, my_fs_hs)

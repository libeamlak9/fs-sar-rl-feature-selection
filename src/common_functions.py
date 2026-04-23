import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import tqdm

# Allowed image extensions
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def print_constant(n_way=5, n_shot=5, n_query=10, n_training_episodes=10000,
                   n_evaluation_tasks=100, image_size=224, learning_rate=0.0001, feature_selector_hidden_size=256):
    # Prints the hyperparameters used in training/evaluation
    params = {
        'N_WAY': n_way,
        'N_SHOT': n_shot,
        'N_QUERY': n_query,
        'N_TRAINING_EPISODES': n_training_episodes,
        'N_EVALUATION_TASKS': n_evaluation_tasks,
        'IMAGE_SIZE': image_size,
        'Learning rate': learning_rate,
        'FEATURE_SELECTOR_HIDDEN_SIZE': feature_selector_hidden_size,
        'FEATURE_SELECTION': 'RL (auto)',  # no manual Top-K
    }

    print('Training/Evaluation hyperparameters\n')
    for key, value in params.items():
        print(f'{key}: {value}')

class ConfigDataset(Dataset):
    # Dataset configuration class for UCMerced LandUse or similar datasets
    def __init__(self, root_dir, transform=None, allowed_classes=None):
        self.root_dir = root_dir
        self.transform = transform
        all_classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        all_classes.sort()
        if allowed_classes is not None:
            allowed_set = set(allowed_classes)
            self.classes = [d for d in all_classes if d in allowed_set]
        else:
            self.classes = all_classes
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            # Support nested "images" subfolder per class (e.g., class_name/images/*.png)
            candidate = os.path.join(cls_path, "images")
            img_dir = candidate if os.path.isdir(candidate) else cls_path
            for img in os.listdir(img_dir):
                img_path = os.path.join(img_dir, img)
                ext = os.path.splitext(img_path)[1].lower()
                if os.path.isfile(img_path) and ext in IMAGE_EXTS:
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # Open as-is to preserve SAR intensity; transforms will handle grayscale->3ch
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_labels(self):
        return [s[1] for s in self.samples]

    def get_class_names(self):
        return list(self.classes)

# -------------------------
# RL Agent: Actor–Critic with Bernoulli gates and schedules
# -------------------------

class RLActorHead(nn.Module):
    """
    Per-map lightweight head that outputs per-element logits with depthwise conv for spatial variation.
    It is created lazily for each map name with the observed channel count.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # Depthwise conv to get spatially varying signals with tiny params
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=True)
        # Pointwise mixing within channels
        self.pw = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x):
        # x: (B=1, C, H, W)
        y = self.dw(x)
        y = F.relu_(y)
        y = self.pw(y)
        return y  # logits per element, same shape as x


class RLCritic(nn.Module):
    """
    Critic over task context vector c
    """
    def __init__(self, ctx_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, c):
        return self.net(c).squeeze(-1)  # (batch,) but we use batch=1


def build_task_context(support_flat: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
    """
    Build a compact task context vector from support features (unmasked, pre-selection)
    support_flat: (Ns, D) float
    support_labels: (Ns,) long with labels in 0..n_way-1
    Returns a 1D tensor context vector on same device
    """
    device = support_flat.device
    uniq = torch.unique(support_labels)
    # Per-class prototypes
    protos = []
    intra_vars = []
    for l in uniq:
        feats = support_flat[support_labels == l]
        mu = feats.mean(dim=0)
        protos.append(mu)
        if feats.shape[0] > 1:
            intra_vars.append(feats.var(dim=0, unbiased=False).mean())
        else:
            intra_vars.append(torch.tensor(0.0, device=device))
    protos = torch.stack(protos, dim=0)  # (n_way, D)
    # Normalize prototypes for distance summaries
    protos_n = F.normalize(protos, p=2, dim=1)
    # Pairwise cosine distances
    dists = 1.0 - protos_n @ protos_n.t()
    iu = torch.triu_indices(dists.size(0), dists.size(1), offset=1, device=device)
    pd = dists[iu[0], iu[1]]
    # Summaries
    proto_mean = protos.mean(dim=0)
    proto_std = protos.std(dim=0)
    ctx = torch.stack([
        proto_mean.mean(),            # scalar
        proto_std.mean(),             # scalar
        pd.mean() if pd.numel() > 0 else torch.tensor(0.0, device=device),  # mean inter-class dist
        torch.min(pd) if pd.numel() > 0 else torch.tensor(0.0, device=device),  # min inter-class dist
        torch.stack(intra_vars).mean() if len(intra_vars) > 0 else torch.tensor(0.0, device=device),  # mean intra
    ])
    return ctx.detach()  # treat as state, do not backprop through context by default


class RLAgent(nn.Module):
    """
    Actor–Critic agent that produces a binary mask for all features using shared per-map heads.
    - Training: Bernoulli sampling per element (exploration), policy gradient with entropy and budget penalty
    - Evaluation: deterministic selection via 'threshold' | 'top_p' | 'topk'
    The same task-specific mask is applied to all support/query examples in the episode.
    """
    def __init__(
        self,
        k_target=None,
        train_mode: str = "bern",  # default to Bernoulli REINFORCE (no fixed K)
        eval_mode: str = "threshold",
        threshold: float = 0.5,
        top_p: float = 0.9,
        entropy_alpha_start: float = 0.1,
        entropy_alpha_end: float = 0.0,
        budget_lambda_start: float = 0.0,
        budget_lambda_end: float = 1.0,
        total_steps: int = 1000,
        ctx_dim: int = 5,
    ):
        super().__init__()
        self.k_target = k_target
        self.train_mode = train_mode
        self.eval_mode = eval_mode
        self.threshold = threshold
        self.top_p = top_p
        # Schedules
        self.entropy_alpha_start = entropy_alpha_start
        self.entropy_alpha_end = entropy_alpha_end
        self.budget_lambda_start = budget_lambda_start
        self.budget_lambda_end = budget_lambda_end
        self.total_steps = max(1, total_steps)
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

        self.heads = nn.ModuleDict()   # map_name -> RLActorHead
        self.critic = RLCritic(ctx_dim=ctx_dim, hidden=128)

        # Buffers for last forward (to compute losses)
        self._last_p = None
        self._last_z = None
        self._last_logprob = None
        self._last_expected_k = None

    def _alpha(self):
        # entropy coefficient schedule (linear)
        t = min(self.global_step.item(), self.total_steps)
        a0, a1 = self.entropy_alpha_start, self.entropy_alpha_end
        return a0 + (a1 - a0) * (t / self.total_steps)

    def _lambda(self):
        # budget penalty schedule (linear)
        t = min(self.global_step.item(), self.total_steps)
        l0, l1 = self.budget_lambda_start, self.budget_lambda_end
        return l0 + (l1 - l0) * (t / self.total_steps)

    def _ensure_head(self, name: str, in_ch: int):
        if name not in self.heads:
            head = RLActorHead(in_ch)
            # Ensure newly created heads are on the same device as the agent (created post .cuda())
            try:
                device = next(self.critic.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            head = head.to(device)
            self.heads[name] = head

    @staticmethod
    def _flatten_maps(maps):
        parts = [m.view(1, -1) for m in maps]  # maps are (1,C,H,W)
        return torch.cat(parts, dim=1)  # (1, N)

    def _bernoulli_select_eval(self, p_flat: torch.Tensor, mode: str, k_target):
        # p_flat: (1, N)
        # If mode requires a target K but none is provided, fall back to threshold
        if mode in ("threshold_calibrated", "topk") and (k_target is None or (isinstance(k_target, int) and k_target <= 0)):
            mode = "threshold"

        if mode == "threshold":
            mask = (p_flat >= self.threshold).float()
        elif mode == "threshold_calibrated":
            # choose tau so that selected count approximates k_target
            k = int(min(max(1, k_target), p_flat.numel()))
            vals, idx = torch.topk(p_flat[0], k)
            tau = vals.min()
            mask = (p_flat >= tau).float()
        elif mode == "top_p":
            probs, idx = torch.sort(p_flat[0], descending=True)
            cum = torch.cumsum(probs, dim=0)
            cutoff = (cum <= self.top_p).float()
            k = int(cutoff.sum().item())
            sel = idx[:max(k, 1)]
            mask = torch.zeros_like(p_flat)
            mask[0, sel] = 1.0
        elif mode == "topk":
            k = min(int(k_target), p_flat.numel())
            vals, idx = torch.topk(p_flat[0], k)
            mask = torch.zeros_like(p_flat)
            mask[0, idx] = 1.0
        else:
            # default to threshold
            mask = (p_flat >= self.threshold).float()
        return mask

    def sample_task_mask(self, support_maps, names, context_vec, train: bool = True):
        """
        support_maps: list of tensors (Ns, C, H, W) or (1, C, H, W) if already averaged
        names: list of strings aligned to support_maps
        context_vec: (ctx_dim,) tensor
        Returns: mask_flat (1, N), p_flat (1, N)
        """
        device = support_maps[0].device
        # Average across support batch dimension to create a single task map per layer
        avg_maps = []
        for m in support_maps:
            if m.dim() != 4:
                raise RuntimeError("Expected 4D map (B,C,H,W)")
            avg_maps.append(m.mean(dim=0, keepdim=True))  # (1,C,H,W)

        # Build logits per map
        logits_maps = []
        p_maps = []
        for m, name in zip(avg_maps, names):
            self._ensure_head(name, m.shape[1])
            logits = self.heads[name](m)  # (1,C,H,W)
            p = torch.sigmoid(logits)
            logits_maps.append(logits)
            p_maps.append(p)

        # Flatten
        p_flat = self._flatten_maps(p_maps)  # (1, N)

        if train:
            if self.train_mode == "st_topk" and self.k_target is not None:
                # Straight-through TopK: forward hard TopK, backward through probabilities
                k = int(min(max(1, self.k_target), p_flat.numel()))
                vals, idx = torch.topk(p_flat[0], k)
                hard = torch.zeros_like(p_flat)
                hard[0, idx] = 1.0
                st_mask = hard + p_flat - p_flat.detach()
                self._last_p = p_flat  # keep gradient path
                self._last_z = hard.detach()
                self._last_logprob = None
                self._last_expected_k = p_flat.sum().detach()
                mask = st_mask
            else:
                # Bernoulli sampling per element (REINFORCE)
                z = torch.bernoulli(p_flat)  # (1, N)
                # Log prob of Bernoulli selections
                eps = 1e-8
                logprob = z * torch.log(p_flat + eps) + (1.0 - z) * torch.log(1.0 - p_flat + eps)
                expected_k = p_flat.sum()
                # Cache for loss
                self._last_p = p_flat.detach()
                self._last_z = z.detach()
                self._last_logprob = logprob
                self._last_expected_k = expected_k.detach()
                mask = z
        else:
            # Deterministic selection at eval (default threshold/top_p/topk)
            mask = self._bernoulli_select_eval(p_flat, self.eval_mode, self.k_target)
            self._last_p = p_flat.detach()
            self._last_z = mask.detach()
            self._last_logprob = None
            self._last_expected_k = p_flat.sum().detach()

        return mask, p_flat

    def rl_losses(self, context_vec, reward_scalar: float):
        """
        Compute policy and critic losses using the cached sampling info.
        For train_mode == 'st_topk', actor/critic losses are zero (learning flows via CE through ST mask);
        we still apply entropy and budget regularizers on probabilities.
        """
        assert self._last_p is not None, "sample_task_mask must be called before rl_losses"
        device = self._last_p.device
        r = torch.as_tensor(reward_scalar, device=device, dtype=torch.float32)

        # Critic and actor losses
        if self.train_mode == "st_topk":
            actor_loss = torch.tensor(0.0, device=device)
            critic_loss = torch.tensor(0.0, device=device)
        else:
            V = self.critic(context_vec.unsqueeze(0))  # (1,)
            advantage = (r - V.detach())
            if self._last_logprob is not None:
                logprob_mean = self._last_logprob.mean()
                actor_loss = -(advantage * logprob_mean)
            else:
                actor_loss = torch.tensor(0.0, device=device)
            critic_loss = F.mse_loss(V, r.detach().unsqueeze(0))

        # Entropy regularization (over Bernoulli probs)
        p = torch.clamp(self._last_p, 1e-6, 1 - 1e-6)
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        entropy_mean = entropy.mean()
        entropy_alpha = self._alpha()
        entropy_loss = -entropy_alpha * entropy_mean

        # Budget penalty on expected-K
        expected_k = self._last_p.sum()
        if self.k_target is not None and self.k_target > 0:
            target = torch.tensor(float(self.k_target), device=device)
            diff = (expected_k - target) / (target + 1e-6)
            lam = self._lambda()
            budget_loss = lam * (diff * diff)
        else:
            budget_loss = torch.tensor(0.0, device=device)

        # Step schedule
        self.global_step += 1

        # Diagnostics
        info = {
            "entropy": entropy_mean.detach().item(),
            "expected_k": expected_k.detach().item(),
        }
        if self.train_mode != "st_topk":
            info["adv"] = float((r - self.critic(context_vec.unsqueeze(0)).detach()).item())

        return actor_loss, critic_loss, entropy_loss, budget_loss, info

def select_top_features(features, feature_importance, top_k):
    # Selects the top-k important features from the feature_importance mask
    _, top_indices = torch.topk(feature_importance.squeeze(), top_k)
    return features[:, top_indices]

class PrototypicalNetworks(nn.Module):
    """
    Prototypical head with configurable metric and temperature.
    metric: 'euclidean' | 'cosine'
    temperature: positive float; logits = -distance / temperature
    """
    def __init__(self, backbone: nn.Module, metric: str = "euclidean", temperature: float = 1.0):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone
        self.metric = metric
        self.temperature = temperature

    def forward(self, support_features, support_labels, query_features):
        unique_labels = torch.unique(support_labels)
        # Compute class prototypes
        z_proto = torch.stack([support_features[support_labels == l].mean(0) for l in unique_labels])  # (C,D)

        if self.metric == "cosine":
            q = F.normalize(query_features, p=2, dim=1)
            p = F.normalize(z_proto, p=2, dim=1)
            # cosine distance = 1 - cosine similarity
            dists = 1.0 - (q @ p.t()).clamp(-1, 1)
        else:
            dists = torch.cdist(query_features, z_proto)

        logits = -dists / max(1e-6, self.temperature)
        return logits

def evaluate_on_one_task(support_features, support_labels, query_features, query_labels, proto_network):
    # Evaluates the model on a single task and returns predictions
    classification_scores = proto_network(support_features, support_labels, query_features)
    preds = torch.max(classification_scores.detach(), 1)[1]
    correct = (preds == query_labels).sum().item()
    total = len(query_labels)
    return correct, total, preds

def evaluate(data_loader, proto_network, feature_selector, feature_extractor, top_k=None, feature_norm='none', post_selection_norm='none', save_confusion_matrix_path=None, cm_normalize='true'):
    """
    Supports two selectors:
      - Legacy MLP selector producing a dense importance vector (FeatureSelectionDQN)
      - RLAgent producing a task-level mask based on support maps
    Returns dict with accuracy, min_accuracy, max_accuracy, f1_score, and selected features info.
    """
    total_predictions = 0
    correct_predictions = 0
    # Track per-episode accuracies for min/max
    episode_accuracies = []
    # For confusion matrix accumulation
    all_true_ids = []
    all_pred_ids = []
    dataset_class_names = None
    try:
        if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'get_class_names'):
            dataset_class_names = data_loader.dataset.get_class_names()
    except Exception:
        dataset_class_names = None
    proto_network.eval()
    was_training_fs = getattr(feature_selector, "training", False)
    if hasattr(feature_selector, "eval"):
        feature_selector.eval()

    with torch.no_grad():
        total_selected_k = 0
        episodes = 0
        iterator = enumerate(data_loader)
        try:
            total_tasks = len(data_loader)
        except TypeError:
            total_tasks = None

        with tqdm(iterator, total=total_tasks, desc="Evaluating episodes", leave=False) as t:
            for episode_index, (support_images, support_labels, query_images, query_labels, episode_class_ids) in t:
                support_images, support_labels, query_images, query_labels = support_images.cuda(), support_labels.cuda(), query_images.cuda(), query_labels.cuda()
                # episode_class_ids: mapping from episodic labels [0..n_way-1] to dataset-level class ids
                if isinstance(episode_class_ids, torch.Tensor):
                    episode_class_ids = episode_class_ids.cpu().tolist()

                # Extract features and maps
                if hasattr(feature_extractor, "forward_with_maps"):  # ResNet path
                    support_flat, s_maps, s_names = feature_extractor.forward_with_maps(support_images)
                    query_flat, q_maps, q_names = feature_extractor.forward_with_maps(query_images)
                    assert s_names == q_names
                elif hasattr(feature_extractor, "maps_and_flat"):     # EfficientNet path
                    support_flat, s_maps, s_names = feature_extractor.maps_and_flat(support_images)
                    query_flat, q_maps, q_names = feature_extractor.maps_and_flat(query_images)
                    assert s_names == q_names
                else:  # Fallback legacy
                    support_flat = feature_extractor(support_images)
                    query_flat = feature_extractor(query_images)
                    s_maps = None

                # Optional feature normalization before selection/classification
                if feature_norm == 'l2':
                    support_flat = F.normalize(support_flat, p=2, dim=1)
                    query_flat = F.normalize(query_flat, p=2, dim=1)

                # Route by selector type
                if 'RLAgent' in type(feature_selector).__name__ and s_maps is not None:
                    ctx = build_task_context(support_flat, support_labels)
                    # Deterministic selection in evaluation according to agent.eval_mode
                    mask_flat, _ = feature_selector.sample_task_mask(s_maps, s_names, ctx, train=False)
                    selected_k = int(mask_flat.sum().item())
                    print(f"Selected features this episode: {selected_k}")
                    total_selected_k += selected_k
                    episodes += 1
                    support_selected = support_flat * mask_flat
                    query_selected = query_flat * mask_flat
                else:
                    mean_support_features = support_flat.mean(dim=0).unsqueeze(0)
                    feature_mask = feature_selector(mean_support_features)
                    # Legacy hard Top-K path retained for backward compatibility; if top_k is None, fallback to all features
                    if top_k is None:
                        support_selected = support_flat
                        query_selected = query_flat
                    else:
                        support_selected = select_top_features(support_flat, feature_mask, top_k)
                        query_selected = select_top_features(query_flat, feature_mask, top_k)
                    selected_k = support_selected.shape[1]
                    print(f"Selected features this episode (legacy): {selected_k}")
                    total_selected_k += selected_k
                    episodes += 1

                # Optional normalization after selection for prototypical distance computation
                if post_selection_norm == 'l2':
                    support_selected = F.normalize(support_selected, p=2, dim=1)
                    query_selected = F.normalize(query_selected, p=2, dim=1)

                correct, total, preds = evaluate_on_one_task(support_selected, support_labels, query_selected, query_labels, proto_network)
                # Track per-episode accuracy
                episode_acc = 100.0 * correct / total if total > 0 else 0.0
                episode_accuracies.append(episode_acc)
                total_predictions += total
                correct_predictions += correct
                # Accumulate for confusion matrix using dataset-level class ids
                if episode_class_ids is not None:
                    preds_cpu = preds.detach().cpu().tolist()
                    qlabels_cpu = query_labels.detach().cpu().tolist()
                    for p_idx, t_idx in zip(preds_cpu, qlabels_cpu):
                        try:
                            all_pred_ids.append(int(episode_class_ids[p_idx]))
                            all_true_ids.append(int(episode_class_ids[t_idx]))
                        except Exception:
                            # Fallback: skip if mapping unavailable
                            pass
                # Update tqdm postfix with running accuracy
                running_acc = 100.0 * correct_predictions / max(1, total_predictions)
                t.set_postfix(acc=f"{running_acc:.2f}%")

    if hasattr(feature_selector, "train") and was_training_fs:
        feature_selector.train()

    accuracy = 100 * correct_predictions / total_predictions if total_predictions > 0 else 0.0
    min_accuracy = min(episode_accuracies) if episode_accuracies else 0.0
    max_accuracy = max(episode_accuracies) if episode_accuracies else 0.0
    
    print(f"Model tested on {len(data_loader)} tasks. Accuracy: {accuracy:.2f}%")
    print(f"Min Episode Accuracy: {min_accuracy:.2f}%")
    print(f"Max Episode Accuracy: {max_accuracy:.2f}%")
    if episodes > 0:
        avg_selected = int(total_selected_k / episodes)
        print(f"Average selected features per episode: {avg_selected}")

    # Compute per-class accuracy and F1 score from confusion matrix
    f1_score = 0.0
    per_class_accuracy = {}
    cm = None
    C = 0
    class_names = []
    # Build and save confusion matrix if requested
    if len(all_true_ids) > 0 and len(all_true_ids) == len(all_pred_ids):
        try:
            # Determine label space
            if dataset_class_names is not None:
                class_names = list(dataset_class_names)
                C = len(class_names)
                # Assume dataset label ids are 0..C-1 aligned with class_names
                cm = torch.zeros((C, C), dtype=torch.long)
                for t, p in zip(all_true_ids, all_pred_ids):
                    if 0 <= t < C and 0 <= p < C:
                        cm[t, p] += 1
            else:
                # Build label set from observed ids
                labels = sorted(set(all_true_ids) | set(all_pred_ids))
                idx = {lab: i for i, lab in enumerate(labels)}
                C = len(labels)
                class_names = [str(l) for l in labels]
                cm = torch.zeros((C, C), dtype=torch.long)
                for t, p in zip(all_true_ids, all_pred_ids):
                    cm[idx[t], idx[p]] += 1

            # Compute per-class accuracy
            if cm is not None:
                for i in range(C):
                    class_total = cm[i, :].sum().item()
                    class_correct = cm[i, i].item()
                    class_acc = 100.0 * class_correct / class_total if class_total > 0 else 0.0
                    per_class_accuracy[class_names[i]] = class_acc
                
                print("\nPer-Class Accuracy:")
                for name, acc in sorted(per_class_accuracy.items()):
                    print(f"  {name}: {acc:.2f}%")

            # Compute F1 score from confusion matrix
            if cm is not None:
                precision_per_class = []
                recall_per_class = []
                f1_per_class = []
                for i in range(C):
                    tp = cm[i, i].item()
                    fp = cm[:, i].sum().item() - tp
                    fn = cm[i, :].sum().item() - tp
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    
                    precision_per_class.append(precision)
                    recall_per_class.append(recall)
                    f1_per_class.append(f1)
                
                # Macro F1 (average of per-class F1 scores)
                f1_score = sum(f1_per_class) / len(f1_per_class) if f1_per_class else 0.0
                print(f"\nMacro F1 Score: {f1_score:.4f}")
                print(f"Per-class F1: {[f'{f:.3f}' for f in f1_per_class]}")

            # Save confusion matrix plot if requested
            if save_confusion_matrix_path is not None:
                # Normalize if requested
                cm_np = cm.numpy().astype(float)
                if cm_normalize in ('true', 'row'):
                    row_sums = cm_np.sum(axis=1, keepdims=True) + 1e-12
                    cm_disp = cm_np / row_sums
                    fmt = '.2f'
                elif cm_normalize in ('pred', 'col'):
                    col_sums = cm_np.sum(axis=0, keepdims=True) + 1e-12
                    cm_disp = cm_np / col_sums
                    fmt = '.2f'
                elif cm_normalize in ('all', 'global'):
                    total = cm_np.sum() + 1e-12
                    cm_disp = cm_np / total
                    fmt = '.4f'
                else:
                    cm_disp = cm_np
                    fmt = 'd'

                # Plot high-quality confusion matrix
                import matplotlib.pyplot as plt
                fig_size = 8
                plt.figure(figsize=(max(8, 0.6 * C), max(6, 0.6 * C)), dpi=300)
                im = plt.imshow(cm_disp, interpolation='nearest', cmap=plt.cm.Blues, aspect='auto')
                plt.title('Confusion Matrix', fontsize=24)
                plt.colorbar(im, fraction=0.046, pad=0.04)
                tick_marks = list(range(C))
                plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=15)
                plt.yticks(tick_marks, class_names, fontsize=15)

                # Annotate cells (limit if too large)
                annotate = C <= 30
                if annotate:
                    thresh = cm_disp.max() / 2.0 if cm_disp.size > 0 else 0.5
                    for i in range(C):
                        for j in range(C):
                            val = cm_disp[i, j]
                            txt = f"{val:{fmt}}"
                            plt.text(j, i, txt,
                                     horizontalalignment="center",
                                     verticalalignment="center",
                                     color="white" if val > thresh else "black",
                                     fontsize=24)

                plt.tight_layout()
                plt.ylabel('True label', fontsize=18)
                plt.xlabel('Predicted label', fontsize=18)
                plt.savefig(save_confusion_matrix_path, bbox_inches='tight')
                plt.close()
                print(f"Saved confusion matrix to {save_confusion_matrix_path}")
            
        except Exception as e:
            print(f"Failed to save confusion matrix or compute metrics: {e}")

    return {
        'accuracy': accuracy,
        'min_accuracy': min_accuracy,
        'max_accuracy': max_accuracy,
        'f1_score': f1_score,
        'avg_selected_features': int(total_selected_k / episodes) if episodes > 0 else 0,
        'per_class_accuracy': per_class_accuracy,
        'confusion_matrix': cm.numpy() if cm is not None else None,
        'class_names': class_names
    }

def smooth_loss(loss, weight=0.9):
    # Smooths the loss curve for better visualization
    smoothed_loss = []
    last = loss[0]
    for point in loss:
        smoothed_loss.append(last * weight + (1 - weight) * point)
        last = smoothed_loss[-1]
    return smoothed_loss

def plot_loss_curve(all_loss):
    # Plots and saves the training loss curve
    smoothed_all_loss = smooth_loss(all_loss)
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed_all_loss, label="Training Loss")
    plt.xlabel("Training Episodes")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig("Landuse_Resnet18_training_loss_curve.png")
    plt.close()

def plot_accuracy_curve(all_accuracy, eval_frequency, name):
    # Plots and saves the accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(all_accuracy, label="Accuracy")
    plt.xlabel(f"Evaluation every {eval_frequency} Episodes")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    name = name + '.png'
    plt.savefig(name)
    plt.close()

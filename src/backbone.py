from torchvision.models import resnet50, resnet18, efficientnet_b0
import torch
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

IMAGE_SIZE = 224

class ModifiedResNet50(nn.Module):
    def __init__(self):
        super(ModifiedResNet50, self).__init__()
        # Prefer new torchvision weights API with fallback for older versions
        try:
            from torchvision.models import ResNet50_Weights  # type: ignore
            original_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            original_resnet = resnet50(pretrained=True)

        # Use original 3-channel conv1 (suitable for RGB or grayscale replicated to 3 channels)
        self.conv1 = original_resnet.conv1

        self.bn1 = original_resnet.bn1
        self.relu = original_resnet.relu
        self.maxpool = original_resnet.maxpool
        self.layer1 = original_resnet.layer1
        self.layer2 = original_resnet.layer2
        self.layer3 = original_resnet.layer3
        self.layer4 = original_resnet.layer4
        self.avgpool = original_resnet.avgpool
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f1 = self.layer1(x)  # Features from layer1
        f2 = self.layer2(f1)  # Features from layer2
        f3 = self.layer3(f2)  # Features from layer3
        f4 = self.layer4(f3)  # Features from layer4

        # Apply avgpool to the output of layer4 (global pooling)
        pooled = self.avgpool(f4)
        flattened = self.flatten(pooled)  # Flatten pooled features

        # Flatten intermediate features from each layer
        f1_flat = f1.view(f1.size(0), -1)
        f2_flat = f2.view(f2.size(0), -1)
        f3_flat = f3.view(f3.size(0), -1)
        f4_flat = f4.view(f4.size(0), -1)

        # Concatenate features from all layers (f1, f2, f3, f4, and pooled)
        concatenated_features = torch.cat((f1_flat, f2_flat, f3_flat, f4_flat, flattened), dim=1)

        return concatenated_features

    def forward_with_maps(self, x):
        # Same as forward but also returns the intermediate maps and their names, aligned with flatten order
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        pooled = self.avgpool(f4)  # (B, C4, 1, 1)

        # Build maps in the exact concatenation order used in forward()
        maps = [f1, f2, f3, f4, pooled]  # pooled kept as (B, C, 1, 1)
        names = ["layer1", "layer2", "layer3", "layer4", "pooled"]

        # Flatten according to the same order
        f1_flat = f1.view(f1.size(0), -1)
        f2_flat = f2.view(f2.size(0), -1)
        f3_flat = f3.view(f3.size(0), -1)
        f4_flat = f4.view(f4.size(0), -1)
        flattened = self.flatten(pooled)
        concatenated_features = torch.cat((f1_flat, f2_flat, f3_flat, f4_flat, flattened), dim=1)

        return concatenated_features, maps, names

    def get_feature_size(self):
        print("Calculating feature size using a dummy input.")
        device = next(self.parameters()).device
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)  # 3-channel input (RGB or replicated grayscale)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy_output = self.forward(dummy_input)
        if was_training:
            self.train()
        print(f"Feature size: {dummy_output.shape[1]}")
        return dummy_output.shape[1]


class ModifiedResNet18(nn.Module):
    def __init__(self):
        super(ModifiedResNet18, self).__init__()
        # Prefer new torchvision weights API with fallback for older versions
        try:
            from torchvision.models import ResNet18_Weights  # type: ignore
            original_resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            original_resnet = resnet18(pretrained=True)

        self.conv1 = original_resnet.conv1
        self.bn1 = original_resnet.bn1
        self.relu = original_resnet.relu
        self.maxpool = original_resnet.maxpool

        self.layer1 = original_resnet.layer1
        self.layer2 = original_resnet.layer2
        self.layer3 = original_resnet.layer3
        self.layer4 = original_resnet.layer4

        self.avgpool = original_resnet.avgpool
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        pooled = self.avgpool(f4)
        flattened = self.flatten(pooled)

        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)
        f3 = f3.view(f3.size(0), -1)
        f4 = f4.view(f4.size(0), -1)

        concatenated_features = torch.cat((f1, f2, f3, f4, flattened), dim=1)
        # print('Extracted features shape: ', concatenated_features.shape)

        return concatenated_features

    def forward_with_maps(self, x):
        # Same as forward but also returns the intermediate maps and their names, aligned with flatten order
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        pooled = self.avgpool(f4)  # (B, C4, 1, 1)

        maps = [f1, f2, f3, f4, pooled]
        names = ["layer1", "layer2", "layer3", "layer4", "pooled"]

        f1_flat = f1.view(f1.size(0), -1)
        f2_flat = f2.view(f2.size(0), -1)
        f3_flat = f3.view(f3.size(0), -1)
        f4_flat = f4.view(f4.size(0), -1)
        flattened = self.flatten(pooled)
        concatenated_features = torch.cat((f1_flat, f2_flat, f3_flat, f4_flat, flattened), dim=1)

        return concatenated_features, maps, names

    def get_feature_size(self):
        device = next(self.parameters()).device
        dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy_output = self.forward(dummy_input)
        if was_training:
            self.train()
        return dummy_output.shape[1]



class HookedFeatureExtractor(nn.Module):
    def __init__(self, model):
        super(HookedFeatureExtractor, self).__init__()
        self.model = model
        self.features = {}

        # Register hooks across the model to capture rich intermediate features
        self._register_hooks()

    def _register_hooks(self):
        # Broadly capture intermediate features similar to ResNet multi-layer concatenation.
        # - For top-level children: if Sequential (e.g., features), register on each block.
        # - Also go one level deeper when blocks are Sequentials with named children.
        # - Skip the classifier head to avoid logits.
        for name, layer in self.model.named_children():
            if isinstance(layer, nn.Sequential):
                for block_name, block in layer.named_children():
                    # Register on the block itself
                    block.register_forward_hook(self._get_hook(f"{name}_{block_name}"))
                    # Also hook deeper named children if present
                    if isinstance(block, nn.Sequential):
                        for sub_name, sub_block in block.named_children():
                            sub_block.register_forward_hook(self._get_hook(f"{name}_{block_name}_{sub_name}"))
            else:
                if hasattr(self.model, "classifier") and layer is self.model.classifier:
                    # Skip classifier to capture pre-classifier features only
                    continue
                layer.register_forward_hook(self._get_hook(name))

    def _get_hook(self, layer_name):
        def hook(module, input, output):
            self.features[layer_name] = output
        return hook

    def forward(self, x):
        # Clear previous features and run a forward pass to trigger hooks
        self.features = {}
        _ = self.model(x)
        return self.features

    def maps_and_flat(self, x):
        """
        Returns:
          flat: (B, N) concatenated flattened features in deterministic order
          maps: list of feature maps (each 4D tensor B,C,H,W). Tensors that are 2D are expanded to (B,C,1,1)
          names: list of layer names aligned with maps order
        """
        feats = self.forward(x)
        names = sorted(feats.keys())
        maps = []
        for n in names:
            t = feats[n]
            if t.dim() == 2:  # (B, C)
                t = t.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            maps.append(t)
        # Flatten in the same deterministic order
        flat_parts = [m.view(m.size(0), -1) for m in maps]
        flat = torch.cat(flat_parts, dim=1) if len(flat_parts) else torch.empty((x.size(0), 0), device=x.device)
        return flat, maps, names

    def get_feature_size(self, image_size=IMAGE_SIZE):
        device = next(self.parameters()).device
        dummy_input = torch.randn(1, 3, image_size, image_size, device=device)
        features = self(dummy_input)
        concatenated_features = concatenate_features(features)
        return concatenated_features.shape[1]

def concatenate_features(features):
    flattened_features = []
    # Ensure deterministic ordering of concatenation
    for key in sorted(features.keys()):
        feature = features[key]
        if isinstance(feature, torch.Tensor):
            flattened_features.append(feature.view(feature.size(0), -1))
    concatenated_features = torch.cat(flattened_features, dim=1) if flattened_features else torch.empty(0)
    return concatenated_features




"""Frequency-domain branch: fixed spectral inputs, optional learnable feature filter, ResNet18.

The branch consumes the same normalized RGB tensors as the RGB benchmark models, so crops and
augmentations are shared exactly; spectral inputs are derived inside the model.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

FREQUENCY_INPUTS = {"log_amplitude": 1, "phase": 2, "highpass": 3}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LUMA = (0.299, 0.587, 0.114)


def validate_inputs(inputs: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return the inputs in canonical order; reject unknown, repeated, or empty selections."""
    if (
        not inputs
        or isinstance(inputs, str)
        or set(inputs) - FREQUENCY_INPUTS.keys()
        or len(set(inputs)) != len(inputs)
    ):
        raise ValueError(
            f"frequency_inputs must be distinct values from {sorted(FREQUENCY_INPUTS)}: {inputs}"
        )
    return tuple(name for name in FREQUENCY_INPUTS if name in inputs)


def preprocessing_id(inputs: list[str] | tuple[str, ...]) -> str:
    return "df40-freq-224-" + "-".join(validate_inputs(inputs)) + "-v1"


def _standardize(values: torch.Tensor) -> torch.Tensor:
    """Zero mean, unit variance per image and channel; constant maps become zero."""
    mean = values.mean(dim=(-2, -1), keepdim=True)
    std = values.std(dim=(-2, -1), keepdim=True)
    return (values - mean) / (std + 1e-6)


class FrequencyFeatures(nn.Module):
    """Map ImageNet-normalized RGB to fixed spectral channels (no trainable parameters)."""

    def __init__(self, inputs: list[str] | tuple[str, ...]) -> None:
        super().__init__()
        self.inputs = validate_inputs(inputs)
        self.channels = sum(FREQUENCY_INPUTS[name] for name in self.inputs)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("luma", torch.tensor(LUMA).view(1, 3, 1, 1))
        laplacian = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        self.register_buffer("laplacian", laplacian.expand(3, 1, 3, 3).clone())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # FFTs on non-power-of-two sizes are unsupported in half precision; stay in fp32.
        with torch.autocast(device_type=images.device.type, enabled=False):
            rgb = images.float() * self.std + self.mean
            outputs = []
            if "log_amplitude" in self.inputs or "phase" in self.inputs:
                luminance = (rgb * self.luma).sum(dim=1, keepdim=True)
                spectrum = torch.fft.fftshift(torch.fft.fft2(luminance), dim=(-2, -1))
                if "log_amplitude" in self.inputs:
                    outputs.append(_standardize(torch.log1p(spectrum.abs())))
                if "phase" in self.inputs:
                    angle = spectrum.angle()
                    outputs.extend([angle.cos(), angle.sin()])
            if "highpass" in self.inputs:
                padded = nn.functional.pad(rgb, (1, 1, 1, 1), mode="reflect")
                residual = nn.functional.conv2d(padded, self.laplacian, groups=3)
                outputs.append(_standardize(residual))
            return torch.cat(outputs, dim=1)


class SpectralFilter(nn.Module):
    """Learnable per-channel frequency-domain filter on a feature map, residual and zero-init.

    ``x + irfft2(rfft2(x) * w)`` with ``w = 0`` is the identity, so pretrained behavior is the
    starting point and the filter only learns spectral emphasis that helps.
    """

    def __init__(self, channels: int, height: int, width: int) -> None:
        super().__init__()
        self.size = (height, width)
        self.weight = nn.Parameter(torch.zeros(channels, height, width // 2 + 1, 2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if tuple(features.shape[-2:]) != self.size:
            raise ValueError(f"spectral filter expects {self.size}, got {features.shape[-2:]}")
        with torch.autocast(device_type=features.device.type, enabled=False):
            values = features.float()
            spectrum = torch.fft.rfft2(values, norm="ortho")
            filtered = spectrum * torch.view_as_complex(self.weight)
            return values + torch.fft.irfft2(filtered, s=self.size, norm="ortho")


class FrequencyResNet(nn.Module):
    """ResNet18 over spectral channels; returns 512-wide pooled features."""

    width = 512

    def __init__(
        self,
        inputs: list[str] | tuple[str, ...],
        *,
        feature_fft: bool = False,
        pretrained: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.features = FrequencyFeatures(inputs)
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        stem = backbone.conv1
        conv1 = nn.Conv2d(self.features.channels, stem.out_channels, kernel_size=7, stride=2,
                          padding=3, bias=False)
        with torch.no_grad():
            # Spread the pretrained RGB edge detectors over the new channels so the stem
            # starts with sensible filters and activations keep their pretrained scale.
            conv1.weight.copy_(
                stem.weight.mean(dim=1, keepdim=True).repeat(1, self.features.channels, 1, 1)
                * 3
                / self.features.channels
            )
        backbone.conv1 = conv1
        backbone.fc = nn.Identity()
        self.backbone = backbone
        # layer2 output: stride 8 from the input.
        side = image_size // 8
        self.spectral_filter = SpectralFilter(128, side, side) if feature_fft else None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(self.features(images)))))
        x = b.layer2(b.layer1(x))
        if self.spectral_filter is not None:
            x = self.spectral_filter(x)
        x = b.layer4(b.layer3(x))
        return torch.flatten(b.avgpool(x), 1)

    def stem_parameters(self) -> list[nn.Parameter]:
        """Parameters that must train from the first epoch: the new input stem and filter."""
        parameters = [*self.backbone.conv1.parameters(), *self.backbone.bn1.parameters()]
        if self.spectral_filter is not None:
            parameters.extend(self.spectral_filter.parameters())
        return parameters

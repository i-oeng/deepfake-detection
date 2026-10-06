import math

import pytest

torch = pytest.importorskip("torch")

from deepfake_detection.train.benchmark import (  # noqa: E402
    PREPROCESS_ID,
    BinaryEncoder,
    preprocessing_id,
    set_trainable,
)
from deepfake_detection.train.frequency import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    FrequencyFeatures,
    FrequencyResNet,
    SpectralFilter,
    validate_inputs,
)

ALL = ["log_amplitude", "phase", "highpass"]


def _normalized(rgb: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (rgb - mean) / std


@pytest.mark.parametrize(
    ("inputs", "channels"),
    [(["log_amplitude"], 1), (["phase"], 2), (["highpass"], 3), (ALL, 6)],
)
def test_feature_channels_follow_selected_inputs(inputs, channels) -> None:
    features = FrequencyFeatures(inputs)
    output = features(torch.randn(2, 3, 32, 32))
    assert features.channels == channels
    assert output.shape == (2, channels, 32, 32)
    assert torch.isfinite(output).all()


def test_inputs_are_validated_and_canonically_ordered() -> None:
    assert validate_inputs(["highpass", "log_amplitude"]) == ("log_amplitude", "highpass")
    for bad in ([], ["fft"], ["phase", "phase"], "phase", None):
        with pytest.raises(ValueError, match="frequency_inputs"):
            validate_inputs(bad)


def test_highpass_is_zero_on_a_flat_image() -> None:
    flat = _normalized(torch.full((1, 3, 16, 16), 0.4))
    assert torch.allclose(FrequencyFeatures(["highpass"])(flat), torch.zeros(1, 3, 16, 16))


def test_log_amplitude_peaks_at_the_sinusoid_frequency() -> None:
    size, cycles = 32, 5
    x = torch.arange(size, dtype=torch.float32)
    wave = 0.5 + 0.4 * torch.cos(2 * math.pi * cycles * x / size)
    rgb = wave.view(1, 1, 1, size).expand(1, 3, size, size)
    amplitude = FrequencyFeatures(["log_amplitude"])(_normalized(rgb))[0, 0]
    centre = size // 2
    amplitude[centre, centre] = float("-inf")  # ignore the DC term
    row, column = divmod(int(amplitude.argmax()), size)
    assert row == centre and abs(column - centre) == cycles


def test_features_are_deterministic_and_ignore_autocast_precision() -> None:
    images = torch.randn(2, 3, 24, 24)
    features = FrequencyFeatures(ALL)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        first = features(images)
    assert first.dtype == torch.float32
    assert torch.equal(first, features(images))


def test_spectral_filter_starts_as_identity_and_can_learn() -> None:
    layer = SpectralFilter(4, 8, 8)
    values = torch.randn(2, 4, 8, 8)
    assert torch.allclose(layer(values), values, atol=1e-6)
    with torch.no_grad():
        layer.weight[..., 0] = 0.5
    assert not torch.allclose(layer(values), values)
    with pytest.raises(ValueError, match="expects"):
        layer(torch.randn(1, 4, 6, 6))


def test_frequency_resnet_returns_pooled_features_with_adapted_stem() -> None:
    model = FrequencyResNet(ALL, feature_fft=True, pretrained=False, image_size=64).eval()
    assert model.backbone.conv1.in_channels == 6
    with torch.no_grad():
        features = model(torch.randn(2, 3, 64, 64))
    assert features.shape == (2, FrequencyResNet.width)


def test_benchmark_encoder_and_trainable_policy() -> None:
    config = {"frequency_inputs": ALL, "feature_fft": True, "pretrained": False}
    model = BinaryEncoder("frequency_resnet18", config)
    logits, features = model.eval()(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2,) and features.shape == (2, 512)

    set_trainable(model, 1)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert "encoder.backbone.conv1.weight" in trainable
    assert "encoder.spectral_filter.weight" in trainable
    assert "head.weight" in trainable
    assert not any(name.startswith("encoder.backbone.layer") for name in trainable)

    set_trainable(model, 2)
    assert all(p.requires_grad for p in model.parameters())


def test_preprocessing_id_separates_frequency_variants_and_keeps_rgb() -> None:
    assert preprocessing_id({"model": "convnext_tiny"}) == PREPROCESS_ID
    assert preprocessing_id({"model": "resnet18"}) == PREPROCESS_ID
    assert (
        preprocessing_id({"model": "frequency_resnet18", "frequency_inputs": ["highpass", "phase"]})
        == "df40-freq-224-phase-highpass-v1"
    )
    with pytest.raises(ValueError, match="frequency_inputs"):
        preprocessing_id({"model": "frequency_resnet18"})

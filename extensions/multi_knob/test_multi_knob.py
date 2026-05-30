"""Tests for the multi_knob extension"""

import pytest
import torch
import numpy as np
import os
from nam.models.metadata import UserMetadata
from multi_knob import (
    KnobMetadata,
    MultiKnobUserMetadata,
    MultiKnobDataset,
    MultiKnobModel,
    KnobConditioningWaveNet,
)

# =============================================================================
# KnobMetadata Tests
# =============================================================================

def test_knob_metadata_creation():
    knob = KnobMetadata(
        name="gain",
        min_value=0.0,
        max_value=10.0,
        default_value=5.0,
        units="dB",
    )
    assert knob.name == "gain"
    assert knob.min_value == 0.0
    assert knob.max_value == 10.0
    assert knob.default_value == 5.0
    assert knob.units == "dB"


def test_knob_metadata_optional_units():
    knob = KnobMetadata(
        name="gain",
        min_value=0.0,
        max_value=10.0,
        default_value=5.0,
    )
    assert knob.units is None


# =============================================================================
# MultiKnobUserMetadata Tests
# =============================================================================

def test_multi_knob_user_metadata():
    knobs = {
        "gain": KnobMetadata(
            name="gain",
            min_value=0.0,
            max_value=10.0,
            default_value=5.0,
            units="dB",
        ),
        "tone": KnobMetadata(
            name="tone",
            min_value=-5.0,
            max_value=5.0,
            default_value=0.0,
        ),
    }
    metadata = MultiKnobUserMetadata(knobs=knobs)
    assert len(metadata.knobs) == 2
    assert "gain" in metadata.knobs
    assert "tone" in metadata.knobs


def test_multi_knob_user_metadata_empty():
    """Default MultiKnobUserMetadata should have no knobs."""
    metadata = MultiKnobUserMetadata()
    assert len(metadata.knobs) == 0


# =============================================================================
# KnobConditioningWaveNet Tests
# =============================================================================

class TestKnobConditioningWaveNet:
    @pytest.fixture
    def cond_dsp(self):
        return KnobConditioningWaveNet(["gain", "tone"], embedding_dim=8)

    def test_creation(self, cond_dsp):
        assert len(cond_dsp.knob_embeddings) == 2
        assert "gain" in cond_dsp.knob_embeddings
        assert "tone" in cond_dsp.knob_embeddings

    def test_receptive_field(self, cond_dsp):
        assert cond_dsp.receptive_field == 1

    def test_condition_size(self, cond_dsp):
        assert cond_dsp._condition_size == 16  # 2 knobs * 8 dims

    def test_forward_with_knobs(self, cond_dsp):
        batch_size = 4
        seq_len = 128
        x = torch.randn(batch_size, 1, seq_len)
        cond_dsp.set_values(torch.tensor(0.5), torch.tensor(0.3))
        c = cond_dsp(x)
        assert c.shape == (batch_size, 16, seq_len)

    def test_forward_without_knobs_raises(self, cond_dsp):
        x = torch.randn(2, 1, 64)
        with pytest.raises(RuntimeError, match="not set"):
            cond_dsp(x)

    def test_forward_with_scalar_knobs(self, cond_dsp):
        x = torch.randn(2, 1, 64)
        cond_dsp.set_values(0.5, 0.3)
        c = cond_dsp(x)
        assert c.shape == (2, 16, 64)

    def test_export_weights(self, cond_dsp):
        weights = cond_dsp._export_weights()
        assert isinstance(weights, list)
        assert len(weights) > 0

    def test_export_config(self, cond_dsp):
        config = cond_dsp._export_config()
        assert config["knob_names"] == ["gain", "tone"]
        assert config["embedding_dim"] == 8


# =============================================================================
# MultiKnobDataset Tests
# =============================================================================

class TestMultiKnobDataset:
    @pytest.fixture
    def sample_dataset(self):
        x = torch.randn(1000)
        y = torch.randn(1000)
        knob_settings = {
            "gain": torch.full((1000,), 5.0),
            "tone": torch.full((1000,), 0.0),
        }
        return MultiKnobDataset(
            x=x,
            y=y,
            knob_settings=knob_settings,
            nx=100,
            ny=50,
            sample_rate=44100,
        )

    def test_creation(self, sample_dataset):
        assert isinstance(sample_dataset, MultiKnobDataset)
        assert sample_dataset.sample_rate == 44100
        assert sample_dataset.nx == 100
        assert sample_dataset.ny == 50

    def test_length(self, sample_dataset):
        expected_length = (1000 - 100 + 1) // 50
        assert len(sample_dataset) == expected_length

    def test_getitem(self, sample_dataset):
        result = sample_dataset[0]
        # Returns (audio, *knobs, y) - 2 knobs = 4 elements
        assert isinstance(result, tuple)
        assert len(result) == 4  # audio + 2 knobs + y

        x_segment = result[0]
        y_segment = result[-1]

        assert isinstance(x_segment, torch.Tensor)
        assert isinstance(y_segment, torch.Tensor)
        assert x_segment.shape[0] == 149  # nx + ny - 1
        assert y_segment.shape[0] == 50  # ny

    def test_out_of_bounds(self, sample_dataset):
        with pytest.raises(IndexError):
            _ = sample_dataset[len(sample_dataset)]


# =============================================================================
# MultiKnobModel Tests
# =============================================================================

class TestMultiKnobModel:
    @pytest.fixture
    def sample_model(self):
        knob_config = {
            "gain": {"embedding_dim": 8, "default_value": 5.0},
            "tone": {"embedding_dim": 8, "default_value": 0.0},
        }
        return MultiKnobModel(
            knob_config=knob_config,
            base_model="WaveNet",
            sample_rate=44100,
        )

    def test_creation(self, sample_model):
        assert isinstance(sample_model, MultiKnobModel)
        assert sample_model.sample_rate == 44100
        assert len(sample_model._knob_names) == 2

    def test_receptive_field(self, sample_model):
        assert isinstance(sample_model.receptive_field, int)
        assert sample_model.receptive_field > 0

    def test_pad_start_default(self, sample_model):
        assert sample_model.pad_start_default is True

    def test_forward_with_knobs(self, sample_model):
        batch_size = 2
        input_length = sample_model.receptive_field + 100
        x = torch.randn(batch_size, input_length)
        output = sample_model(x, torch.tensor(5.0), torch.tensor(0.0))
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, input_length)

    def test_forward_scalar_knobs(self, sample_model):
        batch_size = 2
        input_length = sample_model.receptive_field + 100
        x = torch.randn(batch_size, input_length)
        output = sample_model(x, 5.0, 0.0)
        assert output.shape == (batch_size, input_length)

    def test_export_config(self, sample_model):
        config = sample_model._export_config()
        assert "knob_config" in config
        assert "knob_names" in config
        assert "layers" in config

    def test_export_weights(self, sample_model):
        weights = sample_model._export_weights()
        assert isinstance(weights, np.ndarray)
        assert weights.ndim == 1

    def test_insufficient_samples_error(self, sample_model):
        x = torch.randn(2, 10)  # Too few samples
        with pytest.raises(ValueError):
            sample_model(x, 5.0, 0.0)


# =============================================================================
# Config-based initialization tests
# =============================================================================

def test_dataset_init_from_config(tmp_path):
    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile not installed")

    x_path = str(tmp_path / "input.wav")
    y_path = str(tmp_path / "output.wav")

    sample_rate = 44100
    duration = 1.0
    samples = int(sample_rate * duration)

    x_data = np.random.randn(samples).astype(np.float32) * 0.1
    y_data = np.random.randn(samples).astype(np.float32) * 0.1

    sf.write(x_path, x_data, sample_rate)
    sf.write(y_path, y_data, sample_rate)

    config = {
        "x_path": x_path,
        "y_path": y_path,
        "sample_rate": sample_rate,
        "nx": 100,
        "ny": 50,
        "knob_settings": {"gain": 5.0, "tone": 0.0},
    }

    dataset = MultiKnobDataset.init_from_config(config)
    assert isinstance(dataset, MultiKnobDataset)
    assert dataset.sample_rate == sample_rate
    assert dataset.nx == 100
    assert dataset.ny == 50
    assert len(dataset._knob_settings) == 2
    assert "gain" in dataset._knob_settings
    assert "tone" in dataset._knob_settings


def test_model_init_from_config():
    knob_config = {
        "gain": {"embedding_dim": 8, "default_value": 0.5},
        "tone": {"embedding_dim": 8, "default_value": 0.5},
    }
    config = {
        "knob_config": knob_config,
        "base_model": "WaveNet",
        "sample_rate": 48000,
    }
    model = MultiKnobModel.init_from_config(config)
    assert isinstance(model, MultiKnobModel)
    assert model.sample_rate == 48000

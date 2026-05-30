"""
Multi-knob extension for Neural Amp Modeler
Enables training models with multiple knob parameters
"""

import logging
from typing import Dict, Optional, Sequence, Tuple, Any, Union
import torch
import torch.nn as nn
import numpy as np
from nam.data import AbstractDataset, register_dataset_initializer, wav_to_tensor
from nam.models.base import BaseNet
from nam.models._abc import ImportsWeights
from nam.models.factory import register as register_model
from nam.models.metadata import UserMetadata as _UserMetadata
from nam.models.wavenet._wavenet import WaveNet as _WaveNet
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# Metadata Extensions
# =============================================================================

class KnobMetadata(BaseModel):
    """Metadata for a single knob parameter"""
    name: str
    min_value: float
    max_value: float
    default_value: float
    units: Optional[str] = None


class MultiKnobUserMetadata(_UserMetadata):
    """Extended metadata to include knob information"""
    knobs: Dict[str, KnobMetadata] = {}


# =============================================================================
# Knob Conditioning WaveNet (acts as condition_dsp)
# =============================================================================

class KnobConditioningWaveNet(nn.Module, ImportsWeights):
    """
    Produces conditioning signal from knob values for WaveNet's FiLM mechanism.
    Has the same interface as WaveNet for receptive_field but no internal layers.
    """

    def __init__(self, knob_names: list, embedding_dim: int):
        super().__init__()
        self._knob_names = knob_names
        self._condition_size = len(knob_names) * embedding_dim
        self._stored_values = None

        # Create embedding layers for each knob
        self.knob_embeddings = nn.ModuleDict()
        for name in knob_names:
            self.knob_embeddings[name] = nn.Linear(1, embedding_dim, bias=True)

    def set_values(self, *knob_values: torch.Tensor):
        """Store knob values to use during forward pass."""
        self._stored_values = knob_values

    @property
    def receptive_field(self) -> int:
        """Return 1 to not increase the main model's receptive field."""
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produces conditioning signal from stored knob values.

        Args:
            x: Input audio tensor (B, C, L) - used only for shape reference

        Returns:
            Conditioning tensor (B, condition_size, L)
        """
        if self._stored_values is None:
            raise RuntimeError("Knob values not set. Call set_values() before forward.")

        batch_size = x.shape[0]
        seq_len = x.shape[-1]
        device = x.device
        embedded = []

        for idx, name in enumerate(self._knob_names):
            value = self._stored_values[idx]
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32, device=device)
            if value.ndim == 0:
                value = value.expand(batch_size, seq_len)
            elif value.ndim == 1:
                value = value.unsqueeze(0).expand(batch_size, -1)
            elif value.ndim == 2 and value.shape[-1] != seq_len:
                value = value.expand(batch_size, seq_len)

            v = value.unsqueeze(1).to(device)  # (B, 1, L)
            e = self.knob_embeddings[name](v.transpose(1, 2)).transpose(1, 2)
            embedded.append(e)

        return torch.cat(embedded, dim=1)

    def import_weights(self, weights):
        raise NotImplementedError

    def _export_weights(self) -> list:
        weights = []
        for name in self._knob_names:
            emb = self.knob_embeddings[name]
            weights.extend(emb.weight.data.cpu().numpy().flatten())
            if emb.bias is not None:
                weights.extend(emb.bias.data.cpu().numpy().flatten())
        return weights

    def _export_config(self):
        return {
            "knob_names": self._knob_names,
            "embedding_dim": list(self.knob_embeddings.values())[0].out_features,
        }


# =============================================================================
# Dataset
# =============================================================================

class MultiKnobDataset(AbstractDataset):
    """Dataset that pairs audio with per-sample knob parameter values."""

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        knob_settings: Dict[str, torch.Tensor],
        nx: int,
        ny: Optional[int] = None,
        sample_rate: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self._nx = nx
        self._ny = ny if ny is not None else len(x) - nx + 1
        self._sample_rate = sample_rate
        self._x = x
        self._y = y
        self._knob_settings = knob_settings
        self._knob_names = sorted(knob_settings.keys())

        if len(self._x) != len(self._y):
            raise ValueError(f"Length mismatch: input={len(self._x)}, output={len(self._y)}")

    def __len__(self) -> int:
        n = len(self._x)
        single_pairs = n - self._nx + 1
        return max(single_pairs // self._ny, 0)

    def __getitem__(self, idx: int):
        """
        Returns (audio_segment, *knob_values, y_segment).
        Flat tuple format allows proper batching by DataLoader.
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        i = idx * self._ny
        j = i + self._nx - 1
        x_segment = self._x[i : i + self._nx + self._ny - 1]
        y_segment = self._y[j : j + self._ny]
        knob_values = tuple(self._knob_settings[name][i] for name in self._knob_names)

        return (x_segment, *knob_values, y_segment)

    @property
    def sample_rate(self) -> Optional[float]:
        return self._sample_rate

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def ny(self) -> int:
        return self._ny

    @property
    def x(self) -> torch.Tensor:
        return self._x

    @property
    def y(self) -> torch.Tensor:
        return self._y

    @classmethod
    def init_from_config(cls, config):
        parsed = cls.parse_config(config)
        return cls(**parsed)

    @classmethod
    def parse_config(cls, config):
        from nam.data import wav_to_tensor

        sample_rate = config.pop("sample_rate", None)
        x = wav_to_tensor(config.pop("x_path"), rate=sample_rate)
        y = wav_to_tensor(config.pop("y_path"), rate=sample_rate)

        raw_knob_settings = config.pop("knob_settings", {})
        knob_settings = {}
        for name, value in raw_knob_settings.items():
            if value is not None:
                knob_settings[name] = torch.full((len(x),), value, dtype=torch.float32)

        nx = config.pop("nx", None)
        if nx is None:
            nx = 8192

        return {
            "x": x,
            "y": y,
            "knob_settings": knob_settings,
            "nx": nx,
            "ny": config.pop("ny", None),
            "sample_rate": sample_rate,
            **config,
        }


# =============================================================================
# Model
# =============================================================================

class MultiKnobModel(BaseNet, ImportsWeights):
    """
    WaveNet-based model conditioned on external knob parameters.
    Uses WaveNet's built-in FiLM conditioning via condition_dsp mechanism.
    """

    def __init__(
        self,
        knob_config: Dict[str, Dict[str, Any]],
        base_model: Union[str, BaseNet] = "WaveNet",
        sample_rate: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate)
        self.knob_config = knob_config
        self._knob_names = sorted(knob_config.keys())
        total_embedding_dim = sum(c["embedding_dim"] for c in knob_config.values())

        if isinstance(base_model, str) and base_model == "WaveNet":
            channels = 32
            head_size = channels // 2

            # Create the conditioning DSP module
            conditioning_dsp = KnobConditioningWaveNet(self._knob_names, 8)

            # Build the internal WaveNet's layer arrays directly
            from nam.models.wavenet._head import Head
            from nam.models.wavenet._layer_array import LayerArray

            layer_configs = [{
                "input_size": 1,
                "condition_size": total_embedding_dim,
                "channels": channels,
                "head_size": head_size,
                "kernel_size": 3,
                "dilations": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                "activation": "Tanh",
                "gated": True,
                "head_bias": True,
                "is_first": True,
                "is_last": True,
            }]
            layer_arrays = nn.ModuleList([
                LayerArray.init_from_config(lc) for lc in layer_configs
            ])

            head = Head(
                in_channels=head_size,
                channels=head_size,
                activation="Tanh",
                num_layers=2,
                out_channels=1,
            )

            self._wavenet = _WaveNet(
                layer_arrays=layer_arrays,
                head=head,
                head_scale=0.02,
                condition_dsp=conditioning_dsp,
            )
        elif isinstance(base_model, BaseNet):
            self._wavenet = base_model._net if hasattr(base_model, '_net') else base_model
        else:
            raise ValueError(f"Unknown base model: {base_model}")

        self._receptive_field = self._wavenet.receptive_field

    def forward(self, x: torch.Tensor, *knob_args, **kwargs) -> torch.Tensor:
        pad_start = kwargs.pop("pad_start", None)
        if pad_start is None:
            pad_start = self.pad_start_default

        if pad_start:
            pad_length = self.receptive_field - 1
            if x.ndim == 2:
                x = x.unsqueeze(1)
            elif x.ndim == 1:
                x = x.unsqueeze(0).unsqueeze(0)
            x = torch.cat(
                (torch.zeros(x.shape[0], x.shape[1], pad_length, device=x.device), x),
                dim=2,
            )

        if x.shape[-1] < self.receptive_field:
            raise ValueError(
                f"Input has {x.shape[-1]} samples, need {self.receptive_field}"
            )

        output = self._forward(x, *knob_args, **kwargs)

        if x.ndim == 1:
            output = output[0]
        return output

    def _forward(self, x: torch.Tensor, *knob_args, **kwargs) -> torch.Tensor:
        """Set knob values on conditioning DSP, then run WaveNet."""
        x_input = x.unsqueeze(1) if x.ndim == 2 else x

        if x_input.ndim == 1:
            x_input = x_input.unsqueeze(0).unsqueeze(0)

        # Set knob values on the conditioning DSP
        condition_dsp = self._wavenet._condition_dsp
        condition_dsp.set_values(*knob_args)

        y = self._wavenet(x_input)
        assert y.shape[1] == 1, f"Expected 1 output channel, got {y.shape[1]}"
        return y[:, 0, :]

    def import_weights(self, weights: Sequence[float]):
        raise NotImplementedError("MultiKnobModel weight import not implemented")

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return self._receptive_field

    def _export_config(self):
        """Export config with knob metadata."""
        wavenet_config = self._wavenet.export_config(sample_rate=self.sample_rate)
        return {
            "knob_config": self.knob_config,
            "knob_names": self._knob_names,
            **wavenet_config,
        }

    def _export_weights(self) -> np.ndarray:
        """
        Export weights from main WaveNet + conditioning DSP.
        """
        weights = list(self._wavenet.export_weights())
        # Append conditioning DSP knob embedding weights
        condition_dsp = self._wavenet._condition_dsp
        for name in self._knob_names:
            emb = condition_dsp.knob_embeddings[name]
            weights.extend(emb.weight.data.cpu().numpy().flatten())
            if emb.bias is not None:
                weights.extend(emb.bias.data.cpu().numpy().flatten())
        return np.array(weights)

    @classmethod
    def init_from_config(cls, config):
        return cls(**config)


# =============================================================================
# Registration
# =============================================================================

register_dataset_initializer("multi_knob", MultiKnobDataset.init_from_config)
register_model("MultiKnob", MultiKnobModel.init_from_config)

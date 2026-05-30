# NAM Multi-Knob Extension

This extension adds support for training Neural Amp Models with multiple knob parameters. It enables modeling of amps and effects with multiple controls (like gain, tone, volume, etc.) in a single model.

## Installation

Copy the extension files to your NAM extensions directory:

### Linux/macOS
```bash
mkdir -p ~/.neural-amp-modeler/extensions
cp extensions/multi_knob.py ~/.neural-amp-modeler/extensions/
```

### Windows
```powershell
$extensionsPath = "$env:USERPROFILE\.neural-amp-modeler\extensions"
New-Item -ItemType Directory -Force -Path $extensionsPath
Copy-Item "extensions\multi_knob\multi_knob.py" $extensionsPath
```

## Usage

Multi-knob training uses the `nam-full` command-line trainer (not the simplified GUI trainer). You need three JSON configuration files:

### 1. Dataset Configuration (`multi_knob_dataset.json`)

```json
{
  "type": "multi_knob",
  "train": {
    "x_path": "path/to/input.wav",
    "y_path": "path/to/output.wav",
    "knob_settings": {
      "volume": 0.5,
      "gain": 0.7
    },
    "ny": 8192,
    "delay": 1534
  },
  "validation": {
    "x_path": "path/to/input.wav",
    "y_path": "path/to/validation_output.wav",
    "knob_settings": {
      "volume": 0.5,
      "gain": 0.7
    }
  },
  "common": {
    "sample_rate": 48000
  }
}
```

The `knob_settings` object maps knob names to constant values (0.0-1.0) for the entire training run. Each recording/setup should have its own WAV pairs in separate data configs if knob values vary.

### 2. Model Configuration (`multi_knob_config.json`)

```json
{
  "net": {
    "name": "MultiKnob",
    "config": {
      "knob_config": {
        "volume": {
          "name": "Volume",
          "min_value": 0.0,
          "max_value": 1.0,
          "default_value": 0.5,
          "embedding_dim": 8
        },
        "gain": {
          "name": "Gain",
          "min_value": 0.0,
          "max_value": 1.0,
          "default_value": 0.5,
          "embedding_dim": 8
        }
      },
      "base_model": "WaveNet",
      "sample_rate": 48000
    }
  }
}
```

### 3. Learning Configuration (`multi_knob_learning.json`)

```json
{
  "train_dataloader": {
    "batch_size": 16,
    "shuffle": true,
    "pin_memory": true,
    "drop_last": true,
    "num_workers": 0
  },
  "val_dataloader": {},
  "trainer": {
    "accelerator": "cpu",
    "devices": 1,
    "max_epochs": 30
  },
  "trainer_fit_kwargs": {}
}
```

### Training Command

```bash
nam-full \
  path/to/multi_knob_dataset.json \
  path/to/multi_knob_config.json \
  path/to/multi_knob_learning.json \
  path/to/output_directory
```

## Features

- Support for multiple continuous knob parameters with customizable ranges
- Knob parameter embedding for better generalization
- Compatible with NAM's WaveNet architecture via FiLM conditioning
- Exports to .nam format for plugin compatibility
- Includes knob ranges in metadata

## Delay Compensation

The `delay` parameter in the dataset configuration represents the length difference between input and output audio files. Use the provided `analyze_wav.py` script to determine correct delay values:

```bash
python extensions/multi_knob/analyze_wav.py
```

## Testing

The extension includes a comprehensive test suite:

```bash
pytest extensions/multi_knob/test_multi_knob.py -v
```

## Requirements

- NAM version 0.7.x or later
- PyTorch 2.x
- Python 3.10+

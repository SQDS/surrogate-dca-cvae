# Frozen model weights

All 20 selected checkpoints are included: M, S, S-Y, and MS, each at seeds 42–46. The selected epochs and validation errors are in `manifest.csv`.

```bash
python scripts/download_weights.py
```

This verifies every bundled SHA256. If weights are hosted separately later, pass their actual HTTPS release directory with `--base-url`. No unpublished or guessed download URL is configured.

Each portable checkpoint contains the original `state_dict`, model configuration name, seed, epoch, frozen training scaler and protocol hash. Optimizer states and training histories have been removed. Every model tensor was checked for exact equality with its source checkpoint. Thus the portable file checksum differs from the archive checksum; both are retained as `sha256` and `source_sha256` in the manifest.

Load a model with:

```python
from src.models import load_weights
net, metadata = load_weights('MS', seed=42, device='cpu')
```

MS-main and MS-aux use the same MS checkpoint. Inference uses posterior means. Each manuscript run used 400 epochs and chose the saved epoch with the smallest source-weighted validation prediction MSE, using MS-aux to select MS. The selected epoch ranges are M: 380–400; S: 220–400; S-Y: 300–370; MS: 320–380, matching the manuscript's Matched ablations and neural development section.

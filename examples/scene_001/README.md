# scene_001 (toy held-out protocol sample)

This folder contains tiny deterministic toy metadata for the v0.1 selective rendering protocol.

## Files

- `cameras.json`
  - `num_views`, `height`, `width`
  - used by proxy held-out renderer to produce predictions of shape `[V, H, W]`
- `scene.npz` and `targets.npy`
  - intentionally **not committed as binary files** in this repository
  - for local demos, they are optional because v0.1 loader/evaluator provide deterministic fallbacks
  - if present locally, `scene.npz` should include arrays like `positions [N,3]`, `opacity [N]`
  - if present locally, `targets.npy` can be `[V,H,W]` held-out targets or a 1D vector (auto-tiled)

## Protocol note

SRU curves are computed by progressively pruning low-confidence Gaussians, re-rendering held-out predictions with the proxy renderer, and measuring PSNR/SSIM/LPIPS-like quality against held-out targets.

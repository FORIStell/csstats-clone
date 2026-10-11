# Local Image Gen

Lightweight text-to-image generator that runs on your PC using **SD-Turbo** (~2.5GB, 512px, 1–4 steps).
Tuned for low-VRAM GPUs (4GB works via CPU offload).

## Windows setup
1. Install [Python 3.11](https://www.python.org/downloads/) — tick **"Add to PATH"**.
2. Download this repo (Code → Download ZIP) and extract it.
3. Double-click `install.bat` (one time, ~5GB download incl. PyTorch).
4. Double-click `run.bat` — a browser tab opens. First launch downloads the model (~2.5GB).

No GPU / errors with GPU? Run `run.bat --cpu` (slower, ~20–60s per image).

## Tips
- Steps 1–4 is best; more doesn't help much with Turbo models.
- Describe the person, pose, clothing and setting: `full body photo of a man jogging on a beach, sunset`.
- Reuse a seed to tweak a prompt while keeping the same composition.

## Samples (2 steps, CPU)
![](samples/sample_0.png) ![](samples/sample_1.png)

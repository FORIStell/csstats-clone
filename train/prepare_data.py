"""Downloads Flickr8k, keeps photos of people, and caches VAE latents + captions."""
import io
import random
import re

import numpy as np
import pandas as pd
import torch
from diffusers import AutoencoderKL
from huggingface_hub import hf_hub_download
from PIL import Image

from common import CACHE, MODEL_ID, RES

PEOPLE = re.compile(r"\b(man|men|woman|women|boy|girl|child|children|kid|person|people|lady|guy|player|baby)s?\b", re.I)


def center_crop(img):
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return img.resize((RES, RES), Image.BICUBIC)


def main(limit=600):
    out = CACHE / "latents.pt"
    if out.exists():
        print("cache exists:", out)
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    shards = ["data/train-00000-of-00002-2f8f6bfa852eac4b.parquet", "data/train-00001-of-00002-2173151d8cd6c7fb.parquet"]
    df = pd.concat([pd.read_parquet(hf_hub_download("jxie/flickr8k", s, repo_type="dataset")) for s in shards])
    cap_cols = [c for c in df.columns if c.startswith("caption")]
    rows = [r for _, r in df.iterrows() if PEOPLE.search(r[cap_cols[0]])]
    random.Random(0).shuffle(rows)
    rows = rows[:limit]
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").eval()
    latents, captions = [], []
    with torch.no_grad():
        for i, r in enumerate(rows):
            img = center_crop(Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB"))
            x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(127.5).sub(1)[None]
            latents.append(vae.encode(x).latent_dist.mean[0].mul(vae.config.scaling_factor).half())
            captions.append([r[c] for c in cap_cols])
            if i % 100 == 0:
                print(f"{i}/{len(rows)}", flush=True)
    torch.save({"latents": torch.stack(latents), "captions": captions}, out)
    print("saved", out, len(latents))


if __name__ == "__main__":
    main()

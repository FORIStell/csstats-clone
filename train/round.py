"""One training round: 250 LoRA steps -> eval on fixed prompts -> score -> keep if better.

Outputs rounds/NNN/{winner.png,grid.jpg,metrics.json} and updates lora/best when the round beats the best score.
"""
import argparse
import json
import random
import shutil
import time

import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForText2Image, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import CLIPModel, CLIPProcessor

from common import CACHE, EVAL_PROMPTS, EVAL_SEED, LORA_DIR, MODEL_ID, ROUNDS_DIR, STATE_FILE

LORA_CFG = dict(r=8, lora_alpha=8, init_lora_weights="gaussian", target_modules=["to_k", "to_q", "to_v", "to_out.0"])
GOOD = "a high quality sharp photo of a person with natural anatomy and a realistic face"
BAD = "a blurry distorted photo of a person with a deformed body, extra limbs and a messed up face"


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"round": 0, "best_score": None, "best_round": None, "history": []}


def build_pipe(with_lora):
    pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    if with_lora:
        pipe.unet.add_adapter(LoraConfig(**LORA_CFG))
        raw = LORA_DIR / "peft_state.safetensors"
        if raw.exists():
            set_peft_model_state_dict(pipe.unet, load_file(raw))
    return pipe


def train(pipe, steps, lr, batch):
    data = torch.load(CACHE / "latents.pt")
    latents, captions = data["latents"], data["captions"]
    sched = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    unet, te, tok = pipe.unet, pipe.text_encoder, pipe.tokenizer
    unet.train()
    unet.enable_gradient_checkpointing()
    params = [p for n, p in unet.named_parameters() if "lora" in n]
    for p in unet.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    t0, losses = time.time(), []
    for step in range(steps):
        idx = random.sample(range(len(latents)), batch)
        x0 = latents[idx].float()
        caps = [random.choice(captions[i]) for i in idx]
        with torch.no_grad():
            ids = tok(caps, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids
            enc = te(ids)[0]
        noise = torch.randn_like(x0)
        # Bias toward high-noise timesteps: Turbo generates in 1-4 steps starting from t=999.
        t = (torch.rand(batch).pow(0.5) * sched.config.num_train_timesteps).long().clamp(max=999)
        xt = sched.add_noise(x0, noise, t)
        loss = F.mse_loss(unet(xt, t, enc).sample, noise)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        if (step + 1) % 25 == 0:
            el = time.time() - t0
            print(f"step {step+1}/{steps} loss {sum(losses[-25:])/25:.4f} ({el/(step+1):.1f}s/step)", flush=True)
    unet.eval()
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate(pipe):
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    images, scores = [], []
    for i, prompt in enumerate(EVAL_PROMPTS):
        img = pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0.0,
                   generator=torch.Generator().manual_seed(EVAL_SEED + i)).images[0]
        inp = proc(text=[prompt, GOOD, BAD], images=img, return_tensors="pt", padding=True)
        out = clip(**inp)
        sims = F.cosine_similarity(out.image_embeds, out.text_embeds)
        # Prompt match + "looks good vs looks deformed" contrast.
        scores.append((sims[0] + 2 * (sims[1] - sims[2])).item())
        images.append(img)
    return images, scores


def save_lora(unet):
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    sd = get_peft_model_state_dict(unet)
    save_file({k: v.contiguous() for k, v in sd.items()}, LORA_DIR / "peft_state.safetensors")
    StableDiffusionPipeline.save_lora_weights(LORA_DIR, unet_lora_layers=convert_state_dict_to_diffusers(sd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    state = load_state()
    if state["best_score"] is None:
        print("Scoring the base model (round 0 baseline)...", flush=True)
        imgs, scores = evaluate(build_pipe(with_lora=False))
        state["best_score"] = sum(scores) / len(scores)
        state["best_round"] = 0
        write_round(0, imgs, scores, {"note": "base model, no training"})

    n = state["round"] + 1
    seed = random.randrange(2**31)
    random.seed(seed)
    torch.manual_seed(seed)
    pipe = build_pipe(with_lora=True)  # starts from the current best LoRA
    t0 = time.time()
    loss = train(pipe, args.steps, args.lr, args.batch)
    imgs, scores = evaluate(pipe)
    mean = sum(scores) / len(scores)
    improved = mean > state["best_score"]
    if improved:
        save_lora(pipe.unet)
        state["best_score"], state["best_round"] = mean, n
    info = {"steps": args.steps, "lr": args.lr, "seed": seed, "train_loss": loss, "minutes": (time.time() - t0) / 60,
            "improved": improved, "best_round_after": state["best_round"]}
    write_round(n, imgs, scores, info)
    state["round"] = n
    state["history"].append({"round": n, "score": mean, "improved": improved})
    STATE_FILE.write_text(json.dumps(state, indent=2))
    write_leaderboard(state)
    print(json.dumps(info), "score", mean, flush=True)


def write_round(n, imgs, scores, info):
    d = ROUNDS_DIR / f"{n:03d}"
    d.mkdir(parents=True, exist_ok=True)
    best = max(range(len(imgs)), key=scores.__getitem__)
    imgs[best].save(d / "winner.png")
    grid = Image.new("RGB", (256 * 4, 256 * 2))
    for i, im in enumerate(imgs):
        grid.paste(im.resize((256, 256)), ((i % 4) * 256, (i // 4) * 256))
    grid.save(d / "grid.jpg", quality=85)
    info = {**info, "mean_score": sum(scores) / len(scores), "scores": scores, "winner_prompt": EVAL_PROMPTS[best]}
    (d / "metrics.json").write_text(json.dumps(info, indent=2))


def write_leaderboard(state):
    lines = ["# Training rounds", "", f"Current best: **round {state['best_round']}** (score {state['best_score']:.4f})", "",
             "| Round | Score | Kept? | Winner |", "|---|---|---|---|"]
    for h in reversed(state["history"]):
        lines.append(f"| {h['round']} | {h['score']:.4f} | {'yes' if h['improved'] else 'no'} | ![](rounds/{h['round']:03d}/winner.png) |")
    lines.append("| 0 (base) | — | — | ![](rounds/000/winner.png) |")
    (ROUNDS_DIR.parent / "ROUNDS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

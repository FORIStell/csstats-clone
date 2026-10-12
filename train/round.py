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
from diffusers import AutoencoderTiny, AutoPipelineForText2Image, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

from common import CACHE, EVAL_PROMPTS, EVAL_VERSION, EVAL_SEED, LORA_DIR, MODEL_ID, ROUNDS_DIR, STATE_FILE, random_train_prompt

LORA_CFG = dict(r=8, lora_alpha=8, init_lora_weights="gaussian", target_modules=["to_k", "to_q", "to_v", "to_out.0"])
# Training reward: PickScore, a CLIP-H model fine-tuned on human preference votes (Pick-a-Pic).
# Plain CLIP rewards drifted toward washed-out images that a person rated worse in a blind test.
REWARD_MODEL = "yuvalkirstain/PickScore_v1"
REWARD_PROC = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
EVAL_CLIP = "openai/clip-vit-large-patch14"  # different scorer for judging, to catch reward hacking
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


def pick_reward(model, proc, images, prompts):
    """Differentiable PickScore on [0,1] image tensors: cosine sim between image and prompt embeddings."""
    mean = torch.tensor(proc.image_processor.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_processor.image_std).view(1, 3, 1, 1)
    px = (F.interpolate(images, size=224, mode="bicubic", antialias=True) - mean) / std
    img = F.normalize(model.visual_projection(model.vision_model(pixel_values=px).pooler_output), dim=-1)
    txt = proc.tokenizer(list(prompts), padding=True, truncation=True, max_length=77, return_tensors="pt")
    with torch.no_grad():
        t = F.normalize(model.text_projection(model.text_model(**txt).pooler_output), dim=-1)
    return (img * t).sum(-1)


def train(pipe, steps, lr, batch, kl_weight, round_n):
    """DRaFT-style reward fine-tuning: generate in 1 step, backprop the PickScore reward into the LoRA.

    Plain diffusion (MSE) training blurs SD-Turbo's few-step outputs; optimizing the generated image directly keeps
    it sharp. kl_weight keeps the LoRA's prediction close to the base model's to limit reward hacking.
    """
    rng = random.Random()
    sched = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    abar = sched.alphas_cumprod[999]
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval().requires_grad_(False)
    reward_model = AutoModel.from_pretrained(REWARD_MODEL).eval().requires_grad_(False)
    reward_proc = AutoProcessor.from_pretrained(REWARD_PROC)
    unet, te, tok = pipe.unet, pipe.text_encoder, pipe.tokenizer
    unet.enable_gradient_checkpointing()
    for p in unet.parameters():
        p.requires_grad_(False)
    params = [p for n, p in unet.named_parameters() if "lora" in n]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    t0, rewards, start = time.time(), [], 0
    # Resume a round interrupted by a container restart.
    ckpt_path = CACHE / "round_ckpt.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path)
        if ckpt["round"] == round_n:
            set_peft_model_state_dict(unet, ckpt["lora"])
            opt.load_state_dict(ckpt["opt"])
            start, rewards = ckpt["step"], ckpt["rewards"]
            print(f"resumed round {round_n} at step {start}", flush=True)
    t = torch.full((batch,), 999)
    for step in range(start, steps):
        prompts = [random_train_prompt(rng) for _ in range(batch)]
        with torch.no_grad():
            ids = tok(prompts, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids
            enc = te(ids)[0]
        xt = torch.randn(batch, 4, 64, 64)
        with torch.no_grad():
            unet.disable_adapters()
            eps_base = unet(xt, t, enc).sample
            unet.enable_adapters()
        unet.train()
        eps = unet(xt, t, enc).sample
        x0 = (xt - (1 - abar).sqrt() * eps) / abar.sqrt()
        img = (taesd.decode(x0 / pipe.vae.config.scaling_factor).sample.clamp(-1, 1) + 1) / 2
        reward = pick_reward(reward_model, reward_proc, img, prompts).mean()
        kl = F.mse_loss(eps, eps_base)
        loss = -reward + kl_weight * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()
        rewards.append(reward.item())
        if (step + 1) % 25 == 0:
            el = time.time() - t0
            print(f"step {step+1}/{steps} reward {sum(rewards[-25:])/25:.4f} kl {kl.item():.5f} ({el/(step+1-start):.1f}s/step)", flush=True)
            CACHE.mkdir(parents=True, exist_ok=True)
            torch.save({"round": round_n, "step": step + 1, "rewards": rewards, "opt": opt.state_dict(),
                        "lora": get_peft_model_state_dict(unet)}, ckpt_path)
    unet.eval()
    return sum(rewards) / len(rewards)  # checkpoint is removed by main() once the round is recorded


@torch.no_grad()
def evaluate(pipe):
    clip = CLIPModel.from_pretrained(EVAL_CLIP).eval()
    proc = CLIPProcessor.from_pretrained(EVAL_CLIP)
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


def save_lora(unet, out_dir=LORA_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = get_peft_model_state_dict(unet)
    save_file({k: v.contiguous() for k, v in sd.items()}, out_dir / "peft_state.safetensors")
    StableDiffusionPipeline.save_lora_weights(out_dir, unet_lora_layers=convert_state_dict_to_diffusers(sd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=None, help="default: adaptive, stored in state.json")
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    state = load_state()
    if state["best_score"] is None:
        print("Scoring the base model (round 0 baseline)...", flush=True)
        imgs, scores = evaluate(build_pipe(with_lora=False))
        state["best_score"] = sum(scores) / len(scores)
        state["best_round"] = 0
        write_round(0, imgs, scores, {"note": "base model, no training"})
        state["eval_version"] = EVAL_VERSION
    elif state.get("eval_version", 1) != EVAL_VERSION:
        # Test prompts changed: re-score the current best so later rounds are compared fairly.
        print(f"Re-scoring best (round {state['best_round']}) on eval set v{EVAL_VERSION}...", flush=True)
        imgs, scores = evaluate(build_pipe(with_lora=state["best_round"] != 0))
        state["best_score"] = sum(scores) / len(scores)
        state["eval_version"] = EVAL_VERSION
        state["history"].append({"round": f"{state['best_round']} (re-scored, eval v{EVAL_VERSION})",
                                 "score": state["best_score"], "improved": True})

    n = state["round"] + 1
    # Adaptive LR: halve after a failed round (fine steps near the optimum), never below 2.5e-6.
    if args.lr is None:
        lr = state.get("lr", 2e-5)
        if state["history"] and not state["history"][-1]["improved"]:
            lr = max(lr / 2, 2.5e-6)
        args.lr = state["lr"] = lr
    seed = random.randrange(2**31)
    random.seed(seed)
    torch.manual_seed(seed)
    pipe = build_pipe(with_lora=True)  # starts from the current best LoRA
    t0 = time.time()
    reward = train(pipe, args.steps, args.lr, args.batch, args.kl, n)
    imgs, scores = evaluate(pipe)
    mean = sum(scores) / len(scores)
    improved = mean > state["best_score"]
    save_lora(pipe.unet, CACHE / "candidates" / f"{n:03d}")  # local only, for human A/B tests
    if improved:
        save_lora(pipe.unet)
        state["best_score"], state["best_round"] = mean, n
    info = {"steps": args.steps, "lr": args.lr, "seed": seed, "train_reward": reward, "kl_weight": args.kl, "minutes": (time.time() - t0) / 60,
            "improved": improved, "best_round_after": state["best_round"]}
    write_round(n, imgs, scores, info)
    state["round"] = n
    state["history"].append({"round": n, "score": mean, "improved": improved})
    STATE_FILE.write_text(json.dumps(state, indent=2))
    write_leaderboard(state)
    (CACHE / "round_ckpt.pt").unlink(missing_ok=True)
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
        if isinstance(h["round"], str):  # eval-set change marker
            lines.append(f"| {h['round']} | {h['score']:.4f} | — | new test prompts from here up |")
            continue
        lines.append(f"| {h['round']} | {h['score']:.4f} | {'yes' if h['improved'] else 'no'} | ![](rounds/{h['round']:03d}/winner.png) |")
    lines.append("| 0 (base) | — | — | ![](rounds/000/winner.png) |")
    (ROUNDS_DIR.parent / "ROUNDS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

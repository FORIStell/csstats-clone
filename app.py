"""Lightweight local image generator (SD-Turbo) with a simple web UI."""
import argparse
import random
from pathlib import Path

import gradio as gr
import torch
from diffusers import AutoPipelineForText2Image

MODEL_ID = "stabilityai/sd-turbo"
LORA_DIR = Path(__file__).parent / "lora" / "best"


def load_pipe(device):
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, torch_dtype=dtype, variant="fp16" if device == "cuda" else None)
    if (LORA_DIR / "pytorch_lora_weights.safetensors").exists():
        print("Loading trained LoRA from", LORA_DIR)
        pipe.load_lora_weights(str(LORA_DIR))
        pipe.fuse_lora()
    if device == "cuda":
        # Keeps VRAM use low enough for 4GB cards.
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
    else:
        pipe.to("cpu")
    pipe.enable_attention_slicing()
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Loading {MODEL_ID} on {device} (first run downloads ~2.5GB)...")
    pipe = load_pipe(device)

    def generate(prompt, negative, steps, size, seed):
        seed = int(seed) if seed >= 0 else random.randint(0, 2**31 - 1)
        w, h = (int(x) for x in size.split("x"))
        image = pipe(
            prompt=prompt,
            negative_prompt=negative or None,
            num_inference_steps=int(steps),
            guidance_scale=0.0,  # SD-Turbo is trained without CFG
            width=w,
            height=h,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
        return image, seed

    with gr.Blocks(title="Local Image Gen") as ui:
        gr.Markdown(f"## Local Image Gen\nModel: `{MODEL_ID}` on **{device}**")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="photo of a woman standing in a park, full body, natural light")
                negative = gr.Textbox(label="Negative prompt (optional)", value="blurry, deformed, extra limbs")
                steps = gr.Slider(1, 8, value=4, step=1, label="Steps (1-4 is usually enough)")
                size = gr.Dropdown(["512x512", "512x768", "768x512"], value="512x512", label="Size")
                seed = gr.Number(value=-1, label="Seed (-1 = random)")
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                out = gr.Image(label="Result")
                used_seed = gr.Number(label="Seed used", interactive=False)
        btn.click(generate, [prompt, negative, steps, size, seed], [out, used_seed])

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()

"""Lightweight local image generator (SD-Turbo) with a simple web UI and optional pose control."""
import argparse
import random
import time
from pathlib import Path

import gradio as gr
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image, ImageOps

from poses import POSE_HINTS, POSES, pose_image

MODEL_ID = "stabilityai/sd-turbo"
POSE_MODEL_ID = "thibaud/controlnet-sd21-openpose-diffusers"  # OpenPose ControlNet for SD 2.1 (SD-Turbo's base)
LORA_DIR = Path(__file__).parent / "lora" / "best"
OUTPUT_DIR = Path(__file__).parent / "outputs"

# Appended to the prompt; SD-Turbo follows style words well.
STYLES = {
    "Photo": "",
    "Cinematic": "cinematic film still, dramatic lighting, shallow depth of field",
    "Anime": "anime style illustration, clean line art, vibrant colors",
    "3D render": "3D render, Pixar style, soft studio lighting",
    "Oil painting": "oil painting, visible brush strokes, classical art",
    "Pencil sketch": "pencil sketch, black and white, detailed shading",
    "Watercolor": "watercolor painting, soft washes, paper texture",
    "Comic book": "comic book art, bold ink outlines, halftone shading",
    "Pixel art": "pixel art, 16-bit retro video game style",
    "Cyberpunk": "cyberpunk, neon lights, rainy night city, futuristic",
    "Fantasy art": "epic fantasy concept art, magical atmosphere, detailed",
    "Vintage film": "1970s vintage film photo, grain, faded warm colors",
    "Black & white photo": "black and white photograph, high contrast, film grain",
    "Studio portrait": "professional studio photo, softbox lighting, plain backdrop",
    "Claymation": "claymation character made of plasticine clay, stop-motion animation still, handmade miniature set",
    "Low poly": "low poly 3D art, flat shaded geometric facets",
    "Pop art": "pop art, bright flat colors, Andy Warhol style screen print",
    "Ukiyo-e": "ukiyo-e Japanese woodblock print",
    "Charcoal drawing": "charcoal drawing, rough smudged strokes",
    "Stained glass": "stained glass window, bold lead lines, glowing colors",
}


def load_pipe(device):
    dtype = torch.float16 if device == "cuda" else torch.float32
    controlnet = ControlNetModel.from_pretrained(POSE_MODEL_ID, torch_dtype=dtype)
    # One ControlNet pipeline serves both modes: text-only runs with the pose strength at 0.
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_ID, controlnet=controlnet, torch_dtype=dtype, variant="fp16" if device == "cuda" else None,
        safety_checker=None, requires_safety_checker=False)
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


_detector = None


def detect_pose(photo):
    """Extract an OpenPose skeleton from a photo (downloads the detector on first use)."""
    global _detector
    if _detector is None:
        from controlnet_aux import OpenposeDetector
        _detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    return _detector(photo, include_hand=False, include_face=False, detect_resolution=512, image_resolution=512)


def fit(img, w, h):
    """Letterbox an image onto a black w x h canvas without stretching it."""
    img = img.convert("RGB")
    s = min(w / img.width, h / img.height)
    img = img.resize((round(img.width * s), round(img.height * s)))
    canvas = Image.new("RGB", (w, h))
    canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Loading {MODEL_ID} + pose model on {device} (first run downloads ~4GB)...")
    pipe = load_pipe(device)

    def build_skeleton(mode, preset, photo, skeleton, w, h, mirror):
        if mode == "Preset pose":
            pose = pose_image(preset)
        elif mode == "Copy pose from my photo":
            if photo is None:
                raise gr.Error("Upload a photo of someone in the pose you want.")
            pose = detect_pose(photo)
        else:
            if skeleton is None:
                raise gr.Error("Upload an OpenPose skeleton image.")
            pose = skeleton
        pose = fit(pose, w, h)
        return ImageOps.mirror(pose) if mirror else pose

    def generate(prompt, style, count, steps, size, seed, use_pose, mode, preset, photo, skeleton, strength, add_hint,
                 mirror):
        seed = int(seed) if seed >= 0 else random.randint(0, 2**31 - 1)
        w, h = (int(x) for x in size.split("x"))
        if use_pose:
            control = build_skeleton(mode, preset, photo, skeleton, w, h, mirror)
            if add_hint and mode == "Preset pose" and preset in POSE_HINTS:
                prompt = f"{prompt}, {POSE_HINTS[preset]}"
        else:
            control, strength = Image.new("RGB", (w, h)), 0.0
        if STYLES[style]:
            prompt = f"{prompt}, {STYLES[style]}"
        OUTPUT_DIR.mkdir(exist_ok=True)
        images = []
        for i in range(int(count)):
            image = pipe(
                prompt=prompt,
                image=control,
                controlnet_conditioning_scale=float(strength),
                num_inference_steps=int(steps),
                guidance_scale=0.0,  # SD-Turbo is trained without CFG
                width=w,
                height=h,
                generator=torch.Generator("cpu").manual_seed(seed + i),
            ).images[0]
            image.save(OUTPUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_seed{seed + i}.png")
            images.append(image)
        return images, (control if use_pose else None), seed

    with gr.Blocks(title="Local Image Gen") as ui:
        gr.Markdown(f"## Local Image Gen\nModel: `{MODEL_ID}` on **{device}**")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="photo of a woman in a park, full body, natural light")
                style = gr.Dropdown(list(STYLES), value="Photo", label="Style")
                count = gr.Slider(1, 4, value=1, step=1, label="Number of images (each uses the next seed)")
                steps = gr.Slider(1, 8, value=2, step=1, label="Steps (1-4 is usually enough)")
                size = gr.Dropdown(["512x512", "512x768", "768x512"], value="512x512", label="Size")
                seed = gr.Number(value=-1, label="Seed (-1 = random)")
                with gr.Accordion("Pose control (for handstands, flips, lying down...)", open=True):
                    use_pose = gr.Checkbox(label="Use pose control", value=False)
                    mode = gr.Radio(["Preset pose", "Copy pose from my photo", "My skeleton image"],
                                    value="Preset pose", label="Pose source")
                    preset = gr.Dropdown(list(POSES), value="Handstand", label="Preset pose")
                    photo = gr.Image(label="Photo to copy the pose from", type="pil")
                    skeleton = gr.Image(label="OpenPose skeleton image", type="pil")
                    strength = gr.Slider(0.3, 1.5, value=1.0, step=0.05, label="Pose strength")
                    add_hint = gr.Checkbox(label="Add pose words to prompt (presets)", value=True)
                    mirror = gr.Checkbox(label="Mirror pose (flip left/right)", value=False)
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                out = gr.Gallery(label="Results (also saved to the outputs folder)", columns=2, height="auto")
                pose_used = gr.Image(label="Pose used")
                used_seed = gr.Number(label="First seed used", interactive=False)
        btn.click(generate, [prompt, style, count, steps, size, seed, use_pose, mode, preset, photo, skeleton, strength,
                             add_hint, mirror],
                  [out, pose_used, used_seed])

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()

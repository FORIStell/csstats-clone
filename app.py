"""Lightweight local image generator (SD-Turbo) with a simple web UI and optional pose control."""
import argparse
import random
from pathlib import Path

import gradio as gr
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image

from poses import POSE_HINTS, POSES, pose_image

MODEL_ID = "stabilityai/sd-turbo"
POSE_MODEL_ID = "thibaud/controlnet-sd21-openpose-diffusers"  # OpenPose ControlNet for SD 2.1 (SD-Turbo's base)
LORA_DIR = Path(__file__).parent / "lora" / "best"


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

    def build_skeleton(mode, preset, photo, skeleton, w, h):
        if mode == "Preset pose":
            return fit(pose_image(preset), w, h)
        if mode == "Copy pose from my photo":
            if photo is None:
                raise gr.Error("Upload a photo of someone in the pose you want.")
            return fit(detect_pose(photo), w, h)
        if skeleton is None:
            raise gr.Error("Upload an OpenPose skeleton image.")
        return fit(skeleton, w, h)

    def generate(prompt, steps, size, seed, use_pose, mode, preset, photo, skeleton, strength, add_hint):
        seed = int(seed) if seed >= 0 else random.randint(0, 2**31 - 1)
        w, h = (int(x) for x in size.split("x"))
        if use_pose:
            control = build_skeleton(mode, preset, photo, skeleton, w, h)
            if add_hint and mode == "Preset pose" and preset in POSE_HINTS:
                prompt = f"{prompt}, {POSE_HINTS[preset]}"
        else:
            control, strength = Image.new("RGB", (w, h)), 0.0
        image = pipe(
            prompt=prompt,
            image=control,
            controlnet_conditioning_scale=float(strength),
            num_inference_steps=int(steps),
            guidance_scale=0.0,  # SD-Turbo is trained without CFG
            width=w,
            height=h,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
        return image, (control if use_pose else None), seed

    with gr.Blocks(title="Local Image Gen") as ui:
        gr.Markdown(f"## Local Image Gen\nModel: `{MODEL_ID}` on **{device}**")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="photo of a woman in a park, full body, natural light")
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
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                out = gr.Image(label="Result")
                pose_used = gr.Image(label="Pose used")
                used_seed = gr.Number(label="Seed used", interactive=False)
        btn.click(generate, [prompt, steps, size, seed, use_pose, mode, preset, photo, skeleton, strength, add_hint],
                  [out, pose_used, used_seed])

    ui.launch(inbrowser=True)


if __name__ == "__main__":
    main()

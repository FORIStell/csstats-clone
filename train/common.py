import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL_ID = "stabilityai/sd-turbo"
CACHE = Path(os.environ.get("TRAIN_CACHE", Path.home() / ".cache" / "imggen-train"))
LORA_DIR = REPO / "lora" / "best"
ROUNDS_DIR = REPO / "rounds"
STATE_FILE = REPO / "train" / "state.json"
RES = 512

# Fixed eval set: same prompts + seeds every round so rounds are comparable.
EVAL_PROMPTS = [
    "full body photo of a woman walking down a city street, natural light",
    "photo of a man sitting on a park bench, full body",
    "a child running on a beach, full body, sunny day",
    "an old man smiling, portrait photo, detailed face",
    "two friends standing together and laughing, outdoors",
    "a woman doing yoga on a mat in a living room, full body",
    "a man jogging in a park, side view, full body",
    "a girl holding an umbrella in the rain, full body",
]
EVAL_SEED = 1234

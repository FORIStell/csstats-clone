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
    "photo of a woman lying on a bed reading a book, wearing pajamas, full body",
    "a man jumping in joy with arms raised, outdoors, full body photo",
    "a gymnast mid backflip in the air, full body, action photo",
    "a woman standing with her back turned to the camera, looking at the ocean, full body",
    "a man sitting cross-legged on the floor playing guitar, full body",
    "a girl spinning around in a dress in a field, full body, motion",
    "a skateboarder doing a trick mid air, full body, street photo",
    "a man lying on the grass looking up at the sky, full body, from above",
]
EVAL_SEED = 1234

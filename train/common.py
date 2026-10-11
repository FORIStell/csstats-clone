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

# Training prompts are generated from templates (never the eval prompts) so the score isn't overfit.
SUBJECTS = ["a man", "a woman", "a young man", "a young woman", "an old man", "an old woman", "a boy", "a girl",
            "a dancer", "an athlete", "two friends", "a couple"]
ACTIONS = ["jumping in the air", "running", "doing a cartwheel", "sitting on a chair", "lying on a sofa", "stretching",
           "walking away from the camera", "kneeling on the ground", "climbing a wall", "dancing", "waving",
           "doing a handstand", "sitting on stairs", "lying on a bed", "riding a bicycle", "kicking a ball",
           "leaning against a wall", "crouching down", "throwing a frisbee", "hugging"]
SETTINGS = ["in a park", "on a beach", "in a bedroom", "on a city street", "in a gym", "in a forest", "in a kitchen",
            "on a rooftop", "in a field of flowers", "in a living room", "at sunset", "in the snow"]
SHOTS = ["full body photo", "full body, wide shot", "photo, full body, from the side", "photo from behind, full body",
         "full body, low angle photo"]


def random_train_prompt(rng):
    return f"{rng.choice(SUBJECTS)} {rng.choice(ACTIONS)} {rng.choice(SETTINGS)}, {rng.choice(SHOTS)}"

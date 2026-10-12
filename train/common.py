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
    "a boy doing a cartwheel on the grass, full body",
    "a man jumping in joy with arms raised, outdoors, full body photo",
    "a gymnast mid backflip in the air, full body, action photo",
    "a girl hanging upside down from a tree branch, full body",
    "a woman doing a handstand on a beach, full body",
    "a girl spinning around in a dress in a field, full body, motion",
    "a skateboarder doing a trick mid air, full body, street photo",
    "a man lying on his stomach on a bed reading a phone, full body, side view",
]
EVAL_VERSION = 2  # bump when EVAL_PROMPTS change; the current best is re-scored on the new set
EVAL_SEED = 1234

# Training prompts are generated from templates (never the eval prompts) so the score isn't overfit.
SUBJECTS = ["a man", "a woman", "a young man", "a young woman", "an old man", "an old woman", "a boy", "a girl",
            "a dancer", "an athlete", "two friends", "a couple"]
# Weighted toward hard poses (acrobatics, upside down, lying, unusual angles), where the model is weakest.
ACTIONS = ["doing a cartwheel", "doing a handstand", "doing a backflip", "hanging upside down from a bar",
           "doing a split", "jumping high in the air", "mid somersault", "doing a bridge pose", "doing a cartwheel",
           "doing a handstand against a wall", "lying on their stomach", "lying on their back with legs up",
           "lying on a bed", "curled up on a sofa", "kneeling and stretching", "crouching down", "sitting cross-legged",
           "climbing a rope", "diving into a pool", "falling backwards", "kicking high", "dancing ballet on tiptoes",
           "running", "walking away from the camera"]
SETTINGS = ["in a park", "on a beach", "in a bedroom", "on a city street", "in a gym", "in a forest", "in a kitchen",
            "on a rooftop", "in a field of flowers", "in a living room", "at sunset", "in the snow"]
SHOTS = ["full body photo", "full body, wide shot", "photo, full body, from the side", "photo from behind, full body",
         "full body, low angle photo"]


def random_train_prompt(rng):
    return f"{rng.choice(SUBJECTS)} {rng.choice(ACTIONS)} {rng.choice(SETTINGS)}, {rng.choice(SHOTS)}"

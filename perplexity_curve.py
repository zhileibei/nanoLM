import json
import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

# Usage: first run train.py, then generate perplexity figure from the directory of example samples.

# -----------------------------------------------------------------------------
# default config values
out_dir = "out"
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# directory containing json files
DIR = Path(out_dir)

# regex to extract step number from filename
STEP_RE = re.compile(r"step(\d+)")

steps = []
scores = []

for path in DIR.glob("*.json"):
    match = STEP_RE.search(path.name)
    if match is None:
        continue

    step = int(match.group(1))

    with open(path, "r") as f:
        data = json.load(f)
    # adjust this key if your JSON uses a different field name
    ppls = data["perplexities"]
    # calculate average
    ppl = sum(ppls) / len(ppls)

    steps.append(step)
    scores.append(ppl)

# sort by step
steps, scores = zip(*sorted(zip(steps, scores)))

# plot
plt.figure()
plt.plot(steps, scores, marker="o")
plt.xlabel("Step")
plt.ylabel("Score")
plt.title("Score vs Step")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{DIR}/ppl_curve.png")

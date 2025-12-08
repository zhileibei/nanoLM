import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


# plot
plt.figure()

# directory containing json files
# DIR = Path("/home/beizl42/projects/nanoLM/out-shakespeare/perplexity")
# DIR = Path("/home/beizl42/projects/nanoLM/out-shakespeare-diffusion/perplexity")
# DIR = Path("/home/beizl42/projects/nanoLM/out-shakespeare-diffusion-100ksteps/perplexity")
# DIR = Path("/home/beizl42/projects/nanoLM/diffusion-10ksteps-notime/perplexity")
for model in ['out-shakespeare-10ksteps', 'diffusion-10ksteps-notime']:
    DIR = Path(f"/home/beizl42/projects/nanoLM/{model}/perplexity")
    print(DIR)
    exp_name = "10ksteps-notime"

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
    plt.plot(steps, scores, marker="o", label=model)
    
plt.xlabel("Step")
plt.ylabel("Score")
plt.yscale('log')
plt.title("LLM PPL Score vs Step")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"figures/ppl_curve_{exp_name}.png")

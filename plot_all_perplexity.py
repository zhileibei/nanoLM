import json
import re
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration
out_dirs = [
    "out-shakespeare-random-order-1perm-10ksteps",
    "out-shakespeare-random-order-2perm-10ksteps",
    "out-shakespeare-random-order-4perm-10ksteps",
    "out-shakespeare-random-order-8perm-10ksteps",
    "out-shakespeare-random-order-16perm-10ksteps",
]

# Labels for the legend
labels = ["1 perm", "2 perm", "4 perm", "8 perm", "16 perm"]

# regex to extract step number from filename
STEP_RE = re.compile(r"step(\d+)")

# Create figure
plt.figure(figsize=(10, 6))

# Process each directory
for out_dir, label in zip(out_dirs, labels):
    DIR = Path(out_dir) / "perplexity"

    if not DIR.exists():
        print(f"Warning: Directory {DIR} does not exist, skipping...")
        continue

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
    if steps:
        steps, scores = zip(*sorted(zip(steps, scores)))
        plt.plot(steps, scores, marker="o", label=label, markersize=4)
        print(f"{label}: {len(steps)} data points")

# Formatting
plt.xlabel("Step", fontsize=12)
plt.ylabel("Perplexity", fontsize=12)
plt.yscale('log')
plt.title("Perplexity vs Step - Random Order Permutations for AR models", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save figure
output_path = "perplexity_AR_all_random_order.png"
plt.savefig(output_path, dpi=150)
print(f"\nFigure saved to: {output_path}")
plt.show()

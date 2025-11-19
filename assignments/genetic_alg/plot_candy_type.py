import re
from collections import Counter
import matplotlib.pyplot as plt

# ---- READ FILE ----
with open("output.txt", "r") as f:
    text = f.read()

# ---- EXTRACT CANDY LISTS ----
fruit_raw = re.findall(r"Fruit:\[(.*?)\]", text, re.DOTALL)
choc_raw  = re.findall(r"Chocolate:\[(.*?)\]", text, re.DOTALL)

def parse(raw):
    items = re.findall(r"'([^']*)'|\"([^\"]*)\"", raw)
    return [a if a else b for a, b in items]

# Gather all candies
fruit_all = []
choc_all = []

for block in fruit_raw:
    fruit_all.extend(parse(block))

for block in choc_raw:
    choc_all.extend(parse(block))

# Count
fruit_counts = Counter(fruit_all)
choc_counts  = Counter(choc_all)

# Sort alphabetically for cleaner layout
fruit_items = sorted(fruit_counts.items())
choc_items  = sorted(choc_counts.items())

fruit_labels = [c for c, _ in fruit_items]
fruit_vals   = [v for _, v in fruit_items]

choc_labels = [c for c, _ in choc_items]
choc_vals   = [v for _, v in choc_items]

# ---- PLOT ----
plt.figure(figsize=(16, 6))

# Colors
fruit_color = "tab:pink"
choc_color  = "tab:brown"

# Concatenate positions so they appear left (fruit) then right (chocolate)
all_labels = fruit_labels + choc_labels
all_values = fruit_vals + choc_vals
colors     = [fruit_color]*len(fruit_labels) + [choc_color]*len(choc_labels)

plt.bar(range(len(all_labels)), all_values, color=colors)

plt.xticks(range(len(all_labels)), all_labels, rotation=90)

# Legend
fruit_patch = plt.Rectangle((0,0),1,1,color=fruit_color)
choc_patch  = plt.Rectangle((0,0),1,1,color=choc_color)
plt.legend([fruit_patch, choc_patch], ["Fruit", "Chocolate"])

plt.title("Candy Frequency Across GA Runs")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/candy_count_colored.png")
plt.clf()
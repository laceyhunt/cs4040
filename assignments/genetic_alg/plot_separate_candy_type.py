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

# Gather values
fruit_all = []
choc_all = []

for block in fruit_raw:
    fruit_all.extend(parse(block))

for block in choc_raw:
    choc_all.extend(parse(block))

# Count
fruit_counts = Counter(fruit_all)
choc_counts  = Counter(choc_all)

# Sort items alphabetically for consistency
fruit_items = sorted(fruit_counts.items())
choc_items  = sorted(choc_counts.items())

# ---- FRUIT BAR GRAPH ----
plt.figure(figsize=(10, 5))
labels = [c for c, _ in fruit_items]
values = [v for _, v in fruit_items]

plt.bar(labels, values, color="tab:pink")
plt.xticks(rotation=90)
plt.title("Fruit Candy Frequency Across GA Runs")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/fruit_frequency.png")
plt.clf()

# ---- CHOCOLATE BAR GRAPH ----
plt.figure(figsize=(10, 5))
labels = [c for c, _ in choc_items]
values = [v for _, v in choc_items]

plt.bar(labels, values, color="tab:brown")
plt.xticks(rotation=90)
plt.title("Chocolate Candy Frequency Across GA Runs")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/chocolate_frequency.png")
plt.clf()
import matplotlib.pyplot as plt

# ==== Per-class F1-scores (from your report) ====
classes = [
    "00 Normal",
    "01 Pneumonia",
    "02 Higher density",
    "03 Lower density",
    "04 Obstructive pulmonary diseases",
    "05 Degenerative infectious diseases",
    "06 Encapsulated lesions",
    "07 Mediastinal changes",
    "08 Chest changes"
]

f1_scores = [0.97, 0.94, 1.00, 1.00, 0.97, 1.00, 1.00, 1.00, 1.00]
supports = [133, 90, 34, 57, 51, 34, 66, 33, 39]

# ==== Plot ====
plt.figure(figsize=(12,6))
bars = plt.bar(classes, [s*100 for s in f1_scores],
               color="#5B8FF9", edgecolor="black", alpha=0.9)

# Add text labels on top
for bar, f1, sup in zip(bars, f1_scores, supports):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()*100 + 0.5,
             f"{f1*100:.1f}%\n(n={sup})", 
             ha="center", va="bottom", fontsize=9)

# Formatting
plt.xticks(rotation=30, ha="right", fontsize=10)
plt.ylabel("F1-score (%)", fontsize=12)
plt.title("Per-Class F1-score (LightXrayNet)", fontsize=14, weight="bold")
plt.ylim(85, 105)  # zoom in for better clarity
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("f1_per_class.png", dpi=300)
plt.show()

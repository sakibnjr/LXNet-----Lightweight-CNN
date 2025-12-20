import pandas as pd
import matplotlib.pyplot as plt

# === Raw data from your cross-validation ===
data = [
    # MedNetMini vs ResNet50V2
    ["MedNetMini",1,0.979013641133263,0.983716390514583,0.999480441869382],
    ["ResNet50V2",1,0.726128016789087,0.70736945538421,0.953298428348788],
    ["MedNetMini",2,0.983193277310924,0.986430002963422,0.999595189912885],
    ["ResNet50V2",2,0.727941176470588,0.706449704219238,0.952206316781489],
    ["MedNetMini",3,0.972689075630252,0.978203904005487,0.998844736337617],
    ["ResNet50V2",3,0.748949579831933,0.733431779750021,0.946685008616958],
    ["MedNetMini",4,0.984243697478992,0.985986013757084,0.999598490751834],
    ["ResNet50V2",4,0.763655462184874,0.750129010513953,0.953380002989574],
    ["MedNetMini",5,0.96953781512605,0.975493184292285,0.998034893127037],
    ["ResNet50V2",5,0.720588235294118,0.692624381732469,0.941423989657276],

    # Additional ResNet50V2, DenseNet201, InceptionV3
    ["ResNet50V2",1,0.738719832109129,0.72556056804162,0.953044315078856],
    ["DenseNet201",1,0.672612801678909,0.638078662391762,0.936268173563347],
    ["InceptionV3",1,0.694648478488982,0.678456188065836,0.9449717045405],
    ["ResNet50V2",2,0.724789915966387,0.698661229182313,0.952455622399334],
    ["DenseNet201",2,0.654411764705882,0.613918190255828,0.922841450550065],
    ["InceptionV3",2,0.707983193277311,0.706461325452113,0.941028125448352],
    ["ResNet50V2",3,0.739495798319328,0.723313356012602,0.946426416985073],
    ["DenseNet201",3,0.661764705882353,0.624842177758684,0.926740715558379],
    ["InceptionV3",3,0.723739495798319,0.708741768948877,0.938309716718757],
    ["ResNet50V2",4,0.769957983193277,0.754455824916957,0.954398218159165],
    ["DenseNet201",4,0.691176470588235,0.657355364462171,0.931099911935822],
    ["InceptionV3",4,0.744747899159664,0.723575116999897,0.947278414270394],
    ["ResNet50V2",5,0.725840336134454,0.707518956124554,0.941982319659557],
    ["DenseNet201",5,0.651260504201681,0.61195763898425,0.924170010659895],
    ["InceptionV3",5,0.707983193277311,0.679971340738339,0.939401819869001],
]

df = pd.DataFrame(data, columns=["model","fold","accuracy","f1_macro","auroc_macro"])

# === Summary statistics ===
summary = df.groupby("model").agg(["mean","std"])
print("\nSummary statistics:\n", summary)

# === Bar plot with error bars (one per metric) ===
metrics = ["accuracy", "f1_macro", "auroc_macro"]

for metric in metrics:
    means = df.groupby("model")[metric].mean()
    stds = df.groupby("model")[metric].std()
    
    plt.figure(figsize=(8,6))
    means.plot(kind="bar", yerr=stds, capsize=4, color="skyblue", edgecolor="black")
    plt.ylabel(metric.replace("_"," ").title())
    plt.title(f"Model Comparison – {metric.replace('_',' ').title()}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{metric}_barplot.png", dpi=300)  # High-res
    plt.close()

# === Boxplots across folds ===
for metric in metrics:
    plt.figure(figsize=(8,6))
    df.boxplot(column=metric, by="model", grid=False, patch_artist=True,
               boxprops=dict(facecolor="lightblue", color="black"),
               medianprops=dict(color="red", linewidth=2))
    plt.title(f"{metric.replace('_',' ').title()} Distribution Across Folds")
    plt.suptitle("")  # remove auto title
    plt.ylabel(metric.replace("_"," ").title())
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{metric}_boxplot.png", dpi=300)  # High-res
    plt.close()

# === Grouped bar chart (all metrics together) ===
means = df.groupby("model")[metrics].mean()
stds = df.groupby("model")[metrics].std()

ax = means.plot(kind="bar", yerr=stds, capsize=4, figsize=(10,7))
plt.ylabel("Score")
plt.title("Model Comparison Across Metrics (Mean ± SD)")
plt.xticks(rotation=20, ha="right")
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig("all_metrics_grouped_bar.png", dpi=300)
plt.close()

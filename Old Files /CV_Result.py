import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {
    "Model": ["LXNet","ResNet50V2","InceptionV3","DenseNet201",
              "LXNet","ResNet50V2","InceptionV3","DenseNet201",
              "LXNet","ResNet50V2","InceptionV3","DenseNet201",
              "LXNet","ResNet50V2","InceptionV3","DenseNet201",
              "LXNet","ResNet50V2","InceptionV3","DenseNet201"],
    "Fold": [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5],
    "Accuracy": [0.985185185185185,0.588271604938272,0.699382716049383,0.509876543209877,
                 0.962345679012346,0.604938271604938,0.72037037037037,0.52962962962963,
                 0.982098765432099,0.580864197530864,0.694444444444444,0.500617283950617,
                 0.98641975308642,0.612345679012346,0.707407407407407,0.530864197530864,
                 0.981481481481482,0.597530864197531,0.70679012345679,0.506172839506173]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Filter LXNet only
lightxray_df = df[df['Model'] == 'LXNet'].copy()
lightxray_df['Accuracy (%)'] = lightxray_df['Accuracy'] * 100  # convert to percentage

# Plot
plt.figure(figsize=(10, 6), dpi=300)
bars = plt.bar(lightxray_df['Fold'], lightxray_df['Accuracy (%)'], color='skyblue')

# Set y-axis from 0 to 110 with ticks only at 20, 40, 60, 80, 100
plt.ylim(0, 110)
plt.yticks([20, 40, 60, 80, 100])

plt.xticks(lightxray_df['Fold'])
plt.xlabel('Fold', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('LXNet Accuracy per Fold', fontsize=14)

# Add percentage labels above bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 2, f'{height:.2f}%', 
             ha='center', va='bottom', fontsize=10)

# Remove grid lines
plt.grid(False)

plt.tight_layout()
plt.savefig('LXNet_PerFold_Accuracy.png', dpi=300)
plt.show()

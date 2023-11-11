import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Read the CSV
df = pd.read_csv("val_with_metrics.csv")


numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=["Unnamed: 0"])

dim_reduc = PCA(n_components=2)
reduced_data = dim_reduc.fit_transform(numeric_df)
loadings = pd.DataFrame(dim_reduc.components_.T, columns=['PC1', 'PC2'], index=numeric_df.columns)

colors = [0] * len(df)
with open("true_labels.thingy", "r", encoding="utf-8") as file:
    for line in file.readlines():
        colors[int(line.split()[0])] = int(line.split()[1]) + 1

breakpoint()

# # Plotting feature importance for each PCA component
# for component in loadings.columns:
#     plt.figure(figsize=(10,6))
#     loadings[component].sort_values().plot(kind='barh')
#     plt.title(f"Feature Importance for {component}")
#     plt.xlabel("Loading")
#     plt.ylabel("Features")
#     plt.show()

plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=colors)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Clustered Data Visualization")
plt.show()

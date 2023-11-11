import json
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Read the CSV
df = pd.read_csv("val_with_metrics.csv")


numeric_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=["Unnamed: 0"])

dim_reduc = TSNE(n_components=2)
reduced_data = dim_reduc.fit_transform(numeric_df)

kmeans = KMeans(n_clusters=2, n_init="auto")
clustered_data = kmeans.fit_predict(reduced_data)

with open("cached_res.json", "w", encoding="utf-8") as file:
    json.dump([int(x) for x in clustered_data], file)

for i, text in enumerate(df.text):
    if "..." in text:
        clustered_data[i] = 2
    # if reduced_data[i, 1] > 15:
    #     print(text)
    #     print("----")



plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clustered_data)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Clustered Data Visualization")
plt.show()

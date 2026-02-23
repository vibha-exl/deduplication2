import torch
from kmeans_gpu import KMeans

data = torch.load("embeddings.pt")
emb = data["embeddings"].cuda()    

features = emb.T.unsqueeze(0)       

kmeans = KMeans(
    n_clusters=10,
    max_iter=100,
    tolerance=1e-4,
    distance="euclidean"
)

centroids, features_out = kmeans(points, features)
labels = torch.argmax(features_out, dim=1).squeeze(0).cpu()
print("Centroids shape:", centroids.shape)
print("Labels shape:", labels.shape)
print("Unique clusters:", labels.unique())

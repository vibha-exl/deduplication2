import torch
from kmeans_gpu import KMeans

# ---- Load ----
ckpt = torch.load("embeddings.pt", map_location="cpu")
emb = ckpt["embeddings"]          # (N, D)
texts = ckpt["texts"]             # list length N
assert emb.shape[0] == len(texts)

device = "cuda" if torch.cuda.is_available() else "cpu"
emb = emb.float().to(device)      # float32 on GPU is safest

N, D = emb.shape

# ---- Adapt to library input shapes ----
# points:   (B, num_pts, pts_dim) = (1, N, D)
# features: (B, feature_dim, num_pts) = (1, D, N)  (reuse embeddings as features)
points = emb.unsqueeze(0)         # (1, N, D)
features = emb.t().unsqueeze(0)   # (1, D, N)

# ---- KMeans ----
K = 15
kmeans = KMeans(
    n_clusters=K,
    max_iter=100,
    tolerance=1e-4,
    distance="euclidean",
    sub_sampling=None,
    max_neighbors=15,
).to(device)

centroids, _ = kmeans(points, features)    # centroids: (1, K, D)
centroids = centroids[0]                  # (K, D)

# ---- Labels: nearest centroid ----
# (N, K) distances -> (N,) labels
dists = torch.cdist(emb, centroids)       # Euclidean
labels = torch.argmin(dists, dim=1)       # (N,)

labels_cpu = labels.detach().cpu().tolist()
rows = [{"text": t, "cluster": c, "embeddings":em} for t, c, em in zip(texts, labels_cpu,emb)]
import pandas as pd

df = pd.DataFrame(rows)
df.to_csv("clustered_texts.csv", index=False)
 
import pandas as pd
import numpy as np
import ast
import re
from sklearn.metrics.pairwise import cosine_similarity

print("Starting inspection...")

df = pd.read_csv("clustered_texts.csv")
print("Loaded CSV:", df.shape)

def parse_embedding(emb_str):
    # Remove tensor(...) and device info safely
    emb_str = re.sub(r"tensor\(|device=.*?\)", "", emb_str)
    return np.array(ast.literal_eval(emb_str), dtype=np.float32)

df["embedding_vec"] = df["embeddings"].apply(parse_embedding)

print("Parsed embeddings")

print("\nCluster sizes:")
print(df["cluster"].value_counts())

for cluster_id, group in df.groupby("cluster"):
    print("\n" + "=" * 70)
    print(f"Cluster {cluster_id} | size={len(group)}")

    if len(group) < 2:
        print("Skipping cluster (size < 2)")
        continue

    emb = np.vstack(group["embedding_vec"].values)
    sims = cosine_similarity(emb)

    iu = np.triu_indices(len(group), k=1)
    pair_sims = sims[iu]

    print(f"Max similarity : {pair_sims.max():.4f}")
    print(f"Mean similarity: {pair_sims.mean():.4f}")

    # show top 3
    top = np.argsort(pair_sims)[-3:][::-1]
    rows = group.reset_index(drop=True)

    for idx in top:
        i, j = iu[0][idx], iu[1][idx]
        print(f"\nSimilarity = {pair_sims[idx]:.4f}")
        print("A:", rows.loc[i, "text"][:120])
        print("B:", rows.loc[j, "text"][:120])

print("\nInspection complete.")

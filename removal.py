import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity

INPUT_CSV = "clustered_texts.csv"
OUTPUT_KEPT = "deduplicated_texts.csv"
OUTPUT_AUDIT = "deduplication_audit.csv"
SIM_THRESHOLD = 0.92


print("Loading CSV...")
df = pd.read_csv(INPUT_CSV)

def parse_embedding(emb_str):
    """
    Converts stringified tensor into numpy vector
    """
    emb_str = emb_str.replace("tensor(", "")
    emb_str = emb_str.replace(")", "")
    emb_str = emb_str.split("device=")[0]
    return np.array(ast.literal_eval(emb_str), dtype=np.float32)

print("Parsing embeddings...")
df["embedding_vec"] = df["embeddings"].apply(parse_embedding)

kept_rows = []
audit_rows = []

for cluster_id, group in df.groupby("cluster"):
    group = group.reset_index(drop=False)  # keep original row index
    n = len(group)

    # --- Single item cluster ---
    if n == 1:
        row = group.iloc[0]
        kept_rows.append(row)

        audit_rows.append({
            "cluster": cluster_id,
            "row_index": row["index"],
            "text": row["text"],
            "status": "kept",
            "reason": "single_item",
            "similarity": None,
            "compared_to": None
        })
        continue

    # --- Build embedding matrix ---
    X = np.stack([e.squeeze() for e in group["embedding_vec"].values])
 # (N, D)
    sim_matrix = cosine_similarity(X, X)

    # --- Pick cluster representative (most central row) ---
    mean_sims = sim_matrix.mean(axis=1)
    rep_idx = mean_sims.argmax()

    rep_row = group.iloc[rep_idx]
    kept_rows.append(rep_row)

    audit_rows.append({
        "cluster": cluster_id,
        "row_index": rep_row["index"],
        "text": rep_row["text"],
        "status": "kept",
        "reason": "cluster_representative",
        "similarity": mean_sims[rep_idx],
        "compared_to": None
    })

    # --- Compare others to representative ---
    for i in range(n):
        if i == rep_idx:
            continue

        row = group.iloc[i]
        sim = sim_matrix[rep_idx, i]

        if sim >= SIM_THRESHOLD:
            audit_rows.append({
                "cluster": cluster_id,
                "row_index": row["index"],
                "text": row["text"],
                "status": "removed",
                "reason": "too_similar",
                "similarity": sim,
                "compared_to": rep_row["index"]
            })
        else:
            kept_rows.append(row)
            audit_rows.append({
                "cluster": cluster_id,
                "row_index": row["index"],
                "text": row["text"],
                "status": "kept",
                "reason": "distinct_enough",
                "similarity": sim,
                "compared_to": rep_row["index"]
            })

    print(
        f"Cluster {cluster_id}: "
        f"kept={sum(r['cluster']==cluster_id and r['status']=='kept' for r in audit_rows)}, "
        f"removed={sum(r['cluster']==cluster_id and r['status']=='removed' for r in audit_rows)}"
    )

# =========================
# SAVE OUTPUTS
# =========================
kept_df = pd.DataFrame(kept_rows).drop(columns=["embedding_vec"])
audit_df = pd.DataFrame(audit_rows)

kept_df.to_csv(OUTPUT_KEPT, index=False)
audit_df.to_csv(OUTPUT_AUDIT, index=False)

print("Deduplication complete")
print("Final kept rows:", len(kept_df))
print("Audit rows:", len(audit_df))

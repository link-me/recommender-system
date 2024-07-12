import numpy as np
from typing import Tuple, List, Dict, Any

try:
    import pandas as pd  # type: ignore
    HAS_PANDAS = True
except Exception:
    pd = None  # type: ignore
    HAS_PANDAS = False


def load_interactions(csv_path: str):
    """Load interactions from CSV.

    If pandas is available, returns a DataFrame.
    Otherwise, returns a list of dict records: {user_id, item_id, rating}.
    """
    import csv

    if HAS_PANDAS:
        df = pd.read_csv(csv_path)
        cols = set(df.columns)
        if {"user_id", "item_id"}.issubset(cols) and "rating" not in cols:
            df["rating"] = 1.0
        expected = {"user_id", "item_id", "rating"}
        missing = expected - cols
        if missing:
            raise ValueError(f"CSV must contain columns {expected}, missing: {missing}")
        df["user_id"] = df["user_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
        return df
    # Fallback without pandas
    records: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user = str(row.get("user_id", "")).strip()
            item = str(row.get("item_id", "")).strip()
            rating_str = row.get("rating")
            try:
                rating = float(rating_str) if rating_str not in (None, "") else 1.0
            except Exception:
                rating = 0.0
            if not user or not item:
                raise ValueError("Rows must contain user_id and item_id")
            records.append({"user_id": user, "item_id": item, "rating": rating})
    return records


def build_user_item_matrix(df_or_records) -> Tuple[np.ndarray, List[str], List[str]]:
    """Return (matrix, users, items) where matrix is float numpy array.

    Works with pandas DataFrame or list-of-dicts fallback.
    """
    if HAS_PANDAS and isinstance(df_or_records, pd.DataFrame):
        mat_df = df_or_records.pivot_table(index="user_id", columns="item_id", values="rating", aggfunc="mean", fill_value=0.0)
        mat = mat_df.astype(float).values
        users = list(mat_df.index)
        items = list(mat_df.columns)
        return mat, users, items
    # Fallback: manual build
    recs: List[Dict[str, Any]] = df_or_records
    users = sorted({r["user_id"] for r in recs})
    items = sorted({r["item_id"] for r in recs})
    u_index = {u: i for i, u in enumerate(users)}
    it_index = {it: j for j, it in enumerate(items)}
    mat = np.zeros((len(users), len(items)), dtype=float)
    counts = np.zeros_like(mat)
    for r in recs:
        i = u_index[r["user_id"]]
        j = it_index[r["item_id"]]
        mat[i, j] += float(r["rating"])
        counts[i, j] += 1
    # average if multiple interactions per cell
    nonzero = counts > 0
    mat[nonzero] = mat[nonzero] / counts[nonzero]
    return mat, users, items


def _cosine_sim_matrix(mat: np.ndarray) -> np.ndarray:
    # Normalize rows to unit length to compute cosine similarity via dot product
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    mat_norm = mat / norms
    # Cosine similarity between all users
    return mat_norm @ mat_norm.T


def recommend_for_user(user_id: str, mat_np: np.ndarray, users: List[str], items: List[str], top_n: int = 5, exclude_seen: bool = True) -> List[Tuple[str, float]]:
    if user_id not in users:
        raise KeyError(f"Unknown user_id: {user_id}")
    user_idx = users.index(user_id)

    # Similarities between the target user and others
    sim = _cosine_sim_matrix(mat_np)[user_idx]
    # Zero self-sim to avoid leaking own ratings
    sim[user_idx] = 0.0

    # Weighted sum of other users' ratings
    # score[item] = sum(sim[u] * rating[u,item]) / sum(sim[u] where rating[u,item] > 0)
    ratings = mat_np
    num = sim @ ratings  # shape [items]
    denom = (sim > 0).astype(float) @ (ratings > 0).astype(float)
    # Avoid division by zero
    denom = np.where(denom == 0, 1.0, denom)
    scores = num / denom

    # Optionally exclude seen items
    if exclude_seen:
        seen_mask = ratings[user_idx] > 0
        scores = np.where(seen_mask, -np.inf, scores)

    # Top-N items
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [(items[i], float(scores[i])) for i in top_indices if np.isfinite(scores[i]) and scores[i] > 0]


def popular_items(data, top_n: int = 5) -> List[Tuple[str, float]]:
    """Top-N popular items by total rating.

    Works with pandas DataFrame or list-of-dicts.
    """
    if HAS_PANDAS and isinstance(data, pd.DataFrame):
        pop = data.groupby("item_id")["rating"].sum().sort_values(ascending=False)
        return [(item, float(score)) for item, score in pop.head(top_n).items()]
    # Fallback
    totals: Dict[str, float] = {}
    for r in data:
        totals[r["item_id"]] = totals.get(r["item_id"], 0.0) + float(r["rating"])
    sorted_items = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [(it, float(val)) for it, val in sorted_items]
from tokenizer import BPE

tokenizer = BPE()

text = """
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import solve  # from scipy (optional, but allowed)


# -------------------------------------------------------
# 2. Load dataset and build rating matrix
# -------------------------------------------------------

def load_rating_matrix(csv_path):
    "
    Load a CSV with columns: user_id, item_id, rating
    and build a dense user-item rating matrix R
    where missing ratings are 0.
    "
    df = pd.read_csv(csv_path)

    num_users = df["user_id"].max() + 1
    num_items = df["item_id"].max() + 1

    R = np.zeros((num_users, num_items), dtype=np.float64)

    for row in df.itertuples(index=False):
        u = row.user_id
        i = row.item_id
        r = row.rating
        R[u, i] = r

    return R


# -------------------------------------------------------
# 3. ALS implementation
# -------------------------------------------------------

def als(
    R,
    num_factors=10,
    reg=0.1,
    num_iters=20,
    verbose=True,
):
    "
    Alternating Least Squares for matrix factorization.

    Args:
        R           : (m x n) rating matrix with 0 for missing entries
        num_factors : number of latent factors (k)
        reg         : regularization parameter (lambda)
        num_iters   : number of ALS iterations
        verbose     : print RMSE each iteration

    Returns:
        U           : (m x k) user latent factor matrix
        V           : (n x k) item latent factor matrix
        rmse_hist   : list of RMSE values per iteration
    "
    m, n = R.shape
    k = num_factors

    # Initialize U and V randomly
    rng = np.random.default_rng(0)
    U = rng.normal(0, 0.1, size=(m, k))
    V = rng.normal(0, 0.1, size=(n, k))

    # Mask of observed ratings
    mask = R > 0

    rmse_hist = []

    for it in range(num_iters):
        # ----- Update U (user factors) -----
        for u in range(m):
            idx_items = mask[u, :]  # items rated by user u
            if not np.any(idx_items):
                continue

            V_i = V[idx_items, :]        # (#items_u, k)
            r_u = R[u, idx_items]        # (#items_u,)

            # Solve (V_i^T V_i + reg * I) u_u = V_i^T r_u
            A = V_i.T @ V_i + reg * np.eye(k)
            b = V_i.T @ r_u

            # Use scipy.linalg.solve or np.linalg.solve
            U[u, :] = solve(A, b)

        # ----- Update V (item factors) -----
        for i in range(n):
            idx_users = mask[:, i]  # users who rated item i
            if not np.any(idx_users):
                continue

            U_u = U[idx_users, :]       # (#users_i, k)
            r_i = R[idx_users, i]       # (#users_i,)

            # Solve (U_u^T U_u + reg * I) v_i = U_u^T r_i
            A = U_u.T @ U_u + reg * np.eye(k)
            b = U_u.T @ r_i

            V[i, :] = solve(A, b)

        # ----- Compute training RMSE -----
        R_hat = U @ V.T
        diff = R[mask] - R_hat[mask]
        rmse = np.sqrt(np.mean(diff**2))
        rmse_hist.append(rmse)

        if verbose:
            print(f"Iteration {it + 1:2d}/{num_iters}, RMSE = {rmse:.4f}")

    return U, V, rmse_hist


# -------------------------------------------------------
# 4. Main script: generate data, run ALS, plot
# -------------------------------------------------------

def main():
    csv_file = "data.csv"

    # 1) Generate synthetic dataset if not already present
    if not os.path.exists(csv_file):
        generate_sample_dataset(filename=csv_file)
    else:
        print(f"Found existing dataset: '{csv_file}'")

    # 2) Load rating matrix
    print("Loading rating matrix...")
    R = load_rating_matrix(csv_file)
    print(f"Rating matrix shape: {R.shape}")

    # 3) Run ALS
    num_factors = 10   # latent dimension k
    reg = 0.1          # regularization
    num_iters = 90     # number of ALS iterations

    print("\nRunning ALS...")
    U, V, rmse_hist = als(
        R,
        num_factors=num_factors,
        reg=reg,
        num_iters=num_iters,
        verbose=True,
    )

    # 4) Plot RMSE over iterations
    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(1, len(rmse_hist) + 1), rmse_hist, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("ALS Convergence (Training RMSE)")
    plt.tight_layout()
    plt.show()

    # 5) Example: show predicted rating for a user & item
    user_id = 0
    item_id = 0
    R_hat = U @ V.T
    print(f"\nExample prediction:")
    print(f"Predicted rating for user {user_id}, item {item_id}: {R_hat[user_id, item_id]:.3f}")
    if R[user_id, item_id] > 0:
        print(f"True rating (from data): {R[user_id, item_id]:.3f}")
    else:
        print(f"User {user_id} did not rate item {item_id} in the dataset.")


if __name__ == "__main__":
    main()

"""

text_encoded = tokenizer.encoder(text=text)
tokens_decoded = tokenizer.decode(text_encoded)

print(f"Does the Python Code Remain Same?\nAnswer: {text==tokens_decoded}")
print(f"\nLength Of Actual Code:{len(text)}")
print(f"Length of Tokenized Code: {len(text_encoded)}")

percent_reduced = 100*(1 - (len(text_encoded)/len(text)))
print(f"Percentage of Tokens Reduced: {percent_reduced:.2f}%")



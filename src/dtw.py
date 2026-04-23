import numpy as np

def dtw_distance(a, b):
    n, m = len(a), len(b)
    dp = np.full((n+1, m+1), np.inf)
    dp[0,0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(a[i-1] - b[j-1])
            dp[i,j] = cost + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])

    return dp[n,m]


def compute_dtw_matrix(seqs):
    print("[DTW] Computing pairwise DTW matrix...")

    n = len(seqs)
    mat = np.zeros((n,n))

    for i in range(n):
        if i % 100 == 0:
            print(f"[DTW] Processing {i}/{n}")

        for j in range(i+1, n):
            d = dtw_distance(seqs[i], seqs[j])
            mat[i,j] = mat[j,i] = d

    print("[DTW] Completed DTW matrix")
    return mat
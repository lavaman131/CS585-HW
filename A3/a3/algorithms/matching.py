from typing import Tuple
import numpy as np


class HungarianMatcher:
    @staticmethod
    def pad_matrix(A: np.ndarray) -> np.ndarray:
        n, m = A.shape
        max_value = np.max(A) * 10
        if n > m:
            # More rows than columns, add columns
            padding = np.full((n, n - m), max_value)
            A_padded = np.hstack((A, padding))
        elif n < m:
            # More columns than rows, add rows
            padding = np.full((m - n, m), max_value)
            A_padded = np.vstack((A, padding))
        else:
            # The matrix is already square, no padding needed
            A_padded = A
        return A_padded

    def __call__(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        original_n, original_m = A.shape
        A_padded = HungarianMatcher.pad_matrix(A)
        n, m = A_padded.shape
        u, v = np.zeros(n + 1), np.zeros(m + 1)
        p, way = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
        for i in range(1, n + 1):
            p[0] = i
            minv = np.full(m + 1, np.inf)
            used = np.zeros(m + 1, dtype=bool)
            j0 = 0
            while True:
                used[j0] = True
                i0, delta = p[j0], np.inf
                for j in range(1, m + 1):
                    if not used[j]:
                        cur = A_padded[i0 - 1, j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j], way[j] = cur, j0
                        if minv[j] < delta:
                            delta, j1 = minv[j], j
                for j in range(m + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while j0:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1

        # Adjust for zero-based indexing and filter out padding
        ans = (
            np.zeros(original_n, dtype=int) - 1
        )  # Initialize with -1 to indicate no assignment
        for j in range(1, min(original_m + 1, m + 1)):
            if p[j] - 1 < original_n:
                ans[p[j] - 1] = j - 1

        # Find the rows that have been assigned
        row_ind = np.where(ans >= 0)[0]
        col_ind = ans[row_ind]

        return row_ind, col_ind

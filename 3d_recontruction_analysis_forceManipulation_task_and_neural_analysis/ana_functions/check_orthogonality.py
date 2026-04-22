
import pandas as pd
import numpy as np


# check the orthogonality among the three behavioral vectors
def check_orthogonality(*vectors, tol=1e-6):
    n = len(vectors)
    for i in range(n):
        for j in range(i + 1, n):
            dot = np.dot(vectors[i], vectors[j])
            print(f"Dot product between vector {i} and {j}: {dot:.6f}")
            if abs(dot) > tol:
                print("⚠️ Not orthogonal!")
            else:
                print("✅ Orthogonal")
                
#  Gram-Schmidt Orthogonalization
def gram_schmidt(vectors):
    """Orthogonalize a list of vectors using the Gram-Schmidt process."""
    orthogonal_vectors = []
    for v in vectors:
        # Subtract projection onto all previous orthogonal vectors
        for u in orthogonal_vectors:
            v = v - np.dot(v, u) * u
        # Normalize the vector
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            orthogonal_vectors.append(v / norm)
    return orthogonal_vectors

from verifier import compute_m_height
import numpy as np

# Generate a wider 3x8 generator matrix to allow larger m
# G = np.array([
#     [1, 0, 0, 1, 2, -1, 1, 0],
#     [0, 1, 0, -1, 1, 2, 0, 1],
#     [0, 0, 1, 2, 1, -1, 1, -1]
# ])

# Create a 4x4 identity matrix
identity = np.eye(4)

# Combine with the given matrix to form a 4x9 generator matrix
G = np.hstack((identity, np.array([
    [-1,  1,  2,  2,  1],
    [ 1,  2,  2,  1, -1],
    [ 2,  2,  1, -1,  1],
    [ 2,  1, -1,  1,  2]
])))


# Set m to 4 (since n - k = 8 - 3 = 5, so m=4 is valid)
m = 4

# Test the verifier with wider matrix
try:
    minimal_height = compute_m_height(G, m)
    print(f"Minimal m-height for matrix:\n{G}\nwith m={m} is: {minimal_height}")
except ValueError as e:
    print(f"Error: {e}")


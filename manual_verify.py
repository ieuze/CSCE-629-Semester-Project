from verifier import compute_m_height
import numpy as np

# This function is used to manually verify the minimal m-height of a given generator matrix

# Generate a wider 3x8 generator matrix to allow larger m
# G = np.array([
#     [1, 0, 0, 1, 2, -1, 1, 0],
#     [0, 1, 0, -1, 1, 2, 0, 1],
#     [0, 0, 1, 2, 1, -1, 1, -1]
# ])
# Create a 4x4 identity matrix
# Create a 5x5 identity matrix
identity = np.eye(5)

# Combine with the given matrix to form a 5x9 generator matrix
G = np.hstack((identity, np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 0],
    [1, 1, 0, 1],
    [1, 1, 0, 0],
    [1, 0, 1, 0]
])))

# Set m to 2 (since n - k = 9 - 5 = 4, so m=2 is valid)
m = 2

# Test the verifier with wider matrix
try:
    minimal_height = compute_m_height(G, m)
    print(f"Minimal m-height for matrix:\n{G}\nwith m={m} is: {minimal_height}")
except ValueError as e:
    print(f"Error: {e}")


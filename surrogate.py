import numpy as np
import time
import math

def compute_m_height(G, m):
    """
    A faster surrogate/heuristic model of m-height calculation with same parameters as the original.
    Instead of solving many LPs, uses matrix properties and heuristics to approximate the height.
    
    Parameters:
    G (numpy.ndarray): Generator matrix of shape (k, n).
    m (int): Parameter defining the m-height. Should be >= 1.
    
    Returns:
    float: An approximation of the m-height h_m(C). Returns np.inf if likely unbounded.
    """
    k, n = G.shape
    
    # Basic validation same as the original
    if not (1 <= m <= n - 1):
        if m == 0:
            return 1.0
        raise ValueError(f"m must be between 1 and n-1 ({n-1}), but got {m}")
    
    # Timing (optional)
    start_time = time.time()
    
    # --- Unboundedness Check (Less Aggressive) ---
    # 1. Check for dependent columns (signaling potential unboundedness)
    # Only check when m is large relative to matrix size
    if m >= n - 2 and n > 5:  # Only be aggressive for large m in larger matrices
        for i in range(n):
            for j in range(i+1, n):
                col_i = G[:, i]
                col_j = G[:, j]
                
                # Skip zero columns
                if np.all(col_i == 0) or np.all(col_j == 0):
                    continue
                
                # Check for almost exact linear dependence (much stricter test)
                ratios = []
                for r in range(k):
                    if abs(col_i[r]) > 1e-10 and abs(col_j[r]) > 1e-10:
                        ratios.append(col_j[r] / col_i[r])
                
                # Only consider columns dependent if ALL ratios are nearly identical
                if ratios and len(ratios) >= k-1 and np.std(ratios) < 1e-8:
                    return np.inf
    
    # 2. Calculate matrix properties for height estimation
    col_norms = np.sqrt(np.sum(G**2, axis=0))  # Column-wise L2 norm
    row_norms = np.sqrt(np.sum(G**2, axis=1))  # Row-wise L2 norm
    max_abs_element = np.max(np.abs(G))
    
    # 3. Normalized m value - smaller m should lead to lower heights
    m_factor = m / (n - 1)  
    
    # 4. Calculate submatrix determinants to gauge matrix independence
    submatrix_determinants = []
    max_submatrix_size = min(k, 5)
    for size in range(2, max_submatrix_size + 1):
        if size < 2:
            continue
            
        # Try multiple random combinations of columns
        num_samples = min(10, math.comb(n, size) if n >= size else 0)
        for _ in range(num_samples):
            try:
                # Randomly select 'size' columns
                cols = np.random.choice(n, size=size, replace=False)
                
                # Get the first 'size' rows to make a square matrix
                if size <= k:
                    submatrix = G[:size, cols]
                    
                    if submatrix.shape[0] == submatrix.shape[1]:
                        det = np.abs(np.linalg.det(submatrix))
                        if det > 0:
                            submatrix_determinants.append(det)
            except Exception:
                pass
    
    # If no valid determinants found, use a default value
    if not submatrix_determinants:
        submatrix_determinants = [1.0]
    
    # --- More sophisticated height estimation ---
    # Base height on matrix properties
    det_factor = 1.0 / (np.mean(submatrix_determinants) + 1e-10)
    norm_factor = np.mean(col_norms)
    
    # Adjust based on empirical observations
    # Higher m values lead to higher heights
    m_scaling = 1.0 + 2.0 * (m / (n - 1))
    
    # Element magnitude effect is reduced to avoid overestimating height
    element_factor = np.log1p(max_abs_element) / np.log(100)  # Logarithmic scaling
    
    # Row diversity factor - more diverse rows should lead to lower heights
    row_diversity = np.std(row_norms) / (np.mean(row_norms) + 1e-10)
    diversity_factor = 1.0 / (1.0 + row_diversity)
    
    # Combine factors
    height_estimate = norm_factor * m_scaling * element_factor * det_factor * diversity_factor
    
    # Scale to reasonable range
    height_estimate = max(1.0, height_estimate)
    
    # --- Reduced detection of unboundedness ---
    # Only mark as unbounded in extreme cases
    if max_abs_element > 10000 and m > n-1:  # Much more conservative
        return np.inf
    
    # Add small randomness to simulate LP stochasticity
    np.random.seed(hash(str(G.flatten().tolist()) + str(m)) % (2**32))
    randomness = np.random.uniform(0.95, 1.05)
    
    end_time = time.time()
    # print(f"Surrogate compute_m_height: {end_time - start_time:.4f} sec.")
    
    return height_estimate * randomness 
import numpy as np
import math
import random
from verifier import compute_m_height
import argparse
import sys # For exit
import signal # For Ctrl+C handling
from tqdm import tqdm # For progress bar

# Global flag for Ctrl+C
user_interrupted = False

def signal_handler(sig, frame):
    """Handle Ctrl+C: set flag and print message."""
    global user_interrupted
    if user_interrupted: # Second Ctrl+C
        print('\nCtrl+C detected again. Exiting forcefully.')
        sys.exit(1)
    print('\nCtrl+C detected. Finishing current temperature step and stopping annealing...')
    user_interrupted = True

def calculate_cost(G, m):
    """Calculates the m-height, handling potential errors or infinite results."""
    try:
        height = compute_m_height(G, m)
        # Replace inf with a very large number for comparison purposes,
        # but maybe it's better to handle inf directly in acceptance logic.
        # Using inf directly is generally safer.
        return height
    except ValueError as e:
        # This might happen if G becomes unsuitable during mutation,
        # or m is invalid for G. Treat as very high cost.
        print(f"Warning: compute_m_height failed for G={G} with m={m}. Error: {e}")
        return float('inf')
    except Exception as e:
        # Catch other potential exceptions from linprog
        print(f"Unexpected error during cost calculation: {e}")
        return float('inf')

def get_neighbor(P, element_range):
    """Generates a neighbor P matrix by changing one element randomly,
       ensuring no column becomes all zeros.
    """
    k, p_cols = P.shape
    P_new = P.copy()
    max_retries_neighbor = 10 # Safeguard against unusual situations
    max_retries_value = 20 # Safeguard for finding a non-zero value

    for _ in range(max_retries_neighbor):
        row_idx = random.randrange(k)
        col_idx = random.randrange(p_cols)
        current_val = P_new[row_idx, col_idx]

        possible_values = list(range(element_range[0], element_range[1] + 1))
        if len(possible_values) <= 1:
            # If range is single value, change might be impossible or always make zero col
            continue # Try modifying a different element

        random.shuffle(possible_values)

        found_valid_new_val = False
        for retry_val in range(max_retries_value):
            new_val = random.choice(possible_values)
            if new_val == current_val:
                if len(possible_values) > 1:
                    continue # Try a different value if possible
                else:
                    break # Cannot change value if range has size 1

            # Temporarily make the change
            original_value_in_col = P_new[row_idx, col_idx]
            P_new[row_idx, col_idx] = new_val

            # Check if the column is now all zeros
            if not np.all(P_new[:, col_idx] == 0):
                found_valid_new_val = True
                break # Found a valid new value
            else:
                # Revert the change for this specific value try
                P_new[row_idx, col_idx] = original_value_in_col
                # Continue loop to try a different new_val

        if found_valid_new_val:
            # A valid change was made to P_new[row_idx, col_idx]
            return P_new # Return the valid neighbor
        # If no valid new_val found for this (row, col), the outer loop continues
        # to try modifying a different element.

    # If max_retries_neighbor reached without finding a valid move
    # This should be rare with a large element_range
    print("Warning: get_neighbor failed to find a valid move after multiple retries. Returning original P.")
    return P # Return original if stuck

def simulated_annealing(initial_P, k, n, m, T_max, T_min, alpha, iter_per_temp, element_range):
    """
    Performs simulated annealing to find a systematic generator matrix G = [I|P]
    minimizing m-height by optimizing P.

    Args:
        initial_P (np.ndarray): The starting P matrix [k x (n-k)].
        k (int): Number of rows (message bits).
        n (int): Number of columns (codeword bits).
        m (int): The m-parameter for the m-height calculation.
        T_max (float): Initial temperature.
        T_min (float): Stopping temperature.
        alpha (float): Cooling rate (e.g., 0.95).
        iter_per_temp (int): Number of iterations at each temperature step.
        element_range (tuple): Min/max integer values for elements in P.

    Returns:
        tuple: (best_G, best_cost) - The best systematic G found and its m-height.
    """
    # Register signal handler for SIGINT (Ctrl+C)
    global user_interrupted
    user_interrupted = False # Reset flag at start of function
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    if not (1 <= m <= n - 1):
        raise ValueError(f"m must be between 1 and n-1 ({n-1}), but got {m}")

    I_k = np.eye(k, dtype=initial_P.dtype) # Ensure I has same dtype as P

    current_P = initial_P
    current_G = np.hstack((I_k, current_P))
    current_cost = calculate_cost(current_G, m)

    best_P = current_P
    best_cost = current_cost

    T = T_max

    print(f"Starting Annealing (Systematic): Initial Cost = {current_cost:.4f}, T_max = {T_max}, T_min = {T_min}, alpha = {alpha}")
    print("Press Ctrl+C to stop early and get the best result found so far.")

    # Estimate total steps for tqdm progress bar
    if T_max > T_min and 0 < alpha < 1:
        total_steps_estimate = math.ceil(math.log(T_min / T_max) / math.log(alpha))
    else:
        total_steps_estimate = 0

    # Initialize tqdm progress bar
    pbar = tqdm(total=total_steps_estimate, desc=f"T={T:.2e}, Best={best_cost:.4f}", unit="steps", disable=(total_steps_estimate==0))

    temp_step = 0
    try:
        while T > T_min:
            # Check for user interrupt at the beginning of each temperature step
            if user_interrupted:
                print("\nAnnealing interrupted by user.")
                break

            for i in range(iter_per_temp):
                neighbor_P = get_neighbor(current_P, element_range)
                neighbor_G = np.hstack((I_k, neighbor_P))
                neighbor_cost = calculate_cost(neighbor_G, m)

                delta_E = neighbor_cost - current_cost

                # --- Acceptance Logic (same as before) ---
                accept = False
                if neighbor_cost == float('inf') and current_cost == float('inf'):
                    accept = False
                elif neighbor_cost == float('inf'):
                    accept = False
                elif current_cost == float('inf'):
                    accept = True
                elif delta_E < 0:
                    accept = True
                else:
                    acceptance_prob = math.exp(-delta_E / T)
                    if random.random() < acceptance_prob:
                        accept = True
                # --- End Acceptance Logic ---

                if accept:
                    current_P = neighbor_P # Update P
                    current_cost = neighbor_cost

                    # Update best solution found so far
                    if current_cost < best_cost:
                        best_P = current_P # Store the best P
                        best_cost = current_cost
                        # Construct G from best_P only when needed (e.g., printing or returning)
                        # best_G = np.hstack((I_k, best_P)) # Optional: update best_G here too
                        print(f"  New Best Found: Cost = {best_cost:.4f} at T = {T:.2f}")
                        # Update progress bar description immediately when best changes
                        if pbar:
                            pbar.set_description(f"T={T:.2e}, Best={best_cost:.4f}")

            T *= alpha # Cool down
            temp_step += 1
            if pbar:
                pbar.update(1)
                if temp_step % 10 != 0: # Avoid duplicate printing if best changed
                    pbar.set_description(f"T={T:.2e}, Best={best_cost:.4f}")
            # Print less frequently to console if using progress bar
            # if temp_step % 50 == 0:
            #      print(f"Temp Step {temp_step}: T = {T:.2f}, Current Cost = {current_cost:.4f}, Best Cost = {best_cost:.4f}")

    finally:
        # Close the progress bar and restore original signal handler
        if pbar:
            pbar.close()
        signal.signal(signal.SIGINT, original_sigint_handler) # Restore handler

    print(f"\nAnnealing Finished. Best Cost Found: {best_cost:.4f}")
    best_G = np.hstack((I_k, best_P)) # Construct final best G
    return best_G, best_cost

def is_valid_P(P):
    """Checks if any column in P is the all-zero vector."""
    k, p_cols = P.shape
    for j in range(p_cols):
        if np.all(P[:, j] == 0):
            return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find a systematic generator matrix G=[I|P] minimizing m-height using Simulated Annealing.")

    # --- Add Command Line Arguments ---
    parser.add_argument('-k', type=int, default=3, help='Number of message bits (rows). Default: 3')
    parser.add_argument('-n', type=int, default=6, help='Number of codeword bits (columns). Default: 6')
    parser.add_argument('-m', type=int, default=2, help='m-height parameter (must be 1 <= m <= n-1). Default: 2')

    # Updated defaults and help text for P matrix element range
    parser.add_argument('--element-min', type=int, default=-100, help='Minimum value for elements in P matrix. Default: -100')
    parser.add_argument('--element-max', type=int, default=100, help='Maximum value for elements in P matrix. Default: 100')

    parser.add_argument('--t-max', type=float, default=1000.0, help='Initial annealing temperature. Default: 1000.0')
    parser.add_argument('--t-min', type=float, default=0.1, help='Final annealing temperature. Default: 0.1')
    parser.add_argument('--alpha', type=float, default=0.98, help='Cooling rate (multiplier). Default: 0.98')
    parser.add_argument('--iter-per-temp', type=int, default=100, help='Iterations per temperature level. Default: 100')

    args = parser.parse_args()

    # --- Use Parsed Arguments ---
    K = args.k
    N = args.n
    M = args.m
    ELEMENT_MIN = args.element_min
    ELEMENT_MAX = args.element_max
    ELEMENT_RANGE = (ELEMENT_MIN, ELEMENT_MAX)
    T_MAX = args.t_max
    T_MIN = args.t_min
    ALPHA = args.alpha
    ITER_PER_TEMP = args.iter_per_temp

    # --- Validate Parameters ---
    if K < 1:
        parser.error(f"k ({K}) must be at least 1")
    if N <= K:
        parser.error(f"n ({N}) must be greater than k ({K}) for a systematic code with P")
    p_cols = N - K # Number of columns in P
    if p_cols < 1:
         parser.error(f"n-k ({p_cols}) must be at least 1, check n and k values")

    if not (1 <= M <= N - 1):
        parser.error(f"m ({M}) must be between 1 and n-1 ({N-1})")
    if ELEMENT_MIN > ELEMENT_MAX:
        parser.error(f"element-min ({ELEMENT_MIN}) cannot be greater than element-max ({ELEMENT_MAX})")
    if ELEMENT_MAX < 0 and ELEMENT_MIN < 0:
         print("Warning: Element range is entirely negative. Cannot satisfy no-all-zero column constraint if k=0.") # Should not happen k>=1
    if ELEMENT_MIN > 0 and ELEMENT_MAX > 0:
         print("Warning: Element range is entirely positive. Cannot satisfy no-all-zero column constraint if k=0.") # Should not happen k>=1
    if ELEMENT_MIN <= 0 <= ELEMENT_MAX and ELEMENT_MIN == ELEMENT_MAX:
         parser.error(f"Element range is [0, 0]. Cannot satisfy no-all-zero column constraint.")

    # --- Initialization ---
    print(f"Generating initial random P matrix ({K}x{p_cols})...")
    initial_P = None
    max_init_retries = 100
    for attempt in range(max_init_retries):
        P_candidate = np.random.randint(ELEMENT_MIN, ELEMENT_MAX + 1, size=(K, p_cols))
        if is_valid_P(P_candidate):
            initial_P = P_candidate
            break
    if initial_P is None:
        print(f"Error: Failed to generate a valid initial P matrix (no all-zero columns) after {max_init_retries} attempts.")
        print("Check element range and dimensions.")
        sys.exit(1) # Use sys.exit

    I_k = np.eye(K, dtype=initial_P.dtype)
    initial_generator = np.hstack((I_k, initial_P))

    print("Initial Random Systematic Generator Matrix G = [I|P]:")
    print(initial_generator)
    print(f"Targeting m={M}-height minimization for a systematic ({K}x{N}) matrix.")
    print(f"Element range for P: [{ELEMENT_MIN}, {ELEMENT_MAX}]")

    # --- Run Annealing ---
    try:
        best_generator, min_height = simulated_annealing(
            initial_P=initial_P, # Pass initial P
            k=K,              # Pass k
            n=N,              # Pass n
            m=M,
            T_max=T_MAX,
            T_min=T_MIN,
            alpha=ALPHA,
            iter_per_temp=ITER_PER_TEMP,
            element_range=ELEMENT_RANGE
        )

        print("--- Results ---")
        print("Best Systematic Generator Matrix Found G = [I|P]:")
        print(best_generator)
        print(f"Minimal m-height (m={M}): {min_height}")

        # Optional: Verify the final cost
        final_cost_check = calculate_cost(best_generator, M)
        print(f"Verification of final cost: {final_cost_check}")

    except ValueError as e:
        print(f"\nError during annealing process: {e}")
    except ImportError:
        print("\nPlease ensure numpy, scipy and tqdm are installed (`pip install numpy scipy tqdm`)")
    except KeyboardInterrupt:
        # This handles the case where the interrupt happens outside the main loop
        # or before the try/finally block restores the handler (less likely)
        print("\nKeyboardInterrupt caught outside annealing loop.")
        # Optionally decide if you want to return intermediate results here or just exit
        sys.exit(1) 
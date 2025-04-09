import numpy as np
import argparse
import sys
import random
import time
import os
import concurrent.futures
import multiprocessing
import math
import copy
from tqdm import tqdm

try:
    # Assuming compute_m_height is in verifier.py at the same level
    from verifier import compute_m_height
except ImportError:
    print("Error: Could not import 'compute_m_height' from 'verifier.py'.")
    print("Please ensure 'verifier.py' exists in the same directory or in the Python path.")
    sys.exit(1)

# --- Helper Functions ---

def load_p_matrix(filepath):
    """
    Loads n, k, m, and the P matrix from a text file.
    Expected format:
    n k m
    p11 p12 ... p1(n-k)
    p21 p22 ... p2(n-k)
    ...
    pk1 pk2 ... pk(n-k)
    """
    try:
        with open(filepath, 'r') as f:
            # Read the first line for n, k, m
            first_line = f.readline().strip()
            try:
                n, k, m = map(int, first_line.split())
            except ValueError:
                raise ValueError("First line must contain three integers: n, k, m, separated by spaces.")

            # Basic validation of n, k, m
            if not (k >= 1):
                raise ValueError("k must be >= 1.")
            if not (n > k):
                raise ValueError("n must be > k.")
            p_cols = n - k
            if not (1 <= m <= n - 1):
                 raise ValueError(f"m ({m}) must be between 1 and n-1 ({n-1}).")

            # Read the rest of the lines for the P matrix
            # Use np.loadtxt on the rest of the file stream
            P = np.loadtxt(f, dtype=np.int64)

            # Validate P matrix dimensions
            if P.ndim == 0:
                 raise ValueError("Matrix data appears empty or scalar after the first line.")
            elif P.ndim == 1:
                # If P has only one row (k=1), loadtxt might return a 1D array.
                # Or if p_cols is 1, it might also be 1D.
                if k == 1 and P.shape[0] == p_cols:
                     P = P.reshape(1, p_cols)
                elif p_cols == 1 and P.shape[0] == k:
                     P = P.reshape(k, 1)
                else:
                    # Ambiguous or incorrect shape
                    raise ValueError(f"Loaded matrix data has unexpected 1D shape {P.shape}. Expected ({k}, {p_cols}).")
            elif P.shape != (k, p_cols):
                raise ValueError(f"Loaded P matrix shape {P.shape} does not match dimensions from first line ({k}x{p_cols}).")

        print(f"Loaded n={n}, k={k}, m={m} and seed P matrix from '{filepath}' with shape: {P.shape}")
        return n, k, m, P

    except FileNotFoundError:
        print(f"Error: Seed file not found at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Could not load data from '{filepath}': {e}")
        sys.exit(1)

def save_p_matrix(matrix, filepath):
    """Saves the P matrix to a text file."""
    try:
        np.savetxt(filepath, matrix, fmt='%d')
        print(f"Best P matrix saved to '{filepath}'")
    except Exception as e:
        print(f"Error: Could not save P matrix to '{filepath}': {e}")

def is_valid_P(P):
    """Checks if any column in P is the all-zero vector."""
    if P.shape[1] == 0: # Handle case n=k, P is empty
        return True
    k, p_cols = P.shape
    for j in range(p_cols):
        if np.all(P[:, j] == 0):
            return False
    return True

def calculate_cost(G, m):
    """Calculates the m-height, handling potential errors or infinite results."""
    try:
        height = compute_m_height(G, m)
        return height if np.isfinite(height) else float('inf')
    except Exception as e:
        # print(f"Warning: compute_m_height failed. Error: {e}") # Optional debug info
        return float('inf')

def generate_neighbor(current_P, element_min, element_max):
    """
    Generates a neighboring P matrix.
    Strategy: Randomly select an element in P and change it to a new random value within the range.
    Ensures the modified column is not the all-zero vector.
    """
    k, p_cols = current_P.shape
    if p_cols == 0:
        return current_P.copy() # Cannot generate neighbor if P is empty

    neighbor_P = current_P.copy()
    max_attempts = k * p_cols * 5 # Set an attempt limit to avoid infinite loops

    for _ in range(max_attempts):
        row_idx = random.randrange(k)
        col_idx = random.randrange(p_cols)
        original_value = neighbor_P[row_idx, col_idx]

        # Try different new values
        for _ in range(10): # Try a few times to get a different value
            new_value = random.randint(element_min, element_max)
            if new_value != original_value:
                break
        else:
            # May not be possible to change if the range is very small (e.g., only one value)
            new_value = random.randint(element_min, element_max)

        neighbor_P[row_idx, col_idx] = new_value

        # Check if the modified column is all zeros
        if np.all(neighbor_P[:, col_idx] == 0):
            # If all zeros, revert the change and try a different position
            neighbor_P[row_idx, col_idx] = original_value
            continue # Try modifying another element
        else:
            # Found a valid neighbor
            return neighbor_P

    # If no valid neighbor found after many attempts (very unlikely unless range is very restricted)
    print("Warning: Could not generate a valid neighbor P matrix (avoiding all-zero column). Returning current P.")
    return current_P.copy()

def evaluate_candidate(args):
    """Evaluates the cost of a single candidate P matrix (runs in a separate process)."""
    P_candidate_list, k, m, I_k_list = args
    try:
        P_candidate = np.array(P_candidate_list, dtype=np.int64)
        I_k = np.array(I_k_list, dtype=np.int64).reshape(k,k)
        G_candidate = np.hstack((I_k, P_candidate))
        cost = calculate_cost(G_candidate, m)
        return P_candidate.tolist(), cost
    except Exception as e:
        # print(f"Error evaluating candidate matrix: {e}") # Optional debug
        return P_candidate_list, float('inf')


# --- Simulated Annealing Core ---

def simulated_annealing(initial_P, m, element_min, element_max,
                         initial_temp, cooling_rate, steps_per_temp,
                         num_workers, output_file):
    """Performs the simulated annealing algorithm."""
    k, p_cols = initial_P.shape
    n = k + p_cols
    I_k = np.eye(k, dtype=np.int64)
    I_k_list = I_k.tolist() # For pickling

    current_P = initial_P.copy()
    current_G = np.hstack((I_k, current_P))
    current_cost = 114514  # Set to infinite to ignore seed's initial cost

    best_P = current_P.copy()
    best_cost = current_cost

    second_best_P = current_P.copy()
    second_best_cost = current_cost

    print(f"Initial seed cost (m={m}): {current_cost if current_cost != float('inf') else 'inf'}")
    if best_cost == float('inf'):
        print("Warning: Initial seed matrix has infinite cost. Annealing might not be effective.")


    temperature = initial_temp
    # Estimate total steps for progress bar based on when temperature drops below 0.01
    total_steps = int(steps_per_temp * (math.log(0.01 / initial_temp) / math.log(cooling_rate) if cooling_rate < 1 and initial_temp > 0.01 else 10))

    progress_bar = tqdm(total=total_steps, unit="steps", desc="Annealing", ncols=100)
    steps_done = 0

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            while temperature > 0.01: # Stopping condition
                futures = []
                # Generate and evaluate neighbors in parallel at each temperature
                tasks_at_this_temp = []
                for _ in range(steps_per_temp):
                     # Generate a neighbor
                    neighbor_P = generate_neighbor(current_P, element_min, element_max)
                    if not is_valid_P(neighbor_P): # Double-check validity just in case
                        continue
                    tasks_at_this_temp.append((neighbor_P.tolist(), k, m, I_k_list))

                # Submit evaluation tasks
                # Using executor.map for potentially better memory usage with many tasks
                results = executor.map(evaluate_candidate, tasks_at_this_temp)

                # Process results and apply SA logic
                for neighbor_P_list, neighbor_cost in results:
                    steps_done += 1
                    progress_bar.update(1)

                    neighbor_P = np.array(neighbor_P_list, dtype=np.int64)

                    if neighbor_cost < current_cost:
                        # Accept better solution
                        current_P = neighbor_P
                        current_cost = neighbor_cost
                        # Update the global best solution
                        if neighbor_cost < best_cost:
                            # Update second best to the previous best
                            second_best_P = best_P.copy()
                            second_best_cost = best_cost
                            best_cost = neighbor_cost
                            best_P = neighbor_P.copy()
                            progress_bar.set_description(f"Annealing (T={temperature:.2f}, Best={best_cost:.4f})")
                        elif neighbor_cost < second_best_cost:
                            # Update second best to the current solution
                            second_best_P = neighbor_P.copy()
                            second_best_cost = neighbor_cost

                    else:
                        # Accept worse solution with a certain probability
                        delta_cost = neighbor_cost - current_cost
                        acceptance_prob = math.exp(-delta_cost / temperature)
                        if random.random() < acceptance_prob:
                            current_P = neighbor_P
                            current_cost = neighbor_cost

                # Cool down
                temperature *= cooling_rate
                progress_bar.set_description(f"Annealing (T={temperature:.2f}, Best={best_cost:.4f})")


    except KeyboardInterrupt:
        print("Annealing process interrupted by user.")
        print("Returning the second best solution found so far.")
        print(second_best_P)
        print(f"Best cost found before interruption: {best_cost if best_cost != float('inf') else 'inf'}")
        return second_best_P, second_best_cost
    finally:
        progress_bar.close()

    print("--- Simulated Annealing Finished ---")
    print(f"Best cost found (m={m}): {best_cost if best_cost != float('inf') else 'inf'}")
    print(f"Second best cost found (m={m}): {second_best_cost if second_best_cost != float('inf') else 'inf'}")

    if second_best_cost < float('inf'):
         # Save the second best result
        if output_file:
            save_p_matrix(second_best_P, output_file)
        else:
            print("Second best P matrix:")
            with np.printoptions(precision=3, suppress=True):
                print(second_best_P)
            print("Corresponding G = [I|P] matrix:")
            second_best_G = np.hstack((I_k, second_best_P))
            with np.printoptions(precision=3, suppress=True):
                 print(second_best_G)

    return second_best_P, second_best_cost


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Optimize the m-height of a systematic generator matrix G=[I|P] using simulated annealing, starting from a seed file containing n, k, m, and the P matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--seed-file', type=str, default="test.txt",
                        help='Path to the text file containing n, k, m on the first line, followed by the initial seed P matrix.')
    parser.add_argument('--initial-temp', type=float, default=10,
                        help='Initial temperature for simulated annealing.')
    parser.add_argument('--cooling-rate', type=float, default=0.95,
                        help='Cooling rate (0 < rate < 1).')
    parser.add_argument('--steps-per-temp', type=int, default=5,
                        help='Number of steps (neighbor evaluations) to perform at each temperature.')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of worker processes for parallel neighbor evaluation. Defaults to number of CPU cores.')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Path to save the best found P matrix. If not provided, prints to console.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed) # Ensure numpy's randomness is also seeded

    # Load n, k, m, and initial P matrix from the file
    n, k, m, initial_P = load_p_matrix(args.seed_file)
    p_cols = n - k # Calculate p_cols from loaded n and k

    # Calculate element_min and element_max based on the seed matrix
    element_min = int(np.min(initial_P))
    element_max = int(np.max(initial_P))

    # --- Validate Parameters (some validations are now done in load_p_matrix) ---
    # k, n, m validation is implicitly done by load_p_matrix success
    if element_min > element_max: parser.error("element-min cannot be greater than element-max")
    if args.initial_temp <= 0: parser.error("Initial temperature must be positive")
    if not (0 < args.cooling_rate < 1): parser.error("Cooling rate must be between 0 and 1")
    if args.steps_per_temp < 1: parser.error("Steps per temperature must be at least 1")

    num_workers = args.num_workers or os.cpu_count()
    if num_workers < 1: parser.error("Number of worker processes must be at least 1")

     # Specific check for element range and zero columns
    if element_min == 0 and element_max == 0 and p_cols > 0:
         parser.error("Element range is [0, 0] and P matrix exists (n > k). Cannot satisfy no-all-zero column constraint during neighbor search.")

    # Check initial P validity loaded from file
    if not is_valid_P(initial_P):
        print("Warning: The loaded seed P matrix contains at least one all-zero column.")
        # Decide if this should be an error or just a warning. For SA, it might be okay to start invalid.
        # parser.error("Loaded seed P matrix has an all-zero column, which is invalid.")


    print(f"--- Starting Simulated Annealing ---")
    print(f"Parameters from file: k={k}, n={n}, m={m}")
    print(f"Search Element Range: [{element_min}, {element_max}]")
    print(f"Annealing Params -> Initial Temp: {args.initial_temp}, Cooling Rate: {args.cooling_rate}, Steps/Temp: {args.steps_per_temp}")
    print(f"Using {num_workers} worker processes")
    if args.seed is not None: print(f"Random Seed: {args.seed}")


    start_time = time.time()

    # Execute simulated annealing - pass m read from the file
    best_P, best_cost = simulated_annealing(
        initial_P=initial_P,
        m=m, # Use m from the file
        element_min=element_min,
        element_max=element_max,
        initial_temp=args.initial_temp,
        cooling_rate=args.cooling_rate,
        steps_per_temp=args.steps_per_temp,
        num_workers=num_workers,
        output_file=args.output_file
    )

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows support when compiled
    main() 
import numpy as np
import argparse
import sys
import random
import time
import os
import concurrent.futures
import threading
import multiprocessing
from tqdm import tqdm

try:
    # Assuming compute_m_height is in verifier.py at the same level
    from verifier import compute_m_height
except ImportError:
    print("Error: Could not import 'compute_m_height' from 'verifier.py'.")
    print("Ensure 'verifier.py' exists and is in the same directory or Python path.")
    sys.exit(1)

# We'll use a Manager for sharing variables across processes
manager = multiprocessing.Manager()
# Shared variables for tracking best result
shared_dict = manager.dict()
shared_dict['best_P_global'] = None
shared_dict['min_height_global'] = float('inf')
shared_dict['checked_count'] = 0
# Lock for multiprocessing
shared_lock = manager.Lock()
# Progress bar - will be initialized in main process
progress_bar = None

def is_valid_P(P):
    """Checks if any column in P is the all-zero vector."""
    if P.shape[1] == 0: # Handle case n=k, P is empty
        return True
    k, p_cols = P.shape
    for j in range(p_cols):
        if np.all(P[:, j] == 0):
            return False
    return True

def generate_random_P(k, p_cols, element_min, element_max):
    """Generates a random P matrix with integer elements in the specified range."""
    # Ensure the range is valid
    if element_min > element_max:
        raise ValueError("element_min cannot be greater than element_max")
    return np.random.randint(element_min, element_max + 1, size=(k, p_cols), dtype=np.int64)

def calculate_cost(G, m):
    """Calculates the m-height, handling potential errors or infinite results."""
    with shared_lock:
        shared_dict['checked_count'] += 1
        # Update progress bar if it exists (in the main process)
        if 'main_pid' in shared_dict and shared_dict['main_pid'] == os.getpid() and progress_bar is not None:
            progress_bar.update(1)
    
    try:
        height = compute_m_height(G, m)
        # Ensure infinity is returned consistently if calculation fails or result is inf
        return height if np.isfinite(height) else float('inf')
    except Exception as e:
        # print(f"Warning: compute_m_height failed. Error: {e}") # Optional debug
        return float('inf')

def update_progress_bar_description():
    """Updates the progress bar description with current stats."""
    if progress_bar is None or 'main_pid' not in shared_dict or shared_dict['main_pid'] != os.getpid():
        return
        
    elapsed_time = time.time() - shared_dict['start_time']
    rate = shared_dict['checked_count'] / elapsed_time if elapsed_time > 0 else 0
    current_best = f"{shared_dict['min_height_global']:.4f}" if shared_dict['min_height_global'] != float('inf') else "inf"
    
    progress_bar.set_description(
        f"Rate: {rate:.2f}/s | Best: {current_best}"
    )

def process_sample(args):
    """Process a single random matrix sample. This function will run in a separate process."""
    i, K, p_cols, ELEMENT_MIN, ELEMENT_MAX, M, I_k_list = args
    
    # Reconstruct the identity matrix (numpy arrays aren't directly picklable)
    I_k = np.array(I_k_list, dtype=np.int64).reshape(K, K)
    
    # Each process should have its own random state
    local_random = np.random.RandomState(os.getpid() + i)
    
    # Generate a random P matrix using local random state
    P = local_random.randint(ELEMENT_MIN, ELEMENT_MAX + 1, size=(K, p_cols), dtype=np.int64)

    # Check if P is valid (no all-zero columns)
    if not is_valid_P(P):
        return None  # Skip invalid P
    
    # Construct the full generator matrix G = [I|P]
    G = np.hstack((I_k, P))

    # Print information before calculating cost
    print(f"Checking sample {i}...")

    # Calculate m-height cost
    cost = calculate_cost(G, M)
    
    # Update the best result found so far if needed
    with shared_lock:
        if cost < shared_dict['min_height_global']:
            shared_dict['min_height_global'] = cost
            # We need to store P as a list to make it picklable for the manager
            shared_dict['best_P_global'] = P.tolist()
            
            # Only update progress bar from main process
            if 'main_pid' in shared_dict and shared_dict['main_pid'] == os.getpid():
                update_progress_bar_description()
    
    return cost

def main():
    global progress_bar

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Find a systematic generator matrix G=[I|P] with low m-height using random search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument('-k', type=int, required=True, help='Number of message bits (rows).')
    parser.add_argument('-n', type=int, required=True, help='Number of codeword bits (columns).')
    parser.add_argument('-m', type=int, required=True, help='m-height parameter (must be 1 <= m <= n-1).')
    parser.add_argument('--element-min', type=int, default=-1, help='Minimum value for elements in P matrix.')
    parser.add_argument('--element-max', type=int, default=1, help='Maximum value for elements in P matrix.')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of random P matrices to check.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    parser.add_argument('--num-processes', type=int, default=None, 
                        help='Number of processes to use. Default is number of CPU cores.')
    parser.add_argument('--no-progress-bar', action='store_true', 
                        help='Disable the progress bar.')


    args = parser.parse_args()

    K = args.k
    N = args.n
    M = args.m
    ELEMENT_MIN = args.element_min
    ELEMENT_MAX = args.element_max
    NUM_SAMPLES = args.num_samples
    RANDOM_SEED = args.seed
    NUM_PROCESSES = args.num_processes or os.cpu_count()
    SHOW_PROGRESS_BAR = not args.no_progress_bar

    # Set random seed if provided
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED) # Also seed Python's random for other potential uses

    # --- Validate Parameters ---
    if K < 1: parser.error("k must be at least 1")
    if N <= K: parser.error("n must be greater than k for P matrix to exist")
    p_cols = N - K
    if not (1 <= M <= N - 1): parser.error(f"m ({M}) must be between 1 and n-1 ({N-1})")
    if ELEMENT_MIN > ELEMENT_MAX: parser.error("element-min cannot be greater than element-max")
    if NUM_SAMPLES < 1: parser.error("Number of samples must be at least 1")
    if NUM_PROCESSES < 1: parser.error("Number of processes must be at least 1")
    # Specific check: If range is only 0, and P exists (p_cols > 0), invalid P is guaranteed.
    if ELEMENT_MIN == 0 and ELEMENT_MAX == 0 and p_cols > 0:
         parser.error("Element range is [0, 0] and P matrix exists (n > k). Cannot satisfy no-all-zero column constraint.")

    # --- Start Search ---
    print(f"Starting Random Search for ({K}x{N}) matrix, m={M}-height.")
    print(f"Checking {NUM_SAMPLES} random P matrices using {NUM_PROCESSES} processes.")
    print(f"Element range for P: [{ELEMENT_MIN}, {ELEMENT_MAX}]")
    if RANDOM_SEED is not None:
        print(f"Using random seed: {RANDOM_SEED}")
    
    I_k = np.eye(K, dtype=np.int64) # Identity matrix part
    
    # Store I_k as a list for pickling (multiprocessing needs serializable data)
    I_k_list = I_k.tolist()
    
    # Set up shared variables
    shared_dict['start_time'] = time.time()
    shared_dict['main_pid'] = os.getpid()

    try:
        # Prepare arguments for each sample
        sample_args = [(i, K, p_cols, ELEMENT_MIN, ELEMENT_MAX, M, I_k_list) 
                       for i in range(NUM_SAMPLES)]
        
        # Initialize progress bar if enabled
        if SHOW_PROGRESS_BAR:
            progress_bar = tqdm(total=NUM_SAMPLES, unit="matrices", 
                               desc="Rate: 0.00/s | Best: inf", ncols=100)
        
        # Use ProcessPoolExecutor for multiprocessing
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            # Submit all tasks and get futures
            futures = [executor.submit(process_sample, arg) for arg in sample_args]
            
            # Wait for all tasks to complete or for a KeyboardInterrupt
            try:
                # Process results as they complete to update progress bar
                for future in concurrent.futures.as_completed(futures):
                    # Just ensuring the future completes
                    result = future.result()
                    # Update progress description periodically
                    if SHOW_PROGRESS_BAR and shared_dict['checked_count'] % max(1, NUM_SAMPLES//100) == 0:
                        update_progress_bar_description()
                        
            except KeyboardInterrupt:
                print("\nSearch interrupted by user. Shutting down processes...")
                executor.shutdown(wait=False)
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")

    finally:
        # Clean up the progress bar
        if progress_bar is not None:
            progress_bar.close()
            
        end_time = time.time()
        print(f"Search finished in {end_time - shared_dict['start_time']:.2f} seconds.")
        print(f"Total matrices checked: {shared_dict['checked_count']}")

        # --- Output Result ---
        if shared_dict['best_P_global'] is not None:
            print("--- Best Systematic Generator Matrix Found G = [I|P] ---")
            # Reconstruct the best P matrix
            best_P = np.array(shared_dict['best_P_global'], dtype=np.int64)
            # Construct the best generator matrix G = [I|P]
            best_G = np.hstack((I_k, best_P))
            # Print the resulting matrix with controlled precision
            with np.printoptions(precision=3, suppress=True):
                 print(best_G)
            # Print the minimal m-height found
            print(f"Minimal m-height (m={M}): {shared_dict['min_height_global']:.6g}")

            # --- Optional Verification ---
            # print("Verifying final cost...")
            final_cost_check = calculate_cost(best_G, M) # Rerun calculation
            # print(f"Verification cost calculation: {final_cost_check:.6g}")
            if not np.isclose(final_cost_check, shared_dict['min_height_global'], equal_nan=True):
                 print(f"Warning: Final cost verification ({final_cost_check:.6g}) differs from recorded best cost ({shared_dict['min_height_global']:.6g}).")
            # else:
            #      print("Verification successful.")
        else:
            print("No valid systematic generator matrix found with finite m-height.")
            if shared_dict['checked_count'] < NUM_SAMPLES:
                 print("Note: Not all samples were fully checked (some might have been invalid or interrupted).")

if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows support when compiled
    main() 
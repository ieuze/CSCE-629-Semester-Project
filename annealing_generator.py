import numpy as np
import math
import random
from verifier import compute_m_height
import argparse
import sys # For exit
import signal # For Ctrl+C handling
from tqdm import tqdm # For progress bar
import multiprocessing
import time # For adding slight delay in worker start
import os # Needed for os.getpid() in worker seeding
import itertools
from multiprocessing import Value # For shared counter

# Global flag for Ctrl+C - Primarily for the main process now
user_interrupted = False

# --- Worker Initializer for Shared Counter ---
# Global variable placeholder in the worker process
worker_cost_counter = None

def init_worker_counter(counter):
    """Initializer function for worker processes to inherit the counter."""
    global worker_cost_counter
    worker_cost_counter = counter

def signal_handler(sig, frame):
    """Handle Ctrl+C: set flag and print message."""
    global user_interrupted
    if user_interrupted: # Second Ctrl+C
        print('\nCtrl+C detected again. Exiting forcefully.')
        sys.exit(1)
    print('\nCtrl+C detected. Asking workers to finish current step and stopping...')
    user_interrupted = True
    # Signal handling within workers is more complex; often rely on pool termination

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
        # print(f"Warning: compute_m_height failed for G={G} with m={m}. Error: {e}") # Less verbose in parallel
        return float('inf')
    except Exception as e:
        # Catch other potential exceptions from linprog
        # print(f"Unexpected error during cost calculation: {e}") # Less verbose in parallel
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

        # Try to find a *different* value first
        new_val = current_val
        attempt_val = 0
        while new_val == current_val and attempt_val < max_retries_value:
             new_val = random.choice(possible_values)
             attempt_val += 1

        if new_val == current_val: # Still couldn't find a different value
             # If only one value possible, this is expected. If more, failed to find different.
             continue # Try changing a different element

        # Temporarily make the change
        original_value_in_col = P_new[row_idx, col_idx]
        P_new[row_idx, col_idx] = new_val

        # Check if the column is now all zeros
        if not np.all(P_new[:, col_idx] == 0):
            # found_valid_new_val = True
            return P_new # Found a valid neighbor
        else:
            # Revert the change for this specific value try
            P_new[row_idx, col_idx] = original_value_in_col
            # Continue loop to try a different new_val or (row, col)

        # If no valid new_val found for this (row, col), the outer loop continues
        # to try modifying a different element.

    # If max_retries_neighbor reached without finding a valid move
    # This should be rare with a large element_range
    # print("Warning: get_neighbor failed to find a valid move after multiple retries. Returning original P.") # Less verbose
    return P # Return original if stuck

def simulated_annealing_core(initial_P, k, n, m, T_max, T_min, alpha, iter_per_temp, element_range, worker_id=0, seed=None):
    """Core SA logic for a single run."""
    # Access the counter via the global variable initialized in the worker
    global worker_cost_counter

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed) # Also seed numpy for initial P generation if done here

    if not (1 <= m <= n - 1):
        # This validation should ideally happen before starting workers
        return None, float('inf') # Indicate error

    I_k = np.eye(k, dtype=initial_P.dtype) # Ensure I has same dtype as P

    current_P = initial_P
    current_G = np.hstack((I_k, current_P))
    current_cost = calculate_cost(current_G, m)

    # >>> Increment counter for initial calculation <<<
    if worker_cost_counter is not None:
        with worker_cost_counter.get_lock(): # Use lock for safety
            worker_cost_counter.value += 1

    best_P = current_P
    best_cost = current_cost

    T = T_max

    # print(f"Starting Annealing (Systematic): Initial Cost = {current_cost:.4f}, T_max = {T_max}, T_min = {T_min}, alpha = {alpha}")
    # print("Press Ctrl+C to stop early and get the best result found so far.")

    # Estimate total steps for tqdm progress bar
    # if T_max > T_min and 0 < alpha < 1:
    #     total_steps_estimate = math.ceil(math.log(T_min / T_max) / math.log(alpha))
    # else:
    #     total_steps_estimate = 0

    # Initialize tqdm progress bar
    # pbar = tqdm(total=total_steps_estimate, desc=f"T={T:.2e}, Best={best_cost:.4f}", unit="steps", disable=(total_steps_estimate==0))

    temp_step = 0
    # try: # No longer needed as signal handling is external
    while T > T_min:
        # Check for user interrupt at the beginning of each temperature step
        # Complex to implement reliably within worker, rely on pool termination
        # if user_interrupted:
            # print(f"\nWorker {worker_id} detecting interruption.")
            # break

        for i in range(iter_per_temp):
            neighbor_P = get_neighbor(current_P, element_range)
            neighbor_G = np.hstack((I_k, neighbor_P))
            neighbor_cost = calculate_cost(neighbor_G, m)

            # >>> Increment counter for neighbor calculation <<<
            if worker_cost_counter is not None:
                with worker_cost_counter.get_lock():
                    worker_cost_counter.value += 1

            delta_E = neighbor_cost - current_cost

            # --- Acceptance Logic (same as before, with overflow check) ---
            accept = False
            if neighbor_cost == float('inf') and current_cost == float('inf'):
                accept = False # Cannot improve from inf to inf
            elif neighbor_cost == float('inf'):
                accept = False # Don't accept infinite cost if current is finite
            elif current_cost == float('inf'):
                accept = True # Always accept if moving away from infinite cost
            elif delta_E < 0:
                accept = True
            else:
                 # Avoid math.exp overflow if delta_E / T is huge
                if delta_E == 0: # Avoid 0/T issues if T is very small
                    acceptance_prob = 1.0 # Or 0.0 depending on philosophy, 1.0 allows movement
                elif T <= 0: # Avoid division by zero or negative T
                     acceptance_prob = 0.0
                else:
                    exponent = -delta_E / T
                    if exponent > 700: # math.exp(709) is roughly max float
                        acceptance_prob = 0.0
                    else:
                        acceptance_prob = math.exp(exponent)

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
                    # No printing inside core loop for performance
                    # print(f"  Worker {worker_id}: New Best Found: Cost = {best_cost:.4f} at T = {T:.2f}")
                    # Update progress bar description immediately when best changes
                    # if pbar:
                    #     pbar.set_description(f"T={T:.2e}, Best={best_cost:.4f}")

        T *= alpha # Cool down
        temp_step += 1
        # if pbar:
        #     pbar.update(1)
        #     if temp_step % 10 != 0: # Avoid duplicate printing if best changed
        #         pbar.set_description(f"T={T:.2e}, Best={best_cost:.4f}")
        # Print less frequently to console if using progress bar
        # if temp_step % 50 == 0:
        #      print(f"Temp Step {temp_step}: T = {T:.2f}, Current Cost = {current_cost:.4f}, Best Cost = {best_cost:.4f}")

    # finally: # No longer needed
        # Close the progress bar and restore original signal handler
        # if pbar:
            # pbar.close()
        # signal.signal(signal.SIGINT, original_sigint_handler) # Restore handler

    # print(f"\nWorker {worker_id} Annealing Finished. Best Cost Found: {best_cost:.4f}")
    best_G = np.hstack((I_k, best_P)) # Construct final best G
    return best_G, best_cost

def is_valid_P(P):
    """Checks if any column in P is the all-zero vector."""
    k, p_cols = P.shape
    for j in range(p_cols):
        if np.all(P[:, j] == 0):
            return False
    return True

def run_single_annealing_instance(args_tuple):
    """
    Worker function to run one independent simulated annealing process.
    Takes all arguments packed in a tuple. Handles initialization and seeding.
    """
    # Unpack arguments
    # args_tuple contains (worker_id, k, n, m, element_range, T_max, T_min, alpha, iter_per_temp, seed)
    worker_id, k, n, m, element_range, T_max, T_min, alpha, iter_per_temp, seed = args_tuple
    print(f"[Worker {worker_id}] Starting.")

    # Ensure each worker has a different random seed
    # Use a more robust seeding mechanism if strict reproducibility across runs is needed
    if seed is None:
        # Generate a seed if none provided (e.g., based on time, pid, and worker_id)
        base_seed = int(time.time() * 1000) + random.randint(0, 10000) + worker_id + os.getpid()
        seed = base_seed % (2**32) # Ensure seed is within the valid range [0, 2**32 - 1]
    random.seed(seed)
    np.random.seed(seed)
    # print(f"Worker {worker_id} (PID {os.getpid()}) initialized with seed {seed}") # Debug seeding
    print(f"[Worker {worker_id}] Seeded.")

    # Generate initial P matrix *within* the worker to ensure independence
    p_cols = n - k
    initial_P = None
    max_init_retries = 100
    element_min, element_max = element_range
    print(f"[Worker {worker_id}] Generating initial P...")
    for attempt in range(max_init_retries):
        # Ensure element_max+1 is correct for randint upper bound (exclusive)
        P_candidate = np.random.randint(element_min, element_max + 1, size=(k, p_cols))
        if is_valid_P(P_candidate):
            initial_P = P_candidate
            break

    if initial_P is None:
        print(f"[Worker {worker_id}] Error: Failed to generate a valid initial P matrix after {max_init_retries} attempts.")
        # Return failure indicator: None for G, inf cost, and worker_id
        print(f"[Worker {worker_id}] Returning failure.")
        return None, float('inf'), worker_id

    print(f"[Worker {worker_id}] Initial P generated.")

    # Run the core annealing logic
    print(f"[Worker {worker_id}] Starting annealing core...")
    best_generator, min_height = simulated_annealing_core(
        initial_P=initial_P,
        k=k,
        n=n,
        m=m,
        T_max=T_max,
        T_min=T_min,
        alpha=alpha,
        iter_per_temp=iter_per_temp,
        element_range=element_range,
        worker_id=worker_id,
        seed=seed
    )
    print(f"[Worker {worker_id}] Annealing core finished. Cost: {min_height:.4f}")

    # Return result along with worker_id for tracking if needed
    print(f"[Worker {worker_id}] Returning success.")
    return best_generator, min_height, worker_id

if __name__ == "__main__":
    # Required for multiprocessing spawn method on some OS (like Windows)
    # Needs to be at the top level of the `if __name__ == "__main__":` block
    multiprocessing.freeze_support() # No-op on Linux/macOS, necessary for Windows executable

    parser = argparse.ArgumentParser(description="Find a systematic generator matrix G=[I|P] minimizing m-height using Parallel Simulated Annealing.")

    # --- Arguments (mostly unchanged) ---
    parser.add_argument('-k', type=int, default=3, help='Number of message bits (rows). Default: 3')
    parser.add_argument('-n', type=int, default=6, help='Number of codeword bits (columns). Default: 6')
    parser.add_argument('-m', type=int, default=2, help='m-height parameter (must be 1 <= m <= n-1). Default: 2')

    # Updated defaults and help text for P matrix element range
    parser.add_argument('--element-min', type=int, default=-100, help='Minimum value for elements in P matrix. Default: -100')
    parser.add_argument('--element-max', type=int, default=100, help='Maximum value for elements in P matrix. Default: 100')

    parser.add_argument('--t-max', type=float, default=100.0, help='Initial annealing temperature. Default: 100.0')
    parser.add_argument('--t-min', type=float, default=65.0, help='Final annealing temperature. Default: 65.0')
    parser.add_argument('--alpha', type=float, default=0.95, help='Cooling rate (multiplier). Default: 0.95')
    parser.add_argument('--iter-per-temp', type=int, default=1, help='Iterations per temperature level. Default: 1')

    # --- New Argument for Parallelism ---
    try:
        default_workers = multiprocessing.cpu_count()
    except NotImplementedError:
        default_workers = 1 # Fallback if cpu_count fails
        print("Warning: Could not determine CPU count, defaulting to 1 worker.")
    parser.add_argument('--workers', type=int, default=default_workers,
                        help=f'Number of parallel annealing runs. Default: {default_workers} (logical CPU cores)')

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
    NUM_WORKERS = args.workers

    # --- Calculate total cost calls for progress bar ---
    total_cost_calls = 0
    if T_MAX > T_MIN and 0 < ALPHA < 1:
        try:
            num_steps = math.ceil(math.log(T_MIN / T_MAX) / math.log(ALPHA))
            total_cost_calls = NUM_WORKERS * num_steps * ITER_PER_TEMP
            print(f"Expecting approx. {num_steps} temp steps per worker.")
            print(f"Total expected cost evaluations across {NUM_WORKERS} workers: {total_cost_calls}")
        except ValueError: # Catch potential math domain errors (e.g., log(negative))
            print("Warning: Could not calculate total steps, progress bar may be inaccurate.")
            total_cost_calls = 1 # Avoid division by zero if tqdm total is 0
    else:
        print("Warning: T_max <= T_min or alpha >= 1 or alpha <= 0. Progress bar total may be inaccurate.")
        total_cost_calls = 1 # Avoid division by zero if tqdm total is 0

    # --- Create Shared Counter ---
    # Using Value from multiprocessing for process-safe shared integer
    cost_call_counter = Value('i', 0)

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
    # Check for invalid range [0, 0] specifically
    if ELEMENT_MIN == 0 and ELEMENT_MAX == 0:
         parser.error(f"Element range is [0, 0]. Cannot satisfy no-all-zero column constraint.")
    # Warnings for potentially problematic ranges (less critical now with is_valid_P)
    # if ELEMENT_MAX < 0 and ELEMENT_MIN < 0:
    #      print("Warning: Element range is entirely negative.")
    # if ELEMENT_MIN > 0 and ELEMENT_MAX > 0:
    #      print("Warning: Element range is entirely positive.")
    if NUM_WORKERS < 1:
        parser.error(f"Number of workers ({NUM_WORKERS}) must be at least 1")

    # --- Initialization (moved inside worker) ---
    # print(f"Generating initial random P matrix ({K}x{p_cols})...") # Done by workers

    # I_k = np.eye(K, dtype=initial_P.dtype) # Dtype determined by worker's P
    # initial_generator = np.hstack((I_k, initial_P)) # Example G no longer relevant here

    # print("Initial Random Systematic Generator Matrix G = [I|P]:") # Not applicable here
    # print(initial_generator) # Not applicable here
    print(f"Starting Parallel Simulated Annealing with {NUM_WORKERS} workers.")
    print(f"Targeting m={M}-height minimization for a systematic ({K}x{N}) matrix.")
    print(f"Element range for P: [{ELEMENT_RANGE[0]}, {ELEMENT_RANGE[1]}]")
    print(f"Params: T_max={T_MAX}, T_min={T_MIN}, alpha={ALPHA}, iter_per_temp={ITER_PER_TEMP}")
    print("Press Ctrl+C to stop early (may take a moment to terminate workers).")

    # Prepare arguments for each worker
    # Each worker gets a unique ID (0 to NUM_WORKERS-1) and None seed (will generate its own)
    # Counter is no longer included here
    worker_args = [
        (i, K, N, M, ELEMENT_RANGE, T_MAX, T_MIN, ALPHA, ITER_PER_TEMP, None)
        for i in range(NUM_WORKERS)
    ]

    # --- Setup Signal Handler for Main Process ---
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    # Set the global flag handler
    signal.signal(signal.SIGINT, signal_handler)

    pool = None # Define pool outside try block for finally clause
    results = []
    try:
        # Initialize the pool with the worker initializer function and the counter
        with multiprocessing.Pool(processes=NUM_WORKERS,
                                  initializer=init_worker_counter,
                                  initargs=(cost_call_counter,)) as pool:
            # Use imap_unordered: processes results as they complete, potentially better memory usage
            # than map, order doesn't matter since we find the min cost later.
            # No wrapper needed now, just pass the worker function and args (without counter)
            result_iterator = pool.imap_unordered(run_single_annealing_instance, worker_args)

            # Process results as they come in using tqdm for overall progress
            print(f"Launching {NUM_WORKERS} annealing runs...")
            # Use total_cost_calls for the progress bar total
            pbar = tqdm(total=total_cost_calls, desc="Cost Evals Progress", unit="evals", smoothing=0.1)
            processed_count = 0

            while processed_count < NUM_WORKERS:
                # Check for interrupt FIRST, before waiting for a result
                if user_interrupted:
                    print(f"\nInterrupt detected by main process. Breaking result loop...")
                    break # Exit the main 'while processed_count < NUM_WORKERS' loop

                try:
                    # Wait for the next result. Use a timeout to update progress bar.
                    result = result_iterator.next(timeout=0.5)

                    # Process result ONLY if not interrupted
                    best_G, cost, worker_id = result
                    if best_G is not None:
                        results.append((best_G, cost))
                    else:
                        print(f"Worker {worker_id} failed to produce a valid result.")
                    processed_count += 1
                    # Update progress bar display without incrementing count (n is updated below)
                    pbar.update(0)
                except StopIteration:
                    print("\nResult iterator exhausted.") # Debug print
                    # This happens when the iterator is exhausted normally
                    break # Exit loop if all results are processed
                except KeyboardInterrupt: # Should be caught by the outer try/except
                    print("\nKeyboardInterrupt caught directly in loop.")
                    user_interrupted = True # Ensure flag is set
                    break
                except multiprocessing.TimeoutError:
                    # Timeout occurred, means no worker finished in the last 0.5s
                    # Update progress bar with the current counter value
                    pbar.n = cost_call_counter.value
                    pbar.refresh()
                    continue # Continue loop to check interrupt/wait again
                except Exception as worker_exc: # Catch potential errors from workers
                    print(f"\nError processing result from a worker: {worker_exc}")
                    # Decide how to handle worker errors, e.g., log and continue or stop
                    processed_count += 1 # Count it as processed (failed)
                    pbar.update(1) # Update progress bar even on error

                # Update progress bar after processing a result or catching timeout
                # Ensure the bar reflects the latest count
                pbar.n = cost_call_counter.value
                pbar.refresh()

            pbar.close() # Close progress bar

            # --- Pool Shutdown Logic ---
            if user_interrupted:
                print("\nInterrupt confirmed post-loop. Terminating running worker processes...")
                # Forcefully terminate workers instead of waiting with join()
                pool.terminate()
                # We still need to join to wait for the termination to complete
                pool.join()
                print("Worker pool terminated.")
            else:
                # This path is taken if the loop finished because processed_count == NUM_WORKERS
                print("\nAll scheduled workers completed or failed.")
                # Pool closes and joins automatically via context manager exit
                # if no interrupt occurred.

    except KeyboardInterrupt:
        # This catches Ctrl+C if it happens outside the pool context or during setup/cleanup
        print("\nKeyboardInterrupt caught in main process.")
        # Pool termination is handled by the context manager ('with Pool(...)') exiting
        # or explicitly if needed before exiting the program
        sys.exit(1) # Exit after interrupt
    except Exception as e:
        print(f"\nAn unexpected error occurred in the main process: {e}")
        import traceback
        traceback.print_exc()
        # Pool termination handled by context manager
        sys.exit(1) # Exit on error
    finally:
        # Restore original signal handler *after* pool is guaranteed to be closed
        signal.signal(signal.SIGINT, original_sigint_handler)
        # Ensure pool is terminated if loop exited prematurely without context manager finishing
        if pool is not None and not user_interrupted: # Avoid terminate if already handled by interrupt logic
             # This check might be redundant with the context manager, but safer
             # pool.terminate()
             # pool.join()
             pass

    # --- Aggregate Results ---
    if not results:
        print("\nNo valid results were obtained from any worker.")
        if user_interrupted:
             print("Annealing process was interrupted.")
        sys.exit(1)

    # Find the best result among all workers
    results.sort(key=lambda x: x[1]) # Sort by cost (min_height)
    overall_best_G, overall_best_cost = results[0]

    print("\n--- Overall Best Result ---")
    print(f"Obtained from {len(results)} successful worker run(s).")
    print("Best Systematic Generator Matrix Found G = [I|P]:")
    # Limit printing precision for readability
    with np.printoptions(precision=3, suppress=True):
         print(overall_best_G)
    # Use .g format for cost to handle potentially very small or large numbers well
    print(f"Minimal m-height (m={M}): {overall_best_cost:.6g}")

    # Optional: Verify the final cost again
    final_cost_check = calculate_cost(overall_best_G, M)
    print(f"Verification of final cost: {final_cost_check:.6g}")

    if final_cost_check != overall_best_cost:
         # Added check for floating point comparison issues or potential bugs
         print(f"Warning: Final cost verification ({final_cost_check:.6g}) differs slightly from recorded best cost ({overall_best_cost:.6g}).")

    if user_interrupted:
        print("\nProcess was interrupted. The reported best result is based on completed runs before interruption.")
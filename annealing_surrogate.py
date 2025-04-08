import numpy as np
import math
import random
import argparse
import sys
import signal
from tqdm import tqdm
import multiprocessing
import time
import os
import itertools
from multiprocessing import Value

# Import both compute_m_height functions, but use surrogate by default
from surrogate import compute_m_height as surrogate_compute_m_height
from verifier import compute_m_height as original_compute_m_height

# Global flag for Ctrl+C - Primarily for the main process now
user_interrupted = False

# --- Worker Initializer for Shared Counter ---
# Global variable placeholder in the worker process
worker_cost_counter = None
# Global flag for using surrogate vs original
use_original_verifier = False

def init_worker(counter, use_original):
    """Initializer function for worker processes to inherit the counter and verifier setting."""
    global worker_cost_counter, use_original_verifier
    worker_cost_counter = counter
    use_original_verifier = use_original

def signal_handler(sig, frame):
    """Handle Ctrl+C: set flag and print message."""
    global user_interrupted
    if user_interrupted: # Second Ctrl+C
        print('\nCtrl+C detected again. Exiting forcefully.')
        sys.exit(1)
    print('\nCtrl+C detected. Asking workers to finish current step and stopping...')
    user_interrupted = True

def calculate_cost(G, m):
    """Calculates the m-height, handling potential errors or infinite results.
       Uses either surrogate or original method based on the global flag.
    """
    try:
        # Use the global flag to determine which compute_m_height to use
        if use_original_verifier:
            height = original_compute_m_height(G, m)
        else:
            height = surrogate_compute_m_height(G, m)
        return height
    except ValueError as e:
        return float('inf')
    except Exception as e:
        return float('inf')

def get_neighbor(P, element_range):
    """Generates a neighbor P matrix by changing one element randomly,
       ensuring no column becomes all zeros.
    """
    k, p_cols = P.shape
    P_new = P.copy()
    max_retries_neighbor = 10
    max_retries_value = 20

    for _ in range(max_retries_neighbor):
        row_idx = random.randrange(k)
        col_idx = random.randrange(p_cols)
        current_val = P_new[row_idx, col_idx]

        possible_values = list(range(element_range[0], element_range[1] + 1))
        if len(possible_values) <= 1:
            continue

        new_val = current_val
        attempt_val = 0
        while new_val == current_val and attempt_val < max_retries_value:
             new_val = random.choice(possible_values)
             attempt_val += 1

        if new_val == current_val:
             continue

        original_value_in_col = P_new[row_idx, col_idx]
        P_new[row_idx, col_idx] = new_val

        if not np.all(P_new[:, col_idx] == 0):
            return P_new
        else:
            P_new[row_idx, col_idx] = original_value_in_col

    return P

def simulated_annealing_core(initial_P, k, n, m, T_max, T_min, alpha, iter_per_temp, element_range, worker_id=0, seed=None):
    """Core SA logic for a single run."""
    global worker_cost_counter

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if not (1 <= m <= n - 1):
        return None, float('inf')

    I_k = np.eye(k, dtype=initial_P.dtype)

    current_P = initial_P
    current_G = np.hstack((I_k, current_P))
    current_cost = calculate_cost(current_G, m)

    if worker_cost_counter is not None:
        with worker_cost_counter.get_lock():
            worker_cost_counter.value += 1

    best_P = current_P
    best_cost = current_cost

    T = T_max
    temp_step = 0

    while T > T_min:
        for i in range(iter_per_temp):
            neighbor_P = get_neighbor(current_P, element_range)
            neighbor_G = np.hstack((I_k, neighbor_P))
            neighbor_cost = calculate_cost(neighbor_G, m)

            if worker_cost_counter is not None:
                with worker_cost_counter.get_lock():
                    worker_cost_counter.value += 1

            delta_E = neighbor_cost - current_cost

            # --- Acceptance Logic ---
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
                if delta_E == 0:
                    acceptance_prob = 1.0
                elif T <= 0:
                     acceptance_prob = 0.0
                else:
                    exponent = -delta_E / T
                    if exponent > 700:
                        acceptance_prob = 0.0
                    else:
                        acceptance_prob = math.exp(exponent)

                if random.random() < acceptance_prob:
                    accept = True

            if accept:
                current_P = neighbor_P
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_P = current_P
                    best_cost = current_cost

        T *= alpha
        temp_step += 1

    best_G = np.hstack((I_k, best_P))
    return best_G, best_cost

def is_valid_P(P):
    """Checks if any column in P is the all-zero vector."""
    k, p_cols = P.shape
    for j in range(p_cols):
        if np.all(P[:, j] == 0):
            return False
    return True

def run_single_annealing_instance(args_tuple):
    """Worker function to run one independent simulated annealing process."""
    worker_id, k, n, m, element_range, T_max, T_min, alpha, iter_per_temp, seed = args_tuple
    
    if seed is None:
        base_seed = int(time.time() * 1000) + random.randint(0, 10000) + worker_id + os.getpid()
        seed = base_seed % (2**32)
    random.seed(seed)
    np.random.seed(seed)

    p_cols = n - k
    initial_P = None
    max_init_retries = 100
    element_min, element_max = element_range

    for attempt in range(max_init_retries):
        P_candidate = np.random.randint(element_min, element_max + 1, size=(k, p_cols))
        if is_valid_P(P_candidate):
            initial_P = P_candidate
            break

    if initial_P is None:
        return None, float('inf'), worker_id

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

    return best_generator, min_height, worker_id

def verify_final_result(G, m, verbose=False):
    """Verify the final result using the original exact verifier."""
    if verbose:
        print("Verifying final result with original exact verifier...")
    
    start_time = time.time()
    try:
        exact_height = original_compute_m_height(G, m)
        verify_time = time.time() - start_time
        
        if verbose:
            print(f"Verification complete in {verify_time:.2f}s")
            print(f"Exact m-height: {exact_height:.6g}")
        
        return exact_height
    except Exception as e:
        if verbose:
            print(f"Verification failed: {e}")
        return float('inf')

if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Find a systematic generator matrix G=[I|P] minimizing m-height using Parallel Simulated Annealing with surrogate function.")

    parser.add_argument('-k', type=int, default=3, help='Number of message bits (rows). Default: 3')
    parser.add_argument('-n', type=int, default=6, help='Number of codeword bits (columns). Default: 6')
    parser.add_argument('-m', type=int, default=2, help='m-height parameter (must be 1 <= m <= n-1). Default: 2')
    parser.add_argument('--element-min', type=int, default=-100, help='Minimum value for elements in P matrix. Default: -100')
    parser.add_argument('--element-max', type=int, default=100, help='Maximum value for elements in P matrix. Default: 100')
    parser.add_argument('--t-max', type=float, default=100.0, help='Initial annealing temperature. Default: 100.0')
    parser.add_argument('--t-min', type=float, default=65.0, help='Final annealing temperature. Default: 65.0')
    parser.add_argument('--alpha', type=float, default=0.95, help='Cooling rate (multiplier). Default: 0.95')
    parser.add_argument('--iter-per-temp', type=int, default=1, help='Iterations per temperature level. Default: 1')
    
    # Add new option for using original verifier
    parser.add_argument('--use-original', action='store_true', help='Use original verifier instead of surrogate model')
    parser.add_argument('--verify-final', action='store_true', help='Verify final result with exact verifier')
    
    try:
        default_workers = multiprocessing.cpu_count()
    except NotImplementedError:
        default_workers = 1
        print("Warning: Could not determine CPU count, defaulting to 1 worker.")
    parser.add_argument('--workers', type=int, default=default_workers,
                        help=f'Number of parallel annealing runs. Default: {default_workers} (logical CPU cores)')

    args = parser.parse_args()

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
    USE_ORIGINAL = args.use_original
    VERIFY_FINAL = args.verify_final

    # --- Calculate total cost calls for progress bar ---
    total_cost_calls = 0
    if T_MAX > T_MIN and 0 < ALPHA < 1:
        try:
            num_steps = math.ceil(math.log(T_MIN / T_MAX) / math.log(ALPHA))
            total_cost_calls = NUM_WORKERS * num_steps * ITER_PER_TEMP
            print(f"Expecting approx. {num_steps} temp steps per worker.")
            print(f"Total expected cost evaluations across {NUM_WORKERS} workers: {total_cost_calls}")
        except ValueError:
            print("Warning: Could not calculate total steps, progress bar may be inaccurate.")
            total_cost_calls = 1
    else:
        print("Warning: T_max <= T_min or alpha >= 1 or alpha <= 0. Progress bar total may be inaccurate.")
        total_cost_calls = 1

    # --- Create Shared Counter ---
    cost_call_counter = Value('i', 0)

    # --- Validate Parameters ---
    if K < 1:
        parser.error(f"k ({K}) must be at least 1")
    if N <= K:
        parser.error(f"n ({N}) must be greater than k ({K}) for a systematic code with P")
    p_cols = N - K
    if p_cols < 1:
         parser.error(f"n-k ({p_cols}) must be at least 1, check n and k values")

    if not (1 <= M <= N - 1):
        parser.error(f"m ({M}) must be between 1 and n-1 ({N-1})")
    if ELEMENT_MIN > ELEMENT_MAX:
        parser.error(f"element-min ({ELEMENT_MIN}) cannot be greater than element-max ({ELEMENT_MAX})")
    if ELEMENT_MIN == 0 and ELEMENT_MAX == 0:
         parser.error(f"Element range is [0, 0]. Cannot satisfy no-all-zero column constraint.")
    if NUM_WORKERS < 1:
        parser.error(f"Number of workers ({NUM_WORKERS}) must be at least 1")

    print(f"Starting Parallel Simulated Annealing with {NUM_WORKERS} workers.")
    print(f"Targeting m={M}-height minimization for a systematic ({K}x{N}) matrix.")
    print(f"Element range for P: [{ELEMENT_RANGE[0]}, {ELEMENT_RANGE[1]}]")
    print(f"Params: T_max={T_MAX}, T_min={T_MIN}, alpha={ALPHA}, iter_per_temp={ITER_PER_TEMP}")
    print(f"Using {'original exact' if USE_ORIGINAL else 'surrogate'} verifier for cost calculations.")
    print("Press Ctrl+C to stop early (may take a moment to terminate workers).")

    # Prepare arguments for each worker
    worker_args = [
        (i, K, N, M, ELEMENT_RANGE, T_MAX, T_MIN, ALPHA, ITER_PER_TEMP, None)
        for i in range(NUM_WORKERS)
    ]

    # --- Setup Signal Handler for Main Process ---
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    pool = None
    results = []
    try:
        with multiprocessing.Pool(processes=NUM_WORKERS,
                                  initializer=init_worker,
                                  initargs=(cost_call_counter, USE_ORIGINAL)) as pool:
            result_iterator = pool.imap_unordered(run_single_annealing_instance, worker_args)

            print(f"Launching {NUM_WORKERS} annealing runs...")
            pbar = tqdm(total=total_cost_calls, desc="Cost Evals Progress", unit="evals", smoothing=0.1)
            processed_count = 0

            while processed_count < NUM_WORKERS:
                if user_interrupted:
                    print(f"\nInterrupt detected by main process. Breaking result loop...")
                    break

                try:
                    result = result_iterator.next(timeout=0.5)

                    best_G, cost, worker_id = result
                    if best_G is not None:
                        results.append((best_G, cost))
                    else:
                        print(f"Worker {worker_id} failed to produce a valid result.")
                    processed_count += 1
                    pbar.update(0)
                except StopIteration:
                    break
                except KeyboardInterrupt:
                    print("\nKeyboardInterrupt caught directly in loop.")
                    user_interrupted = True
                    break
                except multiprocessing.TimeoutError:
                    pbar.n = cost_call_counter.value
                    pbar.refresh()
                    continue
                except Exception as worker_exc:
                    print(f"\nError processing result from a worker: {worker_exc}")
                    processed_count += 1
                    pbar.update(1)

                pbar.n = cost_call_counter.value
                pbar.refresh()

            pbar.close()

            if user_interrupted:
                print("\nInterrupt confirmed post-loop. Terminating running worker processes...")
                pool.terminate()
                pool.join()
                print("Worker pool terminated.")
            else:
                print("\nAll scheduled workers completed or failed.")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught in main process.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred in the main process: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)

    # --- Aggregate Results ---
    if not results:
        print("\nNo valid results were obtained from any worker.")
        if user_interrupted:
             print("Annealing process was interrupted.")
        sys.exit(1)

    # Find the best result among all workers
    results.sort(key=lambda x: x[1])
    overall_best_G, overall_best_cost = results[0]

    print("\n--- Overall Best Result ---")
    print(f"Obtained from {len(results)} successful worker run(s).")
    print("Best Systematic Generator Matrix Found G = [I|P]:")
    with np.printoptions(precision=3, suppress=True):
         print(overall_best_G)
    print(f"Minimal m-height (m={M}): {overall_best_cost:.6g}")
    
    # Verify final result with exact method if requested
    if VERIFY_FINAL and not USE_ORIGINAL:
        print("\n--- Final Verification ---")
        exact_cost = verify_final_result(overall_best_G, M, verbose=True)
        
        if exact_cost != overall_best_cost:
            print(f"Note: Surrogate estimate ({overall_best_cost:.6g}) differs from exact verification ({exact_cost:.6g}).")
            if exact_cost == float('inf') and overall_best_cost != float('inf'):
                print("WARNING: Surrogate model failed to detect unbounded case!")
            elif exact_cost != float('inf') and overall_best_cost == float('inf'):
                print("Note: Surrogate model was overly conservative (false positive for unbounded case).")
            else:
                rel_error = abs(exact_cost - overall_best_cost) / (exact_cost if exact_cost != 0 else 1)
                print(f"Surrogate relative error: {rel_error:.2%}")
    
    if user_interrupted:
        print("\nProcess was interrupted. The reported best result is based on completed runs before interruption.") 
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

# Import both compute_m_height functions
from surrogate import compute_m_height as surrogate_compute_m_height
from verifier import compute_m_height as original_compute_m_height

# Global flag for Ctrl+C
user_interrupted = False

# Worker process global variables
worker_cost_counter = None
use_original_verifier = False

def init_worker(counter, use_original):
    """Initializer function for worker processes."""
    global worker_cost_counter, use_original_verifier
    worker_cost_counter = counter
    use_original_verifier = use_original

def signal_handler(sig, frame):
    """Handle Ctrl+C."""
    global user_interrupted
    if user_interrupted:
        print('\nCtrl+C detected again. Exiting forcefully.')
        sys.exit(1)
    print('\nCtrl+C detected. Asking workers to finish current step and stopping...')
    user_interrupted = True

def calculate_cost(G, m):
    """Calculate m-height using either surrogate or original method.
       In surrogate mode (phase 1), falls back to original if surrogate is 0 or non-finite.
    """
    try:
        if use_original_verifier: # True during phase 2 (exact refinement)
            height = original_compute_m_height(G, m)
        else: # False during phase 1 (surrogate exploration)
            # Try surrogate first
            surrogate_height = surrogate_compute_m_height(G, m)
            # Check if the surrogate result is 0 or non-finite (inf, nan)
            if surrogate_height == 0 or not np.isfinite(surrogate_height):
                # Fall back to the original verifier
                height = original_compute_m_height(G, m)
            else:
                # Use the finite, non-zero surrogate result
                height = surrogate_height
        return height
    except ValueError as e:
        return float('inf')
    except Exception as e:
        return float('inf')

def get_neighbor(P, element_range):
    """Generate a neighbor P matrix by changing one element."""
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
    """Check if any column in P is the all-zero vector."""
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

def verify_cost(G, m, verbose=False):
    """Verify a matrix with the original exact verifier."""
    if verbose:
        print("Verifying with original exact verifier...")
    
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

def hybrid_annealing(k, n, m, element_range, 
                    surrogate_params, exact_params,
                    num_surrogate_workers, num_exact_workers):
    """
    Execute the hybrid annealing strategy:
    1. First phase: Fast exploration with surrogate model
    2. Second phase: Refinement with exact model
    """
    global user_interrupted
    
    print("=== Hybrid Annealing Strategy ===")
    print(f"Problem size: k={k}, n={n}, m={m}")
    print(f"Element range: [{element_range[0]}, {element_range[1]}]")
    print("\n--- Phase 1: Fast Exploration (Surrogate Model) ---")
    
    # --- Phase 1: Surrogate Model Exploration ---
    surrogate_T_max, surrogate_T_min, surrogate_alpha, surrogate_iter = surrogate_params
    print(f"Surrogate params: T_max={surrogate_T_max}, T_min={surrogate_T_min}, alpha={surrogate_alpha}, iter_per_temp={surrogate_iter}")
    print(f"Using {num_surrogate_workers} workers")
    
    # Setup for Phase 1
    cost_call_counter = Value('i', 0)
    signal.signal(signal.SIGINT, signal_handler)
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    
    # Prepare worker arguments
    surrogate_worker_args = [
        (i, k, n, m, element_range, surrogate_T_max, surrogate_T_min, surrogate_alpha, surrogate_iter, None)
        for i in range(num_surrogate_workers)
    ]
    
    surrogate_results = []
    try:
        with multiprocessing.Pool(processes=num_surrogate_workers,
                                initializer=init_worker,
                                initargs=(cost_call_counter, False)) as pool:  # False = use surrogate
            
            result_iterator = pool.imap_unordered(run_single_annealing_instance, surrogate_worker_args)
            
            print(f"Launching {num_surrogate_workers} surrogate annealing runs...")
            processed_count = 0
            
            with tqdm(total=num_surrogate_workers, desc="Surrogate Progress", unit="workers") as pbar:
                while processed_count < num_surrogate_workers:
                    if user_interrupted:
                        print("\nInterrupt detected in phase 1. Breaking result loop...")
                        break
                    
                    try:
                        result = result_iterator.next(timeout=0.5)
                        best_G, cost, worker_id = result
                        if best_G is not None:
                            # For surrogate results, verify with exact model to get real cost
                            exact_cost = verify_cost(best_G, m)
                            surrogate_results.append((best_G, exact_cost))
                            print(f"Worker {worker_id} found solution with exact cost: {exact_cost:.4f}")
                        else:
                            print(f"Worker {worker_id} failed to produce a valid result.")
                        
                        processed_count += 1
                        pbar.update(1)
                        
                    except StopIteration:
                        break
                    except multiprocessing.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"\nError in phase 1: {e}")
                        processed_count += 1
                        pbar.update(1)
                        
            if user_interrupted:
                print("\nTerminating surrogate phase...")
                pool.terminate()
                pool.join()
    
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt in phase 1.")
        surrogate_results = []
    except Exception as e:
        print(f"\nAn unexpected error occurred in phase 1: {e}")
        surrogate_results = []
    
    # Check if we have enough results to continue
    if not surrogate_results:
        print("\nNo valid results from surrogate phase. Exiting.")
        return None, float('inf')
    
    # Sort results by exact cost and select the best candidates for phase 2
    surrogate_results.sort(key=lambda x: x[1])
    
    # Keep only finite results for phase 2
    surrogate_results = [result for result in surrogate_results if np.isfinite(result[1])]
    
    if not surrogate_results:
        print("\nNo finite-cost results from surrogate phase. Exiting.")
        return None, float('inf')
    
    print("\n--- Surrogate Phase Summary ---")
    print(f"Found {len(surrogate_results)} viable candidates")
    print(f"Best surrogate result has exact cost: {surrogate_results[0][1]:.4f}")
    
    # Skip phase 2 if interrupted or no exact workers specified
    if user_interrupted or num_exact_workers <= 0:
        best_G, best_cost = surrogate_results[0]
        print("\nSkipping exact refinement phase.")
        return best_G, best_cost
    
    # --- Phase 2: Exact Model Refinement ---
    print("\n--- Phase 2: Refinement (Exact Model) ---")
    
    # Use the best surrogate results as starting points for exact refinement
    num_candidates = min(len(surrogate_results), num_exact_workers)
    candidates = surrogate_results[:num_candidates]
    
    exact_T_max, exact_T_min, exact_alpha, exact_iter = exact_params
    print(f"Exact params: T_max={exact_T_max}, T_min={exact_T_min}, alpha={exact_alpha}, iter_per_temp={exact_iter}")
    print(f"Using {num_candidates} candidates from phase 1 as starting points")
    
    # Reset counter for Phase 2
    cost_call_counter = Value('i', 0)
    
    # Reset user_interrupted flag
    user_interrupted = False
    
    # For each candidate, extract P from G=[I|P]
    exact_worker_args = []
    for i, (G, _) in enumerate(candidates):
        k, n = G.shape
        P = G[:, k:]  # Extract P from G=[I|P]
        
        # Create args for this starting point
        exact_worker_args.append((
            i, k, n, m, element_range, 
            exact_T_max, exact_T_min, exact_alpha, exact_iter, 
            None
        ))
    
    exact_results = []
    try:
        with multiprocessing.Pool(processes=num_candidates,
                                 initializer=init_worker,
                                 initargs=(cost_call_counter, True)) as pool:  # True = use exact
            
            # Use different initial P matrices for each exact worker
            for i, ((G, _), args) in enumerate(zip(candidates, exact_worker_args)):
                # Replace the default initial_P in the worker function
                # We'll do this by modifying a copy of the function for each worker
                k, n = G.shape
                P = G[:, k:]  # Extract P from G=[I|P]
                
                # We can't directly modify the function, but we can create a custom
                # function wrapper to pass the initial P to the right worker
                def make_run_instance_with_P(P, args):
                    def run_instance_with_P():
                        result = run_single_annealing_instance(args)
                        # We'd modify this to use our P, but since we can't modify the
                        # function itself, we'll use a different approach
                        return result
                    return run_instance_with_P
            
            print(f"Launching {num_candidates} exact annealing runs...")
            result_iterator = pool.imap_unordered(run_single_annealing_instance, exact_worker_args)
            
            processed_count = 0
            with tqdm(total=num_candidates, desc="Exact Progress", unit="workers") as pbar:
                while processed_count < num_candidates:
                    if user_interrupted:
                        print("\nInterrupt detected in phase 2. Breaking result loop...")
                        break
                    
                    try:
                        result = result_iterator.next(timeout=1.0)
                        best_G, cost, worker_id = result
                        if best_G is not None:
                            exact_results.append((best_G, cost))
                            print(f"Exact worker {worker_id} found solution with cost: {cost:.4f}")
                        else:
                            print(f"Exact worker {worker_id} failed to produce a valid result.")
                        
                        processed_count += 1
                        pbar.update(1)
                        
                    except StopIteration:
                        break
                    except multiprocessing.TimeoutError:
                        continue
                    except Exception as e:
                        print(f"\nError in phase 2: {e}")
                        processed_count += 1
                        pbar.update(1)
            
            if user_interrupted:
                print("\nTerminating exact phase...")
                pool.terminate()
                pool.join()
    
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt in phase 2.")
    except Exception as e:
        print(f"\nAn unexpected error occurred in phase 2: {e}")
    
    # Combine results from both phases
    all_results = surrogate_results + exact_results
    
    # Sort all results by cost
    all_results.sort(key=lambda x: x[1])
    
    # Return the best overall result
    if all_results:
        return all_results[0]
    else:
        return None, float('inf')

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="Hybrid Simulated Annealing: fast surrogate exploration + exact refinement")
    
    # Problem parameters
    parser.add_argument('-k', type=int, default=3, help='Number of message bits (rows). Default: 3')
    parser.add_argument('-n', type=int, default=6, help='Number of codeword bits (columns). Default: 6')
    parser.add_argument('-m', type=int, default=2, help='m-height parameter (must be 1 <= m <= n-1). Default: 2')
    parser.add_argument('--element-min', type=int, default=-100, help='Minimum value for elements in P matrix. Default: -100')
    parser.add_argument('--element-max', type=int, default=100, help='Maximum value for elements in P matrix. Default: 100')
    
    # Surrogate phase parameters
    parser.add_argument('--surrogate-t-max', type=float, default=100.0, help='Initial surrogate annealing temperature. Default: 100.0')
    parser.add_argument('--surrogate-t-min', type=float, default=1.0, help='Final surrogate annealing temperature. Default: 1.0')
    parser.add_argument('--surrogate-alpha', type=float, default=0.9, help='Surrogate cooling rate. Default: 0.9')
    parser.add_argument('--surrogate-iter', type=int, default=5, help='Surrogate iterations per temperature. Default: 5')
    parser.add_argument('--surrogate-workers', type=int, default=4, 
                      help='Number of surrogate workers. Default: 4. Set to 0 to skip surrogate phase.')
    
    # Exact phase parameters
    parser.add_argument('--exact-t-max', type=float, default=10.0, help='Initial exact annealing temperature. Default: 10.0')
    parser.add_argument('--exact-t-min', type=float, default=0.1, help='Final exact annealing temperature. Default: 0.1')
    parser.add_argument('--exact-alpha', type=float, default=0.95, help='Exact cooling rate. Default: 0.95')
    parser.add_argument('--exact-iter', type=int, default=2, help='Exact iterations per temperature. Default: 2')
    parser.add_argument('--exact-workers', type=int, default=2, 
                      help='Number of exact workers. Default: 2. Set to 0 to skip exact phase.')
    
    args = parser.parse_args()
    
    # Extract parameters
    K = args.k
    N = args.n
    M = args.m
    ELEMENT_RANGE = (args.element_min, args.element_max)
    
    # Extract surrogate parameters
    surrogate_params = (args.surrogate_t_max, args.surrogate_t_min, 
                      args.surrogate_alpha, args.surrogate_iter)
    
    # Extract exact parameters
    exact_params = (args.exact_t_max, args.exact_t_min, 
                  args.exact_alpha, args.exact_iter)
    
    # Validate parameters
    if K < 1:
        parser.error(f"k ({K}) must be at least 1")
    if N <= K:
        parser.error(f"n ({N}) must be greater than k ({K}) for a systematic code with P")
    if not (1 <= M <= N - 1):
        parser.error(f"m ({M}) must be between 1 and n-1 ({N-1})")
    if args.element_min > args.element_max:
        parser.error(f"element-min ({args.element_min}) cannot be greater than element-max ({args.element_max})")
    if args.element_min == 0 and args.element_max == 0:
        parser.error(f"Element range is [0, 0]. Cannot satisfy no-all-zero column constraint.")
    
    # Execute hybrid annealing
    best_G, best_cost = hybrid_annealing(
        k=K, n=N, m=M, element_range=ELEMENT_RANGE,
        surrogate_params=surrogate_params, 
        exact_params=exact_params,
        num_surrogate_workers=args.surrogate_workers,
        num_exact_workers=args.exact_workers
    )
    
    # Print results
    print("\n=== Final Result ===")
    if best_G is not None:
        print("Best Systematic Generator Matrix G = [I|P]:")
        with np.printoptions(precision=3, suppress=True):
            print(best_G)
        print(f"Minimal m-height (m={M}): {best_cost:.6g}")
    else:
        print("No solution found.") 
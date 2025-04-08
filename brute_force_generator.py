import numpy as np
import math
import itertools
import argparse
import sys
import threading
import time
import os
try:
    # Assuming compute_m_height is in verifier.py at the same level
    from verifier import compute_m_height
except ImportError:
    print("Error: Could not import 'compute_m_height' from 'verifier.py'.")
    print("Ensure 'verifier.py' exists and is in the same directory or Python path.")
    sys.exit(1)

# Global variables for best result and lock
best_P_global = None
min_cost_global = float('inf')
lock = threading.Lock()
cost_counter = 0 # Simple counter (approximate in threaded env)
cost_counter_lock = threading.Lock() # Lock for accurate counting if needed, but adds overhead

def calculate_cost(G, m):
    """Calculates the m-height, incrementing counter, handling errors."""
    global cost_counter, cost_counter_lock
    # Increment counter (using lock for accuracy, remove lock for less overhead)
    with cost_counter_lock:
        cost_counter += 1
    try:
        # Assuming compute_m_height handles potential LinAlgError etc.
        height = compute_m_height(G, m)
        # Ensure infinity is returned consistently if calculation fails
        return height if np.isfinite(height) else float('inf')
    except Exception as e:
        # print(f"Warning: compute_m_height failed. Error: {e}") # Optional debug
        return float('inf')

def is_valid_P(P):
    """Checks if any column in P is the all-zero vector."""
    # Check if P is empty or dimensions are invalid before proceeding
    if P is None or P.size == 0 or P.ndim != 2:
         return False # Cannot be valid if not a proper 2D array
    k, p_cols = P.shape
    if p_cols == 0:
         return True # P has no columns, so no all-zero columns exist
    for j in range(p_cols):
        if np.all(P[:, j] == 0):
            return False
    return True

def get_p_matrix_from_index(index, k, p_cols, element_min, element_max):
    """
    Generates the P matrix corresponding to a specific index in the
    iteration space defined by element_min/max and dimensions k, p_cols.
    Maps a single integer index to a unique matrix configuration.
    """
    num_elements = k * p_cols
    range_size = element_max - element_min + 1

    # Handle edge cases
    if range_size <= 0:
        # print(f"Warning: Invalid element range size ({range_size}).")
        return None
    if num_elements == 0:
         # If k=0 or p_cols=0, P should be empty or have zero elements.
         # Return an appropriately shaped empty array.
         return np.empty((k, p_cols), dtype=np.int64)

    # Check if index is out of bounds for the total number of possibilities.
    # Be careful with large exponents.
    try:
        total_candidates = range_size ** num_elements
        if index >= total_candidates:
            # This indicates an issue with the calling logic if it happens within the intended loop range.
            # print(f"Warning: Index {index} is out of bounds ({total_candidates}).")
            return None
    except OverflowError:
        # If the total number itself overflows, we can't directly compare.
        # This scenario implies an extremely large search space.
        # The index calculation below might still work if index is representable.
        pass # Proceed with calculation, might still work for smaller indices

    elements = []
    temp_index = index
    for _ in range(num_elements):
        # Derives the value for each position based on the index in the 'range_size' base system.
        element_value = temp_index % range_size
        elements.append(element_min + element_value)
        temp_index //= range_size

    # The elements list is built from the 'least significant' element first,
    # based on the base conversion. Reverse it to match standard matrix filling order (e.g., row-major).
    elements.reverse()

    try:
      # Reshape the flat list of elements into the (k, p_cols) matrix.
      P = np.array(elements, dtype=np.int64).reshape((k, p_cols))
      return P
    except ValueError:
      # This error occurs if the number of elements generated doesn't match k * p_cols.
      # Should not happen with the current logic unless num_elements was calculated incorrectly.
      print(f"Error: Reshaping failed for index {index}. Num elements: {len(elements)}, Expected: {k*p_cols}")
      return None


def worker_function(worker_id, start_index, end_index, k, n, m, element_min, element_max, progress_interval):
    """Worker thread function to check a range of P matrix candidates."""
    global best_P_global, min_cost_global, lock
    # print(f"Worker {worker_id}: Checking indices [{start_index}, {end_index})") # Debug
    I_k = np.eye(k, dtype=np.int64) # Match dtype used in get_p_matrix_from_index
    p_cols = n - k
    local_best_P = None
    local_min_cost = float('inf')
    checked_count = 0

    for index in range(start_index, end_index):
        # Optional progress within worker (less frequent)
        if progress_interval > 0 and checked_count > 0 and checked_count % progress_interval == 0:
             with lock: # Update global under lock if better than current global
                 if local_min_cost < min_cost_global:
                      # Avoid printing too often, just update
                      min_cost_global = local_min_cost
                      best_P_global = local_best_P # Already copied below

        P = get_p_matrix_from_index(index, k, p_cols, element_min, element_max)

        if P is None: # Should not happen within the loop range normally
            # print(f"Worker {worker_id}: Warning - Failed to generate P for index {index}")
            continue # Skip this index if generation failed

        if not is_valid_P(P):
            continue # Skip if P has an all-zero column

        # Construct the full generator matrix G = [I|P]
        G = np.hstack((I_k, P))
        cost = calculate_cost(G, m) # Calculate m-height

        # Update the best result found *locally* within this thread
        if cost < local_min_cost:
            local_min_cost = cost
            local_best_P = P.copy() # Create a copy to avoid race conditions

        checked_count += 1

    # After checking all assigned indices, update the global best result
    # This reduces lock contention compared to updating on every improvement
    if local_best_P is not None:
        with lock: # Acquire lock to safely compare and update global variables
            if local_min_cost < min_cost_global:
                # Check again in case another thread updated global best in the meantime
                # print(f"Worker {worker_id}: Found new best cost {local_min_cost:.4f} (was {min_cost_global:.4f})") # Progress indicator
                min_cost_global = local_min_cost
                best_P_global = local_best_P # Assign the locally found best P
                # Optionally print the new best P here if needed for debugging
                # print("New best P:\n", best_P_global)

    # print(f"Worker {worker_id}: Finished. Checked {checked_count} valid candidates. Local best cost: {local_min_cost}") # Debug


def main():
    global best_P_global, min_cost_global, cost_counter # Allow modification

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Find the optimal systematic generator matrix G=[I|P] minimizing m-height using multi-threaded brute force search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument('-k', type=int, required=True, help='Number of message bits (rows).')
    parser.add_argument('-n', type=int, required=True, help='Number of codeword bits (columns).')
    parser.add_argument('-m', type=int, required=True, help='m-height parameter (must be 1 <= m <= n-1).')
    parser.add_argument('--element-min', type=int, default=0, help='Minimum value for elements in P matrix.')
    parser.add_argument('--element-max', type=int, default=1, help='Maximum value for elements in P matrix.')
    # Default workers to CPU count, ensuring at least 1
    default_workers = max(1, os.cpu_count() or 1)
    parser.add_argument('--workers', type=int, default=default_workers,
                        help=f'Number of parallel worker threads.')

    args = parser.parse_args()

    K = args.k
    N = args.n
    M = args.m
    ELEMENT_MIN = args.element_min
    ELEMENT_MAX = args.element_max
    NUM_WORKERS = args.workers

    # --- Validate Parameters ---
    if K < 1: parser.error("k must be at least 1")
    if N <= K: parser.error("n must be greater than k for P matrix to exist")
    p_cols = N - K
    # p_cols can be 0 if N=K, but G=[I] in that case. The logic handles this.
    # if p_cols < 1: parser.error("n-k must be at least 1") # Allow n=k, P is empty
    if not (1 <= M <= N - 1): parser.error(f"m ({M}) must be between 1 and n-1 ({N-1})")
    if ELEMENT_MIN > ELEMENT_MAX: parser.error("element-min cannot be greater than element-max")
    if NUM_WORKERS < 1: parser.error("Number of workers must be at least 1")
    # Specific check: If range is only 0, and P exists (p_cols > 0), invalid P is guaranteed.
    if ELEMENT_MIN == 0 and ELEMENT_MAX == 0 and p_cols > 0:
         parser.error("Element range is [0, 0] and P matrix exists (n > k). Cannot satisfy no-all-zero column constraint.")

    # --- Calculate Search Space Size ---
    range_size = ELEMENT_MAX - ELEMENT_MIN + 1
    num_p_elements = K * p_cols
    total_candidates = 0

    if range_size <= 0:
        print("Error: Element range is empty or invalid (min > max).")
        sys.exit(1)
    elif num_p_elements == 0: # Case n=k, P is empty k x 0 matrix
         total_candidates = 1 # There's one matrix G=[I]
         # print("Note: n=k, the only possible matrix is G=[I_k].") # Suppress this note
    else:
        try:
            # Calculate total number of P matrices (before filtering zero columns)
            total_candidates = range_size ** num_p_elements
            # Check for potential overflow if numbers are massive
            if total_candidates == float('inf'):
                 print("Warning: Search space is extremely large (overflow detected).") # Keep warning
            # Provide a warning for large, but representable, search spaces. 10^12 is arbitrary.
            elif total_candidates > 10**12:
                 print(f"Warning: Search space is very large ({total_candidates:.2e} candidates). This will likely take a very long time.") # Keep warning, simplified
            elif total_candidates == 0: # Should only happen if range_size=0, already checked
                 print("Warning: Calculated zero candidates unexpectedly.") # Keep warning

        except OverflowError:
            print(f"Error: Search space size calculation ({range_size}^{num_p_elements}) resulted in overflow.") # Keep error
            print("Parameters (k, n, element range) are too large for brute force.")
            print("Consider reducing the element range or using the annealing generator for larger problems.")
            sys.exit(1)

    if total_candidates == 0 and num_p_elements > 0:
        print("No possible P matrices to check with the given parameters (excluding n=k case).") # Keep info
        sys.exit(0)

    # --- Start Search ---
    # Suppress these startup messages
    # print(f"Starting Brute Force Search for ({K}x{N}) matrix, m={M}-height.")
    # print(f"Element range for P: [{ELEMENT_MIN}, {ELEMENT_MAX}]")
    # if num_p_elements > 0:
    #     print(f"Total P matrix candidates to check (before zero-column filter): {total_candidates}")
    # print(f"Using {NUM_WORKERS} worker threads.")
    print("Starting brute force search... Press Ctrl+C to interrupt.") # Keep simple start message with interrupt info

    # --- Thread Setup ---
    threads = []
    # Ceiling division to distribute work as evenly as possible
    chunk_size = (total_candidates + NUM_WORKERS - 1) // NUM_WORKERS
    # Set progress update interval (e.g., check global best every X candidates)
    # Adjust based on expected runtime; larger interval reduces lock contention.
    progress_interval = max(1000, total_candidates // (NUM_WORKERS * 100)) # Heuristic

    start_time = time.time()

    for i in range(NUM_WORKERS):
        # Calculate the start and end index for this thread's chunk
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, total_candidates)

        # If start_index >= end_index, this thread has no work (can happen with few candidates)
        if start_index >= end_index:
            continue

        # Create and start the thread
        thread = threading.Thread(target=worker_function,
                                  args=(i, start_index, end_index, K, N, M, ELEMENT_MIN, ELEMENT_MAX, progress_interval))
        threads.append(thread)
        thread.start()

    # --- Wait for Threads and Display Progress ---
    try:
        # While any thread is alive, keep updating progress
        while any(t.is_alive() for t in threads):
            # Update progress based on the shared counter
            with cost_counter_lock: # Access counter safely
                 current_checked = cost_counter

            # Read global best cost safely under the main lock
            with lock:
                current_best_cost = min_cost_global

            elapsed_time = time.time() - start_time
            rate = current_checked / elapsed_time if elapsed_time > 0 else 0
            percent_complete = (current_checked / total_candidates) * 100 if total_candidates > 0 else 100

            # Format cost: display 'inf' or a number using the locked value
            cost_display = f"{current_best_cost:.4f}" if current_best_cost != float('inf') else "inf"

            # Write progress to the same line
            sys.stdout.write(
                f"\rProgress: {current_checked}/{total_candidates} ({percent_complete:.2f}%) | Rate: {rate:.1f}/s | Elapsed: {elapsed_time:.1f}s | Best Cost: {cost_display}   "
            )
            sys.stdout.flush()

            # Sleep briefly to avoid busy-waiting and reduce print frequency
            time.sleep(0.5)

        # Final join ensures all threads have completed their execution
        for thread in threads:
            thread.join()
        # Print a newline to move off the progress line cleanly
        print() # Ensures the next print starts on a new line after progress bar
        # print("All workers finished.") # Suppress this

    except KeyboardInterrupt:
        # Keep interrupt messages
        print("\nCtrl+C detected. Interrupting search.")
        print("Note: Threads will finish their current calculation.")
        print("The result shown will be the best found up to the interruption point.")
        # Threads will exit naturally after finishing their current loop iteration.
        # We join them briefly to ensure clean exit if possible.
        for thread in threads:
             thread.join(timeout=0.2) # Give threads a moment to finish/exit
        print("Threads joined or timed out.")


    # --- Final Report ---
    end_time = time.time()
    print(f"Search completed or interrupted in {end_time - start_time:.2f} seconds.") # Keep duration
    with cost_counter_lock: # Read final count safely
         print(f"Total cost function calls: {cost_counter}") # Keep total calls

    # --- Output Result ---
    if best_P_global is not None:
        print("\n--- Optimal Systematic Generator Matrix Found G = [I|P] ---") # Keep result header
        # Ensure I_k has the same dtype as the found P matrix
        I_k = np.eye(K, dtype=best_P_global.dtype)
        # Construct the best generator matrix G = [I|P]
        best_G = np.hstack((I_k, best_P_global))
        # Print the resulting matrix with controlled precision
        with np.printoptions(precision=3, suppress=True):
             print(best_G) # Keep result matrix
        # Print the minimal m-height found
        print(f"Minimal m-height (m={M}): {min_cost_global:.6g}") # Keep result cost

        # --- Optional Verification ---
        # Suppress verification steps output
        # print("\nVerifying final cost...")
        final_cost_check = calculate_cost(best_G, M) # Keep calculation for check below
        # print(f"Verification cost calculation: {final_cost_check:.6g}")
        if not math.isclose(final_cost_check, min_cost_global, rel_tol=1e-9, abs_tol=0.0):
             print(f"Warning: Final cost verification ({final_cost_check:.6g}) differs slightly from recorded best cost ({min_cost_global:.6g}).") # Keep warning if check fails
        # else:
             # print("Verification successful.") # Suppress success message

    else:
        # Keep messages for no result found
        print("\nNo valid systematic generator matrix found satisfying the constraints within the given range.")
        if total_candidates > 0:
             print("Possible reasons: all candidates had an all-zero column, or all resulted in errors/infinite cost during m-height calculation.")
        if M > N - K: # Check if m is too large for the code rate
             print(f"Note: m={M} might be too large relative to n-k={p_cols}, potentially leading to high or infinite costs.")


if __name__ == "__main__":
    # Note on Threading vs. Multiprocessing:
    # For CPU-bound tasks like NumPy/SciPy operations often involved in
    # compute_m_height, Python's Global Interpreter Lock (GIL) limits
    # true parallelism with threading. Multiprocessing typically yields
    # better performance by bypassing the GIL using separate processes.
    # However, this implementation uses threading as initially discussed.
    main() 
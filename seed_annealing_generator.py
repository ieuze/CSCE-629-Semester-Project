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
import subprocess
import io

# --- 配置 C++ Verifier 路径 ---
# 您需要将 'path/to/your/cppheight' 替换为实际编译后的 cppheight 可执行文件路径
# 或者确保它在系统的 PATH 环境变量中。
CPP_VERIFIER_PATH = "./build/cppheight" # <--- 修改为您的路径

try:
    # 尝试找到 C++ Verifier
    if not os.path.exists(CPP_VERIFIER_PATH) or not os.access(CPP_VERIFIER_PATH, os.X_OK):
         print(f"Error: C++ verifier executable not found or not executable at '{CPP_VERIFIER_PATH}'")
         # 尝试在 PATH 中查找
         from shutil import which
         if which("cppheight"):
             print("Found 'cppheight' in PATH, will use that.")
             CPP_VERIFIER_PATH = "cppheight"
         else:
              print("Could not find 'cppheight' in PATH either.")
              sys.exit(1)

    # --- 重新导入 Python verifier ---
    from verifier import compute_m_height
    print("Successfully imported Python verifier 'compute_m_height' from 'verifier.py'.")

except ImportError:
    print("Error: Could not import 'compute_m_height' from 'verifier.py'.")
    print("Please ensure 'verifier.py' exists in the same directory or in the Python path.")
    # 如果找不到 Python verifier，我们应该决定是否继续。
    # 为了确保最终验证能进行，这里选择退出。
    sys.exit(1)
except Exception as e:
     # 处理 C++ Verifier 查找失败等其他错误
     print(f"Error during setup: {e}")
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
    """
    Calculates the m-height by calling the external C++ verifier.
    Handles potential errors or infinite results.
    """
    k, n = G.shape

    # 准备输入给 C++ 程序的字符串
    # 格式: k n m\n G矩阵 (空格分隔)\n
    input_str = f"{k} {n} {m}\n"
    # 使用 StringIO 和 np.savetxt 来格式化矩阵
    with io.StringIO() as s:
        # savetxt 默认用空格分隔，fmt='%d' 确保整数输出
        np.savetxt(s, G, fmt='%d')
        input_str += s.getvalue()

    try:
        # 运行 C++ verifier
        result = subprocess.run(
            [CPP_VERIFIER_PATH],    # 命令和参数列表
            input=input_str,        # 将字符串作为 stdin 输入
            capture_output=True,    # 捕获 stdout 和 stderr
            text=True,              # 以文本模式处理输入输出
            check=True,             # 如果 C++ 程序返回非零退出码，则抛出异常
            timeout=600             # 添加超时（例如 10 分钟）防止卡死
        )

        # 解析 stdout 输出
        output_val_str = result.stdout.strip()
        if output_val_str.lower() == 'inf':
            return float('inf')
        else:
            return float(output_val_str)

    except FileNotFoundError:
         print(f"Error: Could not find C++ verifier executable at '{CPP_VERIFIER_PATH}'.")
         return float('inf')
    except subprocess.CalledProcessError as e:
        print(f"Error: C++ verifier returned non-zero exit status {e.returncode}.")
        print(f"  Stderr: {e.stderr.strip()}")
        return float('inf') # 返回 inf 表示计算失败
    except subprocess.TimeoutExpired:
        print(f"Error: C++ verifier timed out after 600 seconds.")
        return float('inf')
    except ValueError:
        print(f"Error: Could not parse C++ verifier output: '{result.stdout.strip()}'")
        return float('inf')
    except Exception as e:
        print(f"An unexpected error occurred while calling C++ verifier: {e}")
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
    current_cost = float('inf') # Start with inf cost before first calculation

    best_P = current_P.copy()
    best_cost = float('inf')
    second_best_P = current_P.copy()
    second_best_cost = float('inf')

    print(f"Initial seed cost (m={m}, via C++): {current_cost if current_cost != float('inf') else 'inf'}")
    if best_cost == float('inf'):
        print("Warning: Initial seed matrix has infinite cost via C++. Annealing might not be effective.")


    temperature = initial_temp
    # Estimate total steps for progress bar based on when temperature drops below 0.01
    total_steps = int(steps_per_temp * (math.log(0.01 / initial_temp) / math.log(cooling_rate) if cooling_rate < 1 and initial_temp > 0.01 else 10))

    progress_bar = tqdm(total=total_steps, unit="steps", desc="Annealing", ncols=100)
    steps_done = 0

    final_P_to_verify = None
    cost_during_annealing = float('inf')
    annealing_interrupted = False

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
                for neighbor_P_list, neighbor_cost_cpp in results:
                    steps_done += 1
                    progress_bar.update(1)

                    neighbor_P = np.array(neighbor_P_list, dtype=np.int64)

                    if neighbor_cost_cpp < current_cost:
                        # Accept better solution
                        current_P = neighbor_P
                        current_cost = neighbor_cost_cpp # Update current cost (C++)
                        # Update the global best solution
                        if neighbor_cost_cpp < best_cost:
                            # Update second best to the previous best
                            second_best_P = best_P.copy()
                            second_best_cost = best_cost # Store previous best C++ cost
                            # Update best
                            best_cost = neighbor_cost_cpp # Store new best C++ cost
                            best_P = neighbor_P.copy()
                            progress_bar.set_description(f"Annealing (T={temperature:.2f}, Best (C++)={best_cost:.4f})")
                        elif neighbor_cost_cpp < second_best_cost: # Only update second best if better than current second best
                            # Update second best to the current solution
                            second_best_P = neighbor_P.copy()
                            second_best_cost = neighbor_cost_cpp # Store new second best C++ cost

                    else:
                        # Accept worse solution with a certain probability
                         if temperature > 1e-9: # Avoid division by zero if temp is extremely small
                            delta_cost = neighbor_cost_cpp - current_cost
                            # Handle potential inf - inf or inf - finite cases safely
                            if np.isfinite(delta_cost) or (neighbor_cost_cpp == float('inf') and current_cost != float('inf')):
                                acceptance_prob = math.exp(-delta_cost / temperature) if np.isfinite(delta_cost) else 0.0 # Prob is 0 if moving to inf
                                if random.random() < acceptance_prob:
                                    current_P = neighbor_P
                                    current_cost = neighbor_cost_cpp # Update current cost (C++)
                         # Else: If temperature is ~0 or delta_cost is not finite (e.g., inf - inf), don't accept worse

                # Cool down
                temperature *= cooling_rate
                progress_bar.set_description(f"Annealing (T={temperature:.2f}, Best (C++)={best_cost:.4f})")


    except KeyboardInterrupt:
        print("\nAnnealing process interrupted by user.")
        print("Will use the second best solution found for final verification.")
        final_P_to_verify = second_best_P.copy()
        cost_during_annealing = second_best_cost # Use the C++ cost found for second best
        annealing_interrupted = True
    finally:
        progress_bar.close()
        if not annealing_interrupted:
             final_P_to_verify = best_P.copy()
             cost_during_annealing = best_cost # Use the C++ cost found for best


    print("\n--- Simulated Annealing Finished ---")
    print(f"Cost found during annealing (via C++): {cost_during_annealing if cost_during_annealing != float('inf') else 'inf'}")

    # --- Final Verification using Python Verifier ---
    verified_cost = float('inf')
    if final_P_to_verify is not None and cost_during_annealing != float('inf'): # Only verify if we found a finite cost solution
        print("\nPerforming final verification using Python's compute_m_height...")
        final_G = np.hstack((I_k, final_P_to_verify))
        try:
            # Directly call the imported Python function
            verified_cost = compute_m_height(final_G, m)
            print(f"Final verified cost (via Python): {verified_cost if np.isfinite(verified_cost) else 'inf'}")
        except Exception as e:
            print(f"Error during final Python verification: {e}")
            print("Proceeding with C++ cost result for saving/output.")
            # Keep verified_cost as inf if Python verification fails
    elif final_P_to_verify is None:
         print("\nNo valid solution found during annealing to verify.")
    else: # cost_during_annealing was inf
         print("\nBest solution found during annealing had infinite cost (via C++), skipping Python verification.")


    # --- Saving / Output ---
    # Decide which result/cost to save or print based on verification success
    final_P_to_output = final_P_to_verify
    final_cost_to_report = verified_cost if np.isfinite(verified_cost) else cost_during_annealing # Prefer verified cost if finite

    if final_P_to_output is not None:
        if output_file:
             # Consider adding cost to the output file? Or just save P? Currently saves P.
            save_p_matrix(final_P_to_output, output_file)
            print(f"Final P matrix saved to '{output_file}' (corresponding cost reported above).")
        else:
            # Print the P matrix that corresponds to the final_cost_to_report
            print("Final P matrix:")
            with np.printoptions(linewidth=np.inf, threshold=np.inf): # Print full matrix
                print(final_P_to_output)
            print(f"(Reported cost: {final_cost_to_report if final_cost_to_report != float('inf') else 'inf'})")
            # Optional: Print the full G matrix as well
            # print("Corresponding G = [I|P] matrix:")
            # final_G_output = np.hstack((I_k, final_P_to_output))
            # with np.printoptions(linewidth=np.inf, threshold=np.inf):
            #      print(final_G_output)
    else:
        print("No final matrix to save or output.")


    # Return the P matrix and the cost reported (prefer verified cost)
    return final_P_to_output, final_cost_to_report


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Optimize the m-height of a systematic generator matrix G=[I|P] using simulated annealing, starting from a seed file containing n, k, m, and the P matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--seed-file', type=str, default="test.txt",
                        help='Path to the text file containing n, k, m on the first line, followed by the initial seed P matrix.')
    parser.add_argument('--initial-temp', type=float, default=100,
                        help='Initial temperature for simulated annealing.')
    parser.add_argument('--cooling-rate', type=float, default=0.9,
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
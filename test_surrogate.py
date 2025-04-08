import numpy as np
import time
import matplotlib.pyplot as plt
from verifier import compute_m_height as original_compute_m_height
from surrogate import compute_m_height as surrogate_compute_m_height

def test_accuracy_and_speed():
    """Compare the accuracy and speed of the original and surrogate models."""
    # Test matrices with different sizes
    test_cases = [
        # Small matrix
        {
            "G": np.array([[1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 2]]),
            "m_values": [1, 2, 3]
        },
        # Medium matrix
        {
            "G": np.random.randint(-5, 6, size=(4, 8)),
            "m_values": [1, 3, 5]  # 限制到最大 m=5
        },
        # Larger matrix (would be slow with original)
        {
            "G": np.random.randint(-10, 11, size=(5, 10)),
            "m_values": [1, 3, 5]  # 限制到最大 m=5
        }
    ]
    
    # For storing results
    original_times = []
    surrogate_times = []
    original_heights = []
    surrogate_heights = []
    
    print("Comparing original vs surrogate m-height calculation:")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases):
        G = test_case["G"]
        m_values = test_case["m_values"]
        k, n = G.shape
        
        print(f"\nTest Case {i+1}: G matrix shape ({k}x{n})")
        
        for m in m_values:
            # 確保 m 值不超過 n-1 (有效範圍)
            if m >= n:
                print(f"Skipping m={m} as it's >= n={n}")
                continue
                
            # Measure time for original method
            start_time = time.time()
            try:
                original_height = original_compute_m_height(G, m)
                original_time = time.time() - start_time
                original_result = f"{original_height:.4f}"
            except Exception as e:
                original_height = float('inf')
                original_time = time.time() - start_time
                original_result = f"Error: {e}"
            
            # Measure time for surrogate method
            start_time = time.time()
            try:
                surrogate_height = surrogate_compute_m_height(G, m)
                surrogate_time = time.time() - start_time
                surrogate_result = f"{surrogate_height:.4f}"
            except Exception as e:
                surrogate_height = float('inf')
                surrogate_time = time.time() - start_time
                surrogate_result = f"Error: {e}"
            
            # Store results
            original_times.append(original_time)
            surrogate_times.append(surrogate_time)
            
            if np.isfinite(original_height) and np.isfinite(surrogate_height):
                original_heights.append(original_height)
                surrogate_heights.append(surrogate_height)
            
            # Calculate speed improvement
            speed_improvement = original_time / surrogate_time if surrogate_time > 0 else float('inf')
            
            # Calculate accuracy (if both are finite)
            if np.isfinite(original_height) and np.isfinite(surrogate_height):
                rel_error = abs(surrogate_height - original_height) / max(1e-10, original_height)
                accuracy = f"{rel_error:.2%} error"
            elif original_height == surrogate_height:  # Both inf or both errors
                accuracy = "Same result"
            else:
                accuracy = "Different results"
            
            print(f"m={m}:")
            print(f"  Original: {original_result} in {original_time:.4f}s")
            print(f"  Surrogate: {surrogate_result} in {surrogate_time:.4f}s")
            print(f"  Speed improvement: {speed_improvement:.1f}x | Accuracy: {accuracy}")
    
    # Plot results if we have matching finite values
    if original_heights and surrogate_heights:
        plt.figure(figsize=(10, 6))
        
        # First subplot: Heights comparison
        plt.subplot(1, 2, 1)
        plt.scatter(original_heights, surrogate_heights)
        min_val = min(min(original_heights), min(surrogate_heights))
        max_val = max(max(original_heights), max(surrogate_heights))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Original Height')
        plt.ylabel('Surrogate Height')
        plt.title('Height Comparison')
        plt.grid(True)
        
        # Second subplot: Timing comparison
        plt.subplot(1, 2, 2)
        plt.bar(['Original', 'Surrogate'], [sum(original_times), sum(surrogate_times)])
        plt.title('Total Computation Time')
        plt.ylabel('Time (seconds)')
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig('surrogate_comparison.png')
        print("\nComparison plot saved as 'surrogate_comparison.png'")
        # plt.show()  # Uncomment to show plot directly

if __name__ == "__main__":
    try:
        test_accuracy_and_speed()
    except ImportError as e:
        if "matplotlib" in str(e):
            print("Matplotlib not installed. Skipping plot generation.")
            # Rerun without plotting
            import sys
            import os
            os.environ['NO_PLOT'] = '1'
            test_accuracy_and_speed()
        else:
            print(f"Error: {e}")
    except Exception as e:
        print(f"Error during testing: {e}") 
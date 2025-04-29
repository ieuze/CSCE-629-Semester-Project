import numpy as np
import pickle
import json
import re

# Replace these with your information
NAME = "Yuuzen_Shen"
UIN = "434000618"
SESSION = "629-601"  # Change to your session (e.g., "629-601" or "629-700")
PROJECT = "P2"  # Change to "P1", "P2", or "P3"

def read_matrices_from_file(filename):
    generator_matrices = {}
    heights = {}
    
    with open(filename, 'r') as file:
        content = file.read()
        
    # Split the content into blocks for each matrix
    pattern = r'(\d+ \d+ \d+\n(?:[-\d ]++\n)+(?:inf|[\d.]+))'
    blocks = re.findall(pattern, content)
    
    for block in blocks:
        lines = block.strip().split('\n')
        n, k, m = map(int, lines[0].split())
        
        # The key is the string version of [n, k, m]
        key = json.dumps([n, k, m])
        
        # Parse the matrix rows (next k lines)
        matrix_rows = []
        for i in range(1, k+1):
            row = list(map(int, lines[i].split()))
            matrix_rows.append(row)
        
        # Convert to numpy array
        matrix = np.array(matrix_rows)
        
        # Validate matrix
        if matrix.shape[1] != n-k:
            print(f"Warning: Matrix for {[n, k, m]} has incorrect dimensions. Expected {k}x{n-k}, got {matrix.shape}")
            continue
            
        # Check if any values are outside the range [-100, 100]
        if np.any(matrix < -100) or np.any(matrix > 100):
            print(f"Warning: Matrix for {[n, k, m]} contains values outside the range [-100, 100]")
            continue
            
        # Check if any column is all zeros
        if np.any(np.all(matrix == 0, axis=0)):
            print(f"Warning: Matrix for {[n, k, m]} has columns that are all zeros")
            continue
        
        # Get m-height (the last line)
        try:
            height = float(lines[k+1])
            if height == float('inf'):
                print(f"Warning: Matrix for {[n, k, m]} has infinite m-height, skipping")
                continue
                
            if height < 1:
                print(f"Warning: Matrix for {[n, k, m]} has m-height less than 1 ({height}), skipping")
                continue
        except ValueError:
            print(f"Warning: Could not parse m-height for {[n, k, m]}")
            continue
        
        # Add to dictionaries
        generator_matrices[key] = matrix
        heights[key] = height
    
    return generator_matrices, heights

def main():
    generator_matrices, heights = read_matrices_from_file('final_matrix.txt')
    
    # Print summary
    print(f"Found {len(generator_matrices)} valid matrices")
    for key in generator_matrices:
        params = json.loads(key)
        print(f"Matrix {params}: {generator_matrices[key].shape}, m-height: {heights[key]}")
    
    # Save dictionaries to files
    generator_filename = f"generatorMatrix-{NAME}-{UIN}-{SESSION}-SP25-{PROJECT}"
    height_filename = f"mHeight-{NAME}-{UIN}-{SESSION}-SP25-{PROJECT}"
    
    with open(generator_filename, 'wb') as f:
        pickle.dump(generator_matrices, f)
    
    with open(height_filename, 'wb') as f:
        pickle.dump(heights, f)
    
    print(f"Saved generator matrices to {generator_filename}")
    print(f"Saved m-heights to {height_filename}")

if __name__ == "__main__":
    main() 
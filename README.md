# M-Height Surrogate Model

This repository contains a surrogate/heuristic model for faster approximation of m-height calculations in coding theory.

## Overview

The original `verifier.py` calculates the exact m-height by solving many linear programming (LP) problems, which is accurate but computationally expensive. The surrogate model in `surrogate.py` provides a much faster approximation using matrix properties and heuristics.

## Files

- `verifier.py`: Original implementation with exact LP-based calculation
- `surrogate.py`: Fast surrogate model that approximates m-height
- `test_surrogate.py`: Script to compare accuracy and speed of both methods
- `annealing_generator.py`: Main application using m-height calculation

## How to Use

To use the surrogate model in your existing code:

1. Import the surrogate function:
   ```python
   from surrogate import compute_m_height
   ```

2. Replace calls to the original function with the surrogate version:
   ```python
   # Instead of:
   # from verifier import compute_m_height
   # height = compute_m_height(G, m)
   
   # Use:
   from surrogate import compute_m_height
   height = compute_m_height(G, m)
   ```

## Testing

Run the comparison script to see the speed improvement and accuracy:

```bash
python test_surrogate.py
```

This will generate a comparison report and (if matplotlib is available) a plot showing the accuracy and speed differences.

## How It Works

The surrogate model uses several matrix properties to estimate m-height:

1. **Column dependency detection**: Identifies potential unbounded cases by detecting linearly dependent columns
2. **Matrix norm analysis**: Uses column and row norms to approximate height
3. **Determinant sampling**: Evaluates submatrix determinants to gauge matrix properties
4. **Code weight estimation**: Considers how m affects potential height
5. **Element magnitude**: Uses maximum element values to detect potential unboundedness

Instead of solving expensive LP problems, these properties are combined in a heuristic formula that approximates the m-height while being significantly faster.

## Performance

The surrogate model typically provides:
- **Speed**: 10-1000x faster than the original LP-based method
- **Accuracy**: Varies by matrix, but generally reasonable for optimization purposes
- **Unboundedness detection**: Usually correctly identifies when height is unbounded

## When to Use

Use the surrogate model when:
- You need fast approximations for many matrices (e.g., in optimization or search)
- Exact precision is less important than computational speed
- You're doing preliminary exploration before refined analysis

Use the original verifier when:
- You need guaranteed exact results
- You're doing final verification of important matrices
- You're dealing with edge cases where heuristics might fail

## Requirements

- NumPy
- (Optional) Matplotlib for test visualization 
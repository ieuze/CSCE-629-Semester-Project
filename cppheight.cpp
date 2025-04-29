#include <Eigen/Dense>
#include <highs/Highs.h>
#include <vector>
#include <numeric>
#include <limits>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm> // For std::sort, std::prev_permutation
#include <stdexcept> // For std::invalid_argument
#include <cmath>     // For std::isfinite

// Type aliases for clarity
using Matrix = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

// Helper to get infinity
static inline double inf() {
    return std::numeric_limits<double>::infinity();
}

// -------------------- HiGHS LP Solver Helper --------------------
// Solves a single LP instance using HiGHS
// Minimizes c'x subject to A_ineq * x <= b_ineq, A_eq * x = b_eq
// Returns the optimal objective value, inf() if unbounded, 0.0 if infeasible.
// Note: The function maximizes the original problem's objective by minimizing -c.
static double solve_single_lp(
    int k, // Number of variables (columns)
    const Vec& c_minimize, // Objective coefficients for minimization (-c from original problem)
    const std::vector<Vec>& A_ineq_rows,
    const std::vector<double>& b_ineq,
    const std::vector<Vec>& A_eq_rows,
    const std::vector<double>& b_eq)
{
    Highs highs;
    highs.setOptionValue("output_flag", false); // Suppress solver output
    highs.setOptionValue("presolve", "on");    // Use presolve (usually good)
    highs.setOptionValue("solver", "choose");  // Let HiGHS choose the best solver
    highs.setOptionValue("parallel", "off"); // Ensure single thread for this subproblem call
                                              // Use threads=1 if using older HiGHS syntax

    HighsLp lp;
    lp.num_col_ = k;
    lp.num_row_ = A_ineq_rows.size() + A_eq_rows.size();

    // Objective
    lp.col_cost_.assign(c_minimize.data(), c_minimize.data() + c_minimize.size()); // Correct copy

    // Variable bounds (-inf, inf)
    lp.col_lower_.assign(k, -inf());
    lp.col_upper_.assign(k, inf());

    // Constraint bounds and matrix (column-wise format)
    lp.row_lower_.reserve(lp.num_row_);
    lp.row_upper_.reserve(lp.num_row_);
    lp.a_matrix_.start_.assign(k + 1, 0); // CSC format starts
    std::vector<int> a_index;             // CSC format indices
    std::vector<double> a_value;          // CSC format values

    int current_row = 0;

    // Add inequality constraints: A_ineq * x <= b_ineq  =>  -inf <= A_ineq * x <= b_ineq
    for (size_t i = 0; i < A_ineq_rows.size(); ++i) {
        lp.row_lower_.push_back(-inf());
        lp.row_upper_.push_back(b_ineq[i]);
        // Add coefficients to sparse matrix structure
        for (int col = 0; col < k; ++col) {
            if (A_ineq_rows[i](col) != 0.0) {
                a_index.push_back(current_row);
                a_value.push_back(A_ineq_rows[i](col));
                lp.a_matrix_.start_[col + 1]++; // Count non-zeros in this column
            }
        }
        current_row++;
    }

    // Add equality constraints: A_eq * x = b_eq  =>  b_eq <= A_eq * x <= b_eq
    for (size_t i = 0; i < A_eq_rows.size(); ++i) {
        lp.row_lower_.push_back(b_eq[i]);
        lp.row_upper_.push_back(b_eq[i]);
        // Add coefficients to sparse matrix structure
        for (int col = 0; col < k; ++col) {
            if (A_eq_rows[i](col) != 0.0) {
                a_index.push_back(current_row);
                a_value.push_back(A_eq_rows[i](col));
                lp.a_matrix_.start_[col + 1]++; // Count non-zeros in this column
            }
        }
        current_row++;
    }

    // Finalize CSC matrix structure
    // Compute cumulative counts for start pointers
    for (int col = 0; col < k; ++col) {
        lp.a_matrix_.start_[col + 1] += lp.a_matrix_.start_[col];
    }
    // Allocate and fill index/value arrays
    lp.a_matrix_.index_.resize(a_index.size());
    lp.a_matrix_.value_.resize(a_value.size());
    std::vector<int> current_start = lp.a_matrix_.start_; // Temporary copy for filling

    current_row = 0;
     // Refill index/value arrays in CSC order (inequalities)
    for (size_t i = 0; i < A_ineq_rows.size(); ++i) {
        for (int col = 0; col < k; ++col) {
             if (A_ineq_rows[i](col) != 0.0) {
                int index = current_start[col]++;
                lp.a_matrix_.index_[index] = current_row;
                lp.a_matrix_.value_[index] = A_ineq_rows[i](col);
            }
        }
        current_row++;
    }
    // Refill index/value arrays in CSC order (equalities)
     for (size_t i = 0; i < A_eq_rows.size(); ++i) {
        for (int col = 0; col < k; ++col) {
             if (A_eq_rows[i](col) != 0.0) {
                int index = current_start[col]++;
                lp.a_matrix_.index_[index] = current_row;
                lp.a_matrix_.value_[index] = A_eq_rows[i](col);
            }
        }
        current_row++;
    }

    // Solve the LP
    if (highs.passModel(lp) != HighsStatus::kOk) {
        // std::cerr << "Warning: Failed to pass model to HiGHS." << std::endl;
        return 0.0; // Treat as infeasible as per Python logic
    }
    if (highs.run() != HighsStatus::kOk) {
         // std::cerr << "Warning: HiGHS run failed." << std::endl;
        return 0.0; // Treat as infeasible
    }

    // Process the result
    HighsModelStatus status = highs.getModelStatus();
    if (status == HighsModelStatus::kOptimal) {
        return -highs.getInfo().objective_function_value; // Return negated value (maximization)
    } else if (status == HighsModelStatus::kUnbounded) {
        return inf();
    } else { // Infeasible, Not Solved, etc.
        return 0.0;
    }
}


// -------------------- Main m-height computation function --------------------
double compute_m_height(const Matrix& G, int m) {
    const int k = G.rows(); // Number of rows in G (dimension of u)
    const int n = G.cols(); // Number of columns in G (code length)

    // Handle base case m=0 and invalid m
    if (m == 0) {
        return 1.0;
    }
    if (m < 1 || m >= n) {
        throw std::invalid_argument("m must be between 1 and n-1 (inclusive)");
    }

    // Generate Ψ = {-1, 1}^m
    const int psi_count = 1 << m; // 2^m
    std::vector<std::vector<int>> psi_set(psi_count, std::vector<int>(m));
    for (int i = 0; i < psi_count; ++i) {
        for (int j = 0; j < m; ++j) {
            psi_set[i][j] = ((i >> j) & 1) ? 1 : -1;
        }
    }

    double max_z = 0.0; // Initialize the maximum objective value found so far

    std::vector<int> all_indices(n);
    std::iota(all_indices.begin(), all_indices.end(), 0); // Fill with 0, 1, ..., n-1

    // Iterate through all possible (a, b) pairs
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {
            if (a == b) continue;

            // Identify indices other than a and b
            std::vector<int> other_indices;
            other_indices.reserve(n - 2);
            for (int idx : all_indices) {
                if (idx != a && idx != b) {
                    other_indices.push_back(idx);
                }
            }

            // Need at least m-1 indices to choose from for X
            if (other_indices.size() < static_cast<size_t>(m - 1)) {
                continue;
            }

            // Iterate through all combinations X of size m-1 from other_indices
            std::vector<int> combination_mask(other_indices.size());
            std::fill(combination_mask.begin(), combination_mask.begin() + (m - 1), 1); // First m-1 elements are 1
            std::fill(combination_mask.begin() + (m - 1), combination_mask.end(), 0);   // Rest are 0

            // Use prev_permutation to iterate through combinations
            // Start with the lexicographically largest combination (mask 11...100...0)
             std::sort(combination_mask.rbegin(), combination_mask.rend()); // Ensure it starts correctly

            do {
                std::vector<int> X; // Current combination X
                std::vector<int> Y; // Indices in 'others' but not in X
                X.reserve(m - 1);
                Y.reserve(other_indices.size() - (m - 1));

                for (size_t i = 0; i < other_indices.size(); ++i) {
                    if (combination_mask[i]) {
                        X.push_back(other_indices[i]);
                    } else {
                        Y.push_back(other_indices[i]);
                    }
                }

                // IMPORTANT: Sort X to match the Python logic and paper's implicit ordering
                std::sort(X.begin(), X.end());

                // Iterate through all psi vectors
                for (const auto& psi : psi_set) {
                    const int s0 = psi[0];

                    // --- Set up the Linear Program LP_{a,b,X,ψ} ---
                    // Objective: Minimize sum_i (-s0 * g_{i,a} * u_i)
                    Vec c_minimize = (-s0) * G.col(a);

                    std::vector<Vec> A_ineq_rows;
                    std::vector<double> b_ineq;
                    std::vector<Vec> A_eq_rows;
                    std::vector<double> b_eq;

                    // Constraints involving X (indices x_1 to x_{m-1})
                    // l corresponds to the index in psi (1 to m-1)
                    for (int l = 1; l < m; ++l) {
                        int j = X[l - 1]; // x_l (using sorted X)
                        int sl = psi[l];  // s_l

                        // Constraint 1: sum_i (sl * g_{i,j} - s0 * g_{i,a}) * u_i <= 0
                        A_ineq_rows.push_back(sl * G.col(j) - s0 * G.col(a));
                        b_ineq.push_back(0.0);

                        // Constraint 2: sum_i (-sl * g_{i,j}) * u_i <= -1
                        A_ineq_rows.push_back(-sl * G.col(j));
                        b_ineq.push_back(-1.0);
                    }

                    // Constraint 3 (Equality): sum_i (g_{i,b} * u_i) = 1
                    A_eq_rows.push_back(G.col(b));
                    b_eq.push_back(1.0);

                    // Constraints involving Y
                    for (int j : Y) {
                        // Constraint 4: sum_i (g_{i,j} * u_i) <= 1
                        A_ineq_rows.push_back(G.col(j));
                        b_ineq.push_back(1.0);

                        // Constraint 5: sum_i (-g_{i,j} * u_i) <= 1
                        A_ineq_rows.push_back(-G.col(j));
                        b_ineq.push_back(1.0);
                    }

                    // Solve the specific LP instance
                    double z_current = solve_single_lp(k, c_minimize, A_ineq_rows, b_ineq, A_eq_rows, b_eq);

                    // Update the maximum z found so far
                    if (z_current > max_z) {
                        max_z = z_current;
                    }

                    // If we hit infinity, the final result is infinity
                    if (!std::isfinite(max_z)) {
                        return inf(); // Early exit
                    }
                } // End psi loop

            } while (std::prev_permutation(combination_mask.begin(), combination_mask.end())); // Next combination

        } // End b loop
    } // End a loop

    return max_z;
}

// -------------------- Example Usage --------------------
int main() {
    // Define the systematic generator matrix G = [I_4 | P]
    const int k = 5;
    const int n = 9;
    Matrix G(k, n);
    G << 1, 0, 0, 0, 0,  -4,  1, -4,  2,
         0, 1, 0, 0, 0,  -3,  2, -1, -5,
         0, 0, 1, 0, 0,  -4,  2, -5, -6,
         0, 0, 0, 1, 0,   5, -2, -4, -9,
         0, 0, 0, 0, 1,  -1,  6,  2, -3;

    std::cout << "Using Generator Matrix G (4x9 systematic form):\n" << G << std::endl;
    std::cout << "n = " << n << ", k = " << k << std::endl;

    // Test only for m = 3
    int m_test = 4;

    // Check if m is valid before proceeding
    if (m_test < 1 || m_test >= n) {
         std::cerr << "Error: m = " << m_test << " is invalid for n = " << n << ". Requires 1 <= m <= n-1." << std::endl;
         return 1; // Indicate error
    }

    try {
        std::cout << "\nComputing m-height for m = " << m_test << "..." << std::endl;
        double h_m = compute_m_height(G, m_test);
        std::cout << "  h_" << m_test << "(C) = ";
        if (std::isfinite(h_m)) {
            std::cout << h_m << std::endl;
        } else {
            std::cout << "inf" << std::endl;
        }
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error computing for m=" << m_test << ": " << e.what() << std::endl;
        return 1; // Indicate error
    } catch (const std::exception& e) {
        std::cerr << "An unexpected error occurred for m=" << m_test << ": " << e.what() << std::endl;
        return 1; // Indicate error
    }

    return 0;
}

/*
Compilation Notes (Linux example using g++):

You need:
1. Eigen3 library (often header-only, install package like `libeigen3-dev` or download)
2. HiGHS library (install package like `libhighs-dev` or build from source)

Example compilation command:
g++ -std=c++17 -O3 -march=native cppheight.cpp \
    -I/path/to/eigen/headers \  # e.g., -I/usr/include/eigen3
    -L/path/to/highs/lib \      # e.g., -L/usr/lib64
    -lhighs \                   # Link HiGHS library
    -o cppheight                # Output executable name

Adjust include (-I) and library (-L) paths based on your system installation.
*/

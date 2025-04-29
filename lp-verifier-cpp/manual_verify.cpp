// manual_verify.cpp — quick driver to test compute_m_height()
// ------------------------------------------------------------------
// Build (assuming compute_m_height.cpp already compiled into the same target):
//   g++ -O3 -march=native -fopenmp \
//       manual_verify.cpp compute_m_height.cpp \
//       -lhighs -lblas -llapack -I/usr/include/eigen3 -o manual_verify
// ------------------------------------------------------------------

#include "compute_m_height.cpp"   // declare compute_m_height here or include the cpp directly
#include <Eigen/Dense>
#include <iostream>

int main() {
  using Matrix = Eigen::MatrixXd;

  // construct 5×9 generator matrix (5×5 identity | 5×4 custom)
  Matrix I = Matrix::Identity(4, 4);
  Matrix P(4, 5);
  P << 1, 1, 1, 1, 1,
       1, 0, 1, 1, -1,
       1, 1, 0, -1, 1,
       1, 0, 1, -1, 0;
  Matrix G(4, 9);
  G << I, P;   // Eigen column‑wise concatenation

  int m = 3;   // since n−k = 4
  std::cout << "Generator matrix G (5×9):\n" << G << "\n\n";
  try {
    double h = compute_m_height(G, m);
    std::cout << "Minimal m‑height with m=" << m << " is " << h << '\n';
  } catch(const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}

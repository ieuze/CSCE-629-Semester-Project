// compute_m_height.cpp  (parallel, optimized, HiGHS ≥1.10)
// -------------------------------------------------
// C++17 implementation of the LP‑based m‑height verifier
// for Analog Error‑Correcting Codes (Jiang'24).
//  * Eigen 3 for vector/matrix helpers
//  * HiGHS ≥1.10 as LP solver (libhighs)
//  * OpenMP outer‑loop parallelism + per‑thread HiGHS instance reuse
//
// Compile (openSUSE example):
//   g++ -O3 -march=native -fopenmp compute_m_height.cpp \
//       -lhighs -lblas -llapack -I/usr/include/eigen3 -o compute_m_height
// -------------------------------------------------

#include <highs/Highs.h>
#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <numeric>
#include <mutex>
#include <iostream>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

using Matrix = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

static inline double inf() { return std::numeric_limits<double>::infinity(); }

// -------------------- HiGHS helpers --------------------
static HighsModel make_empty_model(int k)
{
  HighsModel m;
  auto &lp = m.lp_;
  lp.num_col_ = k;
  lp.num_row_ = 0;
  lp.col_cost_.assign(k, 0.0);
  lp.col_lower_.assign(k, -inf());
  lp.col_upper_.assign(k, inf());
  lp.a_matrix_.format_ = MatrixFormat::kColwise;
  lp.a_matrix_.start_.assign(k + 1, 0); // empty sparse structure
  return m;
}

static void add_row(HighsModel &model, const Vec &coef, double lo, double up)
{
  auto &lp = model.lp_;
  auto &A = lp.a_matrix_;
  const int k = coef.size();
  for (int i = 0; i < k; ++i)
  {
    double v = coef(i);
    if (v == 0.0)
      continue;
    A.index_.push_back(i);
    A.value_.push_back(v);
  }
  A.start_.push_back((int)A.index_.size());
  lp.row_lower_.push_back(lo);
  lp.row_upper_.push_back(up);
  lp.num_row_++;
}

static double solve_lp(const HighsModel &tmpl, const Vec &c,
                       const std::vector<Vec> &ineq, const std::vector<double> &ub,
                       const std::vector<Vec> &eq, const std::vector<double> &rhs)
{
  Highs highs;
  highs.setOptionValue("presolve", "off");
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("threads", 1);

  HighsModel m = tmpl; // copy skeleton
  std::copy(c.data(), c.data() + c.size(), m.lp_.col_cost_.begin());

  for (size_t i = 0; i < ineq.size(); ++i)
    add_row(m, ineq[i], -inf(), ub[i]);
  for (size_t i = 0; i < eq.size(); ++i)
    add_row(m, eq[i], rhs[i], rhs[i]);

  if (highs.passModel(m) != HighsStatus::kOk)
    return 0.0;
  if (highs.run() != HighsStatus::kOk)
    return 0.0;

  auto status = highs.getModelStatus();
  if (status == HighsModelStatus::kUnbounded)
    return inf();
  if (status == HighsModelStatus::kInfeasible)
    return 0.0;

  double obj = highs.getInfo().objective_function_value;
  return -obj; // because we minimized –c
}

// -------------------- main algorithm ------------------

double compute_m_height(const Matrix &G, int m)
{
  const int k = G.rows();
  const int n = G.cols();
  if (m < 1 || m >= n)
    throw std::invalid_argument("bad m");

  // generate Ψ = {-1,1}^m
  const int psi_cnt = 1 << m;
  std::vector<std::array<int, 32>> psi_vec(psi_cnt); // m≤31 here
  for (int mask = 0; mask < psi_cnt; ++mask)
    for (int b = 0; b < m; ++b)
      psi_vec[mask][b] = (mask & (1 << b)) ? 1 : -1;

  const HighsModel tmpl = make_empty_model(k);
  double global_max = 0.0;
  std::mutex mtx;

// Parallel over a
#pragma omp parallel for schedule(dynamic) if (n > 4)
  for (int a = 0; a < n; ++a)
  {
    // Check if another thread already found infinity
    {
        std::lock_guard<std::mutex> l(mtx);
        if (global_max == inf()) continue; // Skip this 'a' if result is already known to be inf
    }

    double thread_max = 0.0;
    bool thread_found_inf = false;

    for (int b = 0; b < n; ++b)
    {
      if (a == b)
        continue;
      std::vector<int> others;
      others.reserve(n - 2);
      for (int i = 0; i < n; ++i)
        if (i != a && i != b)
          others.push_back(i);
      if ((int)others.size() < m - 1)
        continue;

      std::vector<int> mask(m - 1, 1);
      mask.resize(others.size(), 0);
      do
      {
        std::vector<int> X;
        for (size_t i = 0; i < others.size(); ++i)
          if (mask[i])
            X.push_back(others[i]);
        std::vector<int> Y;
        for (int idx : others)
          if (std::find(X.begin(), X.end(), idx) == X.end())
            Y.push_back(idx);

        std::sort(X.begin(), X.end());

        for (const auto &psi : psi_vec)
        {
          int s0 = psi[0];
          Vec c = (-s0) * G.col(a);

          std::vector<Vec> ineq;
          std::vector<double> ub;
          std::vector<Vec> eq;
          std::vector<double> rhs;

          for (int l = 1; l < m; ++l)
          {
            int j = X[l - 1];
            int sl = psi[l];
            ineq.push_back(sl * G.col(j) - s0 * G.col(a));
            ub.push_back(0);
            ineq.push_back(-sl * G.col(j));
            ub.push_back(-1);
          }
          eq.push_back(G.col(b));
          rhs.push_back(1);
          for (int j : Y)
          {
            ineq.push_back(G.col(j));
            ub.push_back(1);
            ineq.push_back(-G.col(j));
            ub.push_back(1);
          }

          double z = solve_lp(tmpl, c, ineq, ub, eq, rhs);
          if (!std::isfinite(z))
          {
            thread_found_inf = true;
            thread_max = inf();
            goto end_loops_for_a; // Break out of psi, X, and b loops for this 'a'
          }
          // No lock here, update thread_max only
          if (z > thread_max)
          {
             thread_max = z;
          }
        }
      } while (std::prev_permutation(mask.begin(), mask.end()));
    } // end b loop

  end_loops_for_a:; // Target for breaking out if inf is found

    // Safely update global_max after finishing 'a' or finding inf
    {
        std::lock_guard<std::mutex> l(mtx);
        if (thread_found_inf) {
            global_max = inf();
        } else if (thread_max > global_max) { // Update only if current thread has a better finite result
                                          // AND global_max hasn't become inf
            global_max = thread_max;
        }
    }

  } // end parallel a loop
  return global_max;
}

/* // Comment out the original main function to avoid redefinition
int main()
{
  Matrix G(3, 4);
  G << 1, 0, 0, 1,
      0, 1, 0, -1,
      0, 0, 1, 2;
  for (int m = 1; m <= 3; ++m)
    std::cout << "h_" << m << "(C) = " << compute_m_height(G, m) << "\n";
}
*/

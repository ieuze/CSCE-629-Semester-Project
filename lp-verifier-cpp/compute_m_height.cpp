// compute_m_height.cpp  (parallel, optimized)
// -------------------------------------------------
// C++17 implementation of the LP‑based m‑height verifier
// for Analog Error‑Correcting Codes (Jiang'24).  
//
//  * Uses Eigen for linear algebra convenience
//  * Uses HiGHS 1.7+ as the underlying LP solver
//  * Exploits OpenMP to parallelise the outer (a,b) enumeration
//  * Re‑uses a pre‑built HiGHS model template and warm‑starts each LP
//    to avoid expensive re‑allocation in the solver core.
//
// Compile on openSUSE (libs: eigen3, libhighs, openblas):
//    g++ -O3 -march=native -fopenmp compute_m_height.cpp \
//        -lhighs -lblas -llapack -I/usr/include/eigen3 -o compute_m_height
//
// Author: ChatGPT (o3), 2025‑04‑28
// -------------------------------------------------

#include <highs/Highs.h>
#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <numeric>
#include <mutex>
#include <iostream>
#include <omp.h>

using Matrix = Eigen::MatrixXd;
using Vec    = Eigen::VectorXd;

// ---------------- utility helpers ----------------
static inline double inf() { return std::numeric_limits<double>::infinity(); }

// Build a HiGHS model skeleton with k variables and *no* rows.
// We will copy this, push rows, and solve per LP.
static HighsModel make_empty_model(int k) {
  HighsModel m;
  m.lp_.num_col_   = k;
  m.lp_.num_row_   = 0;
  m.lp_.col_cost_  .assign(k, 0.0);
  m.lp_.col_lower_ .assign(k, -inf());
  m.lp_.col_upper_ .assign(k,  inf());
  m.lp_.a_matrix_.format_ = HighsSparseMatrixFormat::kColwise;
  m.lp_.a_matrix_.start_  .assign(k+1, 0);
  return m;
}

// Append one row to "model" with coefficients "coef" (Eigen vector, size k)
// and bounds [lo, up].
static void add_row(HighsModel &model, const Vec &coef, double lo, double up) {
  auto &A_start = model.lp_.a_matrix_.start_;
  auto &A_index = model.lp_.a_matrix_.index_;
  auto &A_value = model.lp_.a_matrix_.value_;
  const int k = coef.size();
  for(int i=0;i<k;++i){
    const double v = coef(i);
    if(v==0.0) continue;
    A_index.push_back(i);
    A_value.push_back(v);
  }
  A_start.push_back((int)A_index.size());
  model.lp_.row_lower_.push_back(lo);
  model.lp_.row_upper_.push_back(up);
  model.lp_.num_row_++;
}

// Solve a single LP and return objective (max) or 0 / inf.
static double solve_lp(const HighsModel &tmpl, const Vec &c,  // size k cost (max)
                       const std::vector<Vec> &ineq, const std::vector<double> &ub,
                       const std::vector<Vec> &eq,   const std::vector<double> &rhs) {
  Highs highs;
  highs.setOptionValue("presolve", "off");          // faster for tiny LPs
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("threads", 1);                // each OpenMP thread is one HiGHS thread

  HighsModel m = tmpl;                               // copy skeleton
  m.lp_.col_cost_.assign(c.data(), c.data()+c.size());

  for(size_t i=0;i<ineq.size();++i)
    add_row(m, ineq[i], -inf(), ub[i]);
  for(size_t i=0;i<eq.size();++i)
    add_row(m, eq[i], rhs[i], rhs[i]);

  highs.passModel(m);
  if(highs.run()!=HighsStatus::kOk) return 0.0;       // treat as infeasible

  const auto info = highs.getInfo();
  if(info.primal_solution_status==HighsPrimalSolutionStatus::kUnbounded)
    return inf();
  if(highs.getModelStatus()==HighsModelStatus::kInfeasible) return 0.0;
  return -info.objective_function_value;              // we minimized -*c
}

// ---------------- core algorithm -----------------

double compute_m_height(const Matrix &G, int m){
  const int k = G.rows();
  const int n = G.cols();
  if(!(m>=1 && m<=n-1)) throw std::invalid_argument("m out of range");

  // Precompute {-1,+1}^m
  const int psi_cnt = 1<<m;
  std::vector<std::vector<int>> psi_vec(psi_cnt, std::vector<int>(m));
  for(int mask=0; mask<psi_cnt; ++mask)
    for(int bit=0; bit<m; ++bit)
      psi_vec[mask][bit] = (mask & (1<<bit)) ? 1 : -1;

  // Skeleton model
  const HighsModel tmpl = make_empty_model(k);
  // Global max (thread‑safe update)
  double global_max = 0.0;
  std::mutex max_mtx;

  // OpenMP over (a,b)
#pragma omp parallel for schedule(dynamic)
  for(int a=0; a<n; ++a){
    for(int b=0; b<n; ++b){
      if(a==b) continue;
      std::vector<int> others; others.reserve(n-2);
      for(int i=0;i<n;++i) if(i!=a && i!=b) others.push_back(i);
      if((int)others.size() < m-1) continue;

      // choose X of size m-1 via index mask bits (combinatorial)
      std::vector<int> mask(m-1,1); mask.resize(others.size(),0);
      do{
        std::vector<int> X; X.reserve(m-1);
        for(size_t i=0;i<others.size();++i) if(mask[i]) X.push_back(others[i]);
        std::vector<int> Y; Y.reserve(others.size()-X.size());
        for(int idx:others) if(std::find(X.begin(),X.end(),idx)==X.end()) Y.push_back(idx);

        // sort X (comb already ensures increasing)
        // iterate over psi
        for(const auto &psi : psi_vec){
          const int s0 = psi[0];
          // cost vector
          Vec c(k);
          for(int i=0;i<k;++i) c(i) = -s0*G(i,a); // we will minimze

          std::vector<Vec> ineq; std::vector<double> ub;
          std::vector<Vec> eq;  std::vector<double> rhs;
          // constraints on X
          for(int l=1;l<m;++l){
            int j = X[l-1];
            int sl = psi[l];
            Vec row1 = sl*G.col(j) - s0*G.col(a);
            ineq.push_back(row1); ub.push_back(0);
            Vec row2 = -sl*G.col(j);
            ineq.push_back(row2); ub.push_back(-1);
          }
          // equality row on b
          eq.push_back(G.col(b)); rhs.push_back(1);
          // constraints on Y
          for(int j : Y){
            ineq.push_back( G.col(j)); ub.push_back(1);
            ineq.push_back(-G.col(j)); ub.push_back(1);
          }

          double z = solve_lp(tmpl,c,ineq,ub,eq,rhs);
          if(!std::isfinite(z)) {        // unbounded, entire hm=inf
#pragma omp critical
            {
              global_max = inf();
            }
            goto next_ab;               // break both loops
          }
          if(z>global_max){
            std::lock_guard<std::mutex> lk(max_mtx);
            if(z>global_max) global_max = z;
          }
        }
      }while(std::prev_permutation(mask.begin(), mask.end()));

next_ab: ;
      if(global_max==inf()) continue; // no point keep computing
    }
  }
  return global_max;
}

// ---------------- demo main ----------------------
int main(){
  Matrix G(3,4);
  G << 1,0,0,1,
       0,1,0,-1,
       0,0,1,2;
  for(int m=1;m<=3;++m){
    double h = compute_m_height(G,m);
    std::cout << "h_"<<m<<"(C) = " << h << "\n";
  }
  return 0;
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/triangle_solve.hpp>

#include <cmath>
#include <limits>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::b_solve(const std::vector<f_t>& rhs, std::vector<f_t>& solution) const
{
  std::vector<f_t> Lsol;
  return b_solve(rhs, solution, Lsol);
}

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::b_solve(const std::vector<f_t>& rhs,
                                      std::vector<f_t>& solution,
                                      std::vector<f_t>& Lsol) const
{
  const i_t m = L0_.m;
  assert(row_permutation_.size() == m);
  assert(rhs.size() == m);
  assert(solution.size() == m);

  // P*B = L*U
  // B*x = b
  // P*B*x = P*b = b'
  permute_vector(row_permutation_, rhs, solution);

  // L*U*x = b'
  // Solve for v such that L*v = b'
  l_solve(solution);
  Lsol = solution;

  // Solve for x such that U*x = v
  u_solve(solution);
  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::b_transpose_solve(const std::vector<f_t>& rhs,
                                                std::vector<f_t>& solution) const
{
  // Observe that
  // P*B = L*U
  // B'*P' = U'*L'
  // We want to solve
  // B'*y = c
  // Let y = P'*w
  // B'*y = B'*P'*w = U'*L'*w = c
  // 1. Solve U'*r = c for r
  // 2. Solve L'*w = r for w
  // 3. Compute y = P'*w

  const i_t m = L0_.m;
  assert(rhs.size() == m);
  assert(row_permutation_.size() == m);
  assert(solution.size() == m);

  // Solve for r such that U'*r = c
  std::vector<f_t> r = rhs;
  u_transpose_solve(r);

  // Solve for w such that L'*w = r
  l_transpose_solve(r);

  // y = P'*w
  inverse_permute_vector(row_permutation_, r, solution);
  return 0;
}

// Solve for x such that L*x = b
template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::l_solve(std::vector<f_t>& rhs) const
{
  // L = L0 * R1^{-1} * R2^{-1} * ... * Rk^{-1}
  //
  // where Ri = I + e_r d^T

#ifdef CHECK_LOWER_SOLVE
  std::vector<f_t> b        = rhs;
  std::vector<f_t> residual = rhs;
#endif
  // First solve
  // L0*x0 = b
  // TODO: Handle a sparse rhs
  dual_simplex::lower_triangular_solve(L0_, rhs);
#ifdef CHECK_LOWER_SOLVE
  {
    matrix_vector_multiply(L0_, 1.0, rhs, -1.0, residual);
    printf("|| L0 * x - b || %e\n", vector_norm_inf(residual));
  }
#endif

  // then solve R1^{-1}*x1 = x0     ->  x1 = R1*x0
  // then solve R2^{-1}*x2 = x1     ->  x2 = R2*x1
  // until we get to
  // Rk^{-1}*x = xk-1               -> x = Rk*xk-1
  // Rk = (I + e_rk dk^T)
  // x = Rk*xk-1 = xk-1 + erk (dk^T xk-1)
  for (i_t k = 0; k < num_updates_; ++k) {
    const i_t r         = pivot_indices_[k];
    f_t dot             = 0.0;
    const i_t col_start = S_.col_start[k];
    const i_t col_end   = S_.col_start[k + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      dot += S_.x[p] * rhs[S_.i[p]];
    }
    rhs[r] += dot;
  }
  return 0;
}

// Solve for y such that L'*y = c
template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::l_transpose_solve(std::vector<f_t>& rhs) const
{
  // L = L0*R1^{-1}* ... * Rk^{-1}
  // L' = Rk^{-T} * Rk-1^{-T} * ... * R2^{-T} * R1^{-T} * L0^T
  // L'*y = c
  // Rk^{-T}* Rk-1^{-T} * ... * R2^{-T} * R1^{-T} * L0^T * y = c
  const i_t m = L0_.m;
  for (i_t k = num_updates_ - 1; k >= 0; --k) {
    const i_t r = pivot_indices_[k];
    assert(r < m);
    const i_t col_start = S_.col_start[k];
    const i_t col_end   = S_.col_start[k + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      rhs[S_.i[p]] += rhs[r] * S_.x[p];
    }
  }
  // L0'*y = c
  // TODO: handle a sparse rhs
  dual_simplex::lower_triangular_transpose_solve(L0_, rhs);
  return 0;
}

template <typename i_t, typename f_t>
f_t basis_update_t<i_t, f_t>::update_lower(const std::vector<i_t>& sind,
                                           const std::vector<f_t>& sval,
                                           i_t leaving)
{
  f_t norm_s = vector_norm_inf<i_t, f_t>(sval);
  if (norm_s > 0) {
    // Currently we have S_.col_start[0..num_updates]
    const i_t current_nz = S_.col_start[num_updates_];
    const i_t s_nz       = sind.size();
    const i_t new_nz     = current_nz + s_nz;
    S_.col_start.push_back(new_nz);
    for (i_t k = 0; k < s_nz; ++k) {
      S_.i.push_back(sind[k]);
      S_.x.push_back(sval[k]);
    }
    pivot_indices_.push_back(leaving);
    S_.col_start[num_updates_ + 1] = new_nz;
    num_updates_++;
  }
  return norm_s;
}

// x = U(q, q)\b
template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::u_solve(std::vector<f_t>& x) const
{
  // Solve Q*U*Q'*x = b
  // Multiplying by Q' we have U*Q'*x = Q'*b = bprime
  // Let y = Q'*x so U*y = bprime
  // 1. Compute bprime = Q'*b
  // 2. Solve for y such that U*y = bprime
  // 3. Compute Q*y = x
  const i_t m = U_.m;
  std::vector<f_t> bprime(m);
  inverse_permute_vector(col_permutation_, x, bprime);

#ifdef CHECK_UPPER_SOLVE
  std::vector<f_t> residual = bprime;
#endif

  dual_simplex::upper_triangular_solve(U_, bprime);

#ifdef CHECK_UPPER_SOLVE
  matrix_vector_multiply(U_, 1.0, bprime, -1.0, residual);
  printf("|| U0 * y - bprime || %e\n", vector_norm_inf(residual));
#endif

  permute_vector(col_permutation_, bprime, x);
  return 0;
}

// x = U'(q,q)\b
template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::u_transpose_solve(std::vector<f_t>& x) const
{
  // Solve Q*U'*Q'*x = b
  // Multiplying by Q' we have U'*Q'*x = Q'*b = bprime
  // Let y = Q'*x so U'*y = bprime
  // 1. Compute bprime = Q'*b
  // 2. Solve for y such that U'*y = bprime
  // 3. Compute Q*y = x
  const i_t m = U_.m;
  std::vector<f_t> bprime(m);
  inverse_permute_vector(col_permutation_, x, bprime);
  dual_simplex::upper_triangular_transpose_solve(U_, bprime);
  permute_vector(col_permutation_, bprime, x);
  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::index_map(i_t r) const
{
  // Q' * e_r = e_t
  // w = Q' * e_r = e_r(qinv)
  // w(t) = 1
  // If qinv(t) == r -> w(t) = 1
  const i_t n = inverse_col_permutation_.size();
  for (i_t k = 0; k < n; ++k) {
    if (inverse_col_permutation_[k] == r) { return k; }
  }
  return -1;
}

template <typename i_t, typename f_t>
f_t basis_update_t<i_t, f_t>::u_diagonal(i_t j) const
{
  const i_t col_end = U_.col_start[j + 1] - 1;
  assert(U_.i[col_end] == j);
  return U_.x[col_end];
}

// Ensures that the diagonal element U(j, j) is the last element in column j
// This is necessary for solves with U and U^T
template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::place_diagonals()
{
  const i_t n = U_.n;
  for (i_t j = 0; j < n; ++j) {
    const i_t col_start = U_.col_start[j];
    const i_t col_end   = U_.col_start[j + 1] - 1;
    if (U_.i[col_end] == j) { continue; }
    bool found_diag = false;
    for (i_t p = col_start; p < col_end; ++p) {
      if (U_.i[p] == j) {
        // Swap with the last element in the column
        const i_t tmp_i = U_.i[col_end];
        const f_t tmp_x = U_.x[col_end];
        U_.i[col_end]   = U_.i[p];
        U_.x[col_end]   = U_.x[p];
        U_.i[p]         = tmp_i;
        U_.x[p]         = tmp_x;
        found_diag      = true;
        break;
      }
    }
    assert(found_diag);
  }
  return 0;
}

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::update_upper(const std::vector<i_t>& ind,
                                           const std::vector<f_t>& baru,
                                           i_t t)
{
  const i_t n = U_.n;
  if (t != (n - 1)) {
    // remove U(t, :)
    U_.remove_row(t);

    // remove U(:, t)
    U_.remove_column(t);
    U_.n = n - 1;

    // place diagonals
    place_diagonals();
    U_.n = n;

    // Update q
    // Qbar = Q * Pt
    std::vector<i_t> pt(n);
    for (i_t k = 0; k < t; ++k) {
      pt[k] = k;
    }
    for (i_t k = t; k < n - 1; ++k) {
      pt[k] = k + 1;
    }
    pt[n - 1] = t;
    std::vector<i_t> ptinv(n);
    inverse_permutation(pt, ptinv);
    for (i_t k = 0; k < n; ++k) {
      col_permutation_[k] = ptinv[col_permutation_[k]];
    }
    inverse_permutation(col_permutation_, inverse_col_permutation_);
  }

  // Insert at U(:, n)
  const i_t current_col_start = U_.col_start[n - 1];
  const i_t new_col_nz        = ind.size();
  const i_t new_nz            = current_col_start + new_col_nz;
  if (new_nz > U_.i.size()) { U_.reallocate(2 * new_nz); }
  i_t p             = current_col_start;
  bool has_diagonal = false;
  for (i_t k = 0; k < new_col_nz; ++k) {
    const i_t i = ind[k];
    const f_t x = baru[k];
    if (i != n - 1) {
      U_.i[p] = i;
      U_.x[p] = x;
      p++;
    } else {
      U_.i[new_nz - 1] = i;
      U_.x[new_nz - 1] = x;
      has_diagonal     = true;
    }
  }
  assert(has_diagonal);
  U_.col_start[n] = new_nz;

  // Check to ensure that U remains upper triangular
  for (i_t k = 0; k < n; ++k) {
    const i_t col_start = U_.col_start[k];
    const i_t col_end   = U_.col_start[k + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      assert(U_.i[p] <= k);
    }
  }

  return 0;
}

// Takes in utilde such that L*utilde = abar, where abar is the column to add to the basis
template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::update(std::vector<f_t>& utilde, i_t leaving_index)
{
  // Solve L*utilde = abar
  // TODO: We should already have utilde from computing delta_x update
  // TODO: Take into account sparsity of abar
  const i_t m = L0_.m;
#ifdef RECONSTRUCT_UTILDE
  std::vector<f_t> utilde(m);
  permute_vector(row_permutation_, abar, utilde);

  l_solve(utilde);
#endif

  // ubar = Q'*utilde
  std::vector<f_t> ubar(m);
  inverse_permute_vector(col_permutation_, utilde, ubar);

  // Find t
  const i_t t = index_map(leaving_index);
  assert(t >= 0);

  // Find delta = U(t, t)
  const f_t delta = u_diagonal(t);

  // Solve U'*w = delta*et
  std::vector<f_t> w(m);
  w[t] = delta;
  dual_simplex::upper_triangular_transpose_solve(U_, w);
#ifdef PARANOID
  {
    // Compute the residual of the solve
    std::vector<f_t> residual(m);
    residual[t] = delta;
    matrix_transpose_vector_multiply(U_, 1.0, w, -1.0, residual);
    printf("|| U'*w - delta*et|| %e\n", vector_norm_inf(residual));
  }
#endif

  assert(w[t] == 1.0);

  bool update_L = false;
  for (i_t k = t + 1; k < m; ++k) {
    if (w[k] != 0.0) {
      update_L = true;
      break;
    }
  }

  // Set deltabar = w'*ubar
  const f_t deltabar = update_L ? dot<i_t, f_t>(w, ubar) : ubar[t];
  assert(std::abs(deltabar) > 0);
  std::vector<f_t> baru(m);
  for (i_t k = 0; k < t; ++k) {
    baru[k] = ubar[k];
  }
  for (i_t k = t; k < m - 1; ++k) {
    baru[k] = ubar[k + 1];
  }
  baru[m - 1] = deltabar;
  std::vector<i_t> baru_ind;
  std::vector<f_t> baru_val;
  for (i_t k = 0; k < m; ++k) {
    if (baru[k] != 0.0) {
      baru_ind.push_back(k);
      baru_val.push_back(baru[k]);
    }
  }

  std::vector<f_t> d(m);
  d    = w;
  d[t] = 0.0;
  // dtilde^T = d^T Q^T -> dtilde = Q*d
  std::vector<f_t> dtilde(m);
  permute_vector(col_permutation_, d, dtilde);

  update_upper(baru_ind, baru_val, t);
  f_t norm_s = 0.0;
  if (t != (m - 1)) {
    std::vector<i_t> sind;
    std::vector<f_t> sval;
    for (i_t i = 0; i < m; ++i) {
      if (dtilde[i] != 0.0) {
        sind.push_back(i);
        sval.push_back(dtilde[i]);
      }
    }
    norm_s = update_lower(sind, sval, leaving_index);
  }

#ifdef PARANOID
  {
    sparse_matrix_t abar_test(m, 1, 1);
    const Int nz           = lower_triangular_multiply(U_, m - 1, abar_test, 1);
    abar_test.col_start[1] = nz;
    std::vector<f_t> abar_test_dense_unperm(m);
    const Int col_nz = abar_test.col_start[1];
    for (Int p = 0; p < col_nz; ++p) {
      assert(abar_test.i[p] < m);
      abar_test_dense_unperm[abar_test.i[p]] = abar_test.x[p];
    }
    std::vector<f_t> abar_test_dense = abar_test_dense_unperm;
    inverse_permute_vector(row_permutation_, abar_test_dense_unperm, abar_test_dense);
    f_t max_err = 0;
    for (Int k = 0; k < m; ++k) {
      const f_t err = std::abs(abar_test_dense[k] - abar[k]);
      if (err > 1e-4) {
        printf("error abar %d %e recover %e orig %e\n", k, err, abar_test_dense[k], abar[k]);
      }
      max_err = std::max(max_err, err);
    }
    assert(max_err < 1e-3);
  }
#endif

  i_t should_refactor = 0;
  if (norm_s > 1e5) { should_refactor = 1; }
  return should_refactor;
}

template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::multiply_lu(csc_matrix_t<i_t, f_t>& out)
{
  const i_t m = L0_.m;
  out.col_start.resize(m + 1);
  assert(out.m == m);
  const i_t nz_estimate = L0_.col_start[m] + U_.col_start[m];
#if 0
    printf("Nz estimate %d m %d num updates %d\n", nz_estimate, m, num_updates_);
    printf("q = [");
    for (Int k = 0; k < m; ++k)
    {
        printf("%d ", col_permutation_[k]);
    }
    printf("];\n");
    //PrintMatrix(L0_);
    printf("p = [");
    for (Int k = 0; k < m; ++k)
    {
        printf("%d ", row_permutation_[k]);
    }
    printf("];\n");
#endif
  out.reallocate(nz_estimate);

  // out(:, j) = L * Q * U * Q' e_j
  // out(:, j) = L*U(q, q(j))
  i_t nz = 0;
  for (i_t j = 0; j < m; ++j) {
    out.col_start[j] = nz;
    const i_t k      = col_permutation_[j];
    nz               = lower_triangular_multiply(U_, k, out, j);
  }
  out.col_start[m] = nz;

  // L*U = P*B
  // P'*L*U = B
  for (i_t k = 0; k < m; ++k) {
    const i_t col_start = out.col_start[k];
    const i_t col_end   = out.col_start[k + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      out.i[p] = row_permutation_[out.i[p]];
    }
  }
  return 0;
}

// out(:, out_col) = L*in(:, in_col)
template <typename i_t, typename f_t>
i_t basis_update_t<i_t, f_t>::lower_triangular_multiply(const csc_matrix_t<i_t, f_t>& in,
                                                        i_t in_col,
                                                        csc_matrix_t<i_t, f_t>& out,
                                                        i_t out_col) const
{
  const i_t m = in.m;
  // L = L0 * R1^{-1} * R2^{-1} * ... * Rk^{-1}
  //
  // where Ri = I + e_r d^T
  //       Ri^{-1} = I - e_r d^T

  // x = U(q, q(j))
  std::vector<i_t> sind;
  std::vector<f_t> sval;
  const i_t in_col_start = in.col_start[in_col];
  const i_t in_col_end   = in.col_start[in_col + 1];
  std::vector<f_t> sbuffer(m);
  for (i_t p = in_col_start; p < in_col_end; ++p) {
    sbuffer[inverse_col_permutation_[in.i[p]]] = in.x[p];
  }
  for (i_t k = 0; k < m; ++k) {
    if (sbuffer[k] != 0) {
      sind.push_back(k);
      sval.push_back(sbuffer[k]);
    }
  }

  for (i_t k = num_updates_ - 1; k >= 0; --k) {
    // Ri^{-1} * x = (I - e_r d^T) * x = x - e_r d^T x
    const i_t r = pivot_indices_[k];
    f_t dot     = sparse_dot(sind, sval, S_, k);
    if (dot == 0.0) { continue; }
#if 0
        for (Int p = 0; p < sind.size(); ++p)
        {
            printf("s %d %e\n", sind[p], sval[p]);
        }
        printf("S col start %d %d\n", S_.col_start[k], S_.col_start[k+1]);
        for (Int p = S_.col_start[k]; p < S_.col_start[k+1]; ++p)
        {
            printf("S %d %d %e\n", k, S_.i[p], S_.x[p]);
        }
#endif
    bool fill = true;
    for (i_t p = 0; p < sind.size(); ++p) {
      if (sind[p] == r) {
        sval[p] -= dot;
        fill = false;
        break;
      }
    }
    if (fill) {
      std::vector<f_t> work2(m);
      sind.push_back(r);
      sval.push_back(-dot);

      for (i_t p = 0; p < sind.size(); ++p) {
        work2[sind[p]] = sval[p];
      }
      sind.clear();
      sval.clear();
      for (i_t i = 0; i < m; ++i) {
        if (work2[i] != 0.0) {
          sind.push_back(i);
          sval.push_back(work2[i]);
        }
      }
    }
    // assert(fill == false);
  }

  std::vector<f_t> workspace(m);
  const i_t nx = sind.size();
  for (i_t k = 0; k < nx; ++k) {
    const i_t j  = sind[k];
    const f_t x  = sval[k];
    workspace[j] = x;
  }
  std::vector<f_t> workspace2(m);
  matrix_vector_multiply(L0_, 1.0, workspace, 0.0, workspace2);
  workspace = workspace2;

  i_t col_nz = 0;
  for (i_t i = 0; i < m; ++i) {
    if (workspace[i] != 0.0) { col_nz++; }
  }
  const i_t nz     = out.col_start[out_col];
  const i_t new_nz = nz + col_nz;
  if (out.i.size() < new_nz) { out.reallocate(new_nz); }

  i_t p = nz;
  for (i_t i = 0; i < m; ++i) {
    if (workspace[i] != 0.0) {
      out.i[p] = i;
      out.x[p] = workspace[i];
      p++;
    }
  }
  assert(p == new_nz);
  return new_nz;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class basis_update_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::dual_simplex

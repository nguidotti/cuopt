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

#pragma once

#include <dual_simplex/types.hpp>
#include <dual_simplex/vector_math.hpp>
#include <dual_simplex/random.hpp>

#include <cassert>
#include <cstdio>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
class csr_matrix_t;  // Forward declaration of CSR matrix needed to define CSC matrix

// A sparse matrix stored in compressed sparse column format
template <typename i_t, typename f_t>
class csc_matrix_t {
 public:
  csc_matrix_t(i_t rows, i_t cols, i_t nz)
    : m(rows), n(cols), nz_max(nz), col_start(n + 1), i(nz_max), x(nz_max)
  {
  }

  // Adjust to i and x vectors for a new number of nonzeros
  void reallocate(i_t new_nz);

  // Convert the CSC matrix to a CSR matrix
  i_t to_compressed_row(
    cuopt::linear_programming::dual_simplex::csr_matrix_t<i_t, f_t>& Arow) const;

  // Permutes rows of a sparse matrix A. Computes C = A(p, :)
  i_t permute_rows(const std::vector<i_t>& pinv, csc_matrix_t<i_t, f_t>& C) const;

  // Permutes rows and columns of a sparse matrix A. Computes C = A(p, q)
  i_t permute_rows_and_cols(const std::vector<i_t>& pinv,
                            const std::vector<i_t>& q,
                            csc_matrix_t<i_t, f_t>& C) const;

  // Aj <- A(:, j), where Aj is a dense vector initially all zero
  i_t load_a_column(i_t j, std::vector<f_t>& Aj) const;

  // Compute the transpose of A
  i_t transpose(csc_matrix_t<i_t, f_t>& AT) const;

  // Remove columns from the matrix
  i_t remove_columns(const std::vector<i_t>& cols_to_remove);

  // Removes a single column from the matrix
  i_t remove_column(i_t col);

  // Removes a single row from the matrix
  i_t remove_row(i_t row);

  // Prints the matrix to stdout
  void print_matrix() const;

  // Prints the matrix to a file
  void print_matrix(FILE* fid) const;

  // Compute || A ||_1 = max_j (sum {i = 1 to m} | A(i, j) | )
  f_t norm1() const;

  f_t norm2_estimate(f_t tol = 1e-6) const
  {
    i_t m    = this->m;
    i_t n    = this->n;
    f_t norm = 0.0;

    std::vector<f_t> x(n);
    std::vector<f_t> Sx(m);

    for (i_t j = 0; j < n; ++j) {
      const i_t col_start = this->col_start[j];
      const i_t col_end   = this->col_start[j + 1];
      x[j]                = 0.0;
      for (i_t p = col_start; p < col_end; ++p) {
        x[j] += std::abs(this->x[p]);
      }
    }

    f_t e = vector_norm2<i_t, f_t>(x);
    if (e == 0.0) { return 0.0; }

    for (i_t j = 0; j < n; ++j) {
      x[j] /= e;
    }

    f_t e0 = 0.0;

    i_t iter           = 0;
    const i_t max_iter = 100;
    while (std::abs(e - e0) > tol * e) {
      e0 = e;
      matrix_vector_multiply(*this, 1.0, x, 0.0, Sx);
      f_t Sx_norm = vector_norm2<i_t, f_t>(Sx);
      if (Sx_norm == 0.0) {
        random_t<i_t> rng(0);
        for (i_t i = 0; i < m; ++i) {
          Sx[i] = rng.random_value(0.0, 1.0);
        }
        Sx_norm = vector_norm2<i_t, f_t>(Sx);
      }
      matrix_transpose_vector_multiply(*this, 1.0, Sx, 0.0, x);
      f_t norm_x = vector_norm2<i_t, f_t>(x);
      e          = norm_x / Sx_norm;
      for (i_t j = 0; j < n; ++j) {
        x[j] /= norm_x;
      }

      iter++;
      if (iter > max_iter) { break; }
    }
    return e;
  }

  i_t nz_max;                  // maximum number of entries
  i_t m;                       // number of rows
  i_t n;                       // number of columns
  std::vector<i_t> col_start;  // column pointers (size n + 1)
  std::vector<i_t> i;          // row indices, size nz_max
  std::vector<f_t> x;          // numerical values, size nz_max

  static_assert(std::is_signed_v<i_t>);  // Require signed integers (we make use of this
                                         // to avoid extra space / computation)
};

// A sparse matrix stored in compressed sparse row format
template <typename i_t, typename f_t>
class csr_matrix_t {
 public:
  // Convert the CSR matrix to CSC
  i_t to_compressed_col(csc_matrix_t<i_t, f_t>& Acol) const;

  // Create a new matrix with the marked rows removed
  i_t remove_rows(std::vector<i_t>& row_marker, csr_matrix_t<i_t, f_t>& Aout) const;

  i_t nz_max;                  // maximum number of nonzero entries
  i_t m;                       // number of rows
  i_t n;                       // number of cols
  std::vector<i_t> row_start;  // row pointers (size m + 1)
  std::vector<i_t> j;          // column inidices, size nz_max
  std::vector<f_t> x;          // numerical valuse, size nz_max

  static_assert(std::is_signed_v<i_t>);
};

template <typename i_t>
void cumulative_sum(std::vector<i_t>& inout, std::vector<i_t>& output);

template <typename i_t, typename f_t>
i_t coo_to_csc(const std::vector<i_t>& Ai,
               const std::vector<i_t>& Aj,
               const std::vector<f_t>& Ax,
               csc_matrix_t<i_t, f_t>& A);

template <typename i_t, typename f_t>
i_t scatter(const csc_matrix_t<i_t, f_t>& A,
            i_t j,
            f_t beta,
            std::vector<i_t>& workspace,
            std::vector<f_t>& x,
            i_t mark,
            csc_matrix_t<i_t, f_t>& C,
            i_t nz);

// x <- x + alpha * A(:, j)
template <typename i_t, typename f_t>
void scatter_dense(const csc_matrix_t<i_t, f_t>& A, i_t j, f_t alpha, std::vector<f_t>& x);

// Compute C = A*B where C is m x n, A is m x k, and B = k x n
// Do this by computing C(:, j) = A*B(:, j) = sum (i=1 to k) A(:, k)*B(i, j)
template <typename i_t, typename f_t>
i_t multiply(const csc_matrix_t<i_t, f_t>& A,
             const csc_matrix_t<i_t, f_t>& B,
             csc_matrix_t<i_t, f_t>& C);

// Compute C = alpha*A + beta*B
template <typename i_t, typename f_t>
i_t add(const csc_matrix_t<i_t, f_t>& A,
        const csc_matrix_t<i_t, f_t>& B,
        f_t alpha,
        f_t beta,
        csc_matrix_t<i_t, f_t>& C);

template <typename i_t, typename f_t>
f_t sparse_dot(const std::vector<i_t>& xind,
               const std::vector<f_t>& xval,
               const csc_matrix_t<i_t, f_t>& Y,
               i_t y_col);

// y <- alpha*A*x + beta*y
template <typename i_t, typename f_t>
i_t matrix_vector_multiply(const csc_matrix_t<i_t, f_t>& A,
                           f_t alpha,
                           const std::vector<f_t>& x,
                           f_t beta,
                           std::vector<f_t>& y);

// y <- alpha*A'*x + beta*y
template <typename i_t, typename f_t>
i_t matrix_transpose_vector_multiply(const csc_matrix_t<i_t, f_t>& A,
                                     f_t alpha,
                                     const std::vector<f_t>& x,
                                     f_t beta,
                                     std::vector<f_t>& y);

}  // namespace cuopt::linear_programming::dual_simplex

/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/tricks/markshare.hpp>

#include <mip_heuristics/mip_constants.hpp>
#include <utilities/logger.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <format>
#include <numeric>
#include <type_traits>

#include <omp.h>
#include <unistd.h>

#include <cstdio>

namespace cuopt::mathematical_optimization::mip {

namespace {

// Shifts `src` left by `shift` bits into `dst`, truncated to `bits` significant bits.
// `dst` must not alias `src`.
void shift_left_into(const uint64_t* src, uint64_t* dst, size_t words, size_t shift, size_t bits)
{
  const size_t word_shift = shift / 64;
  const size_t bit_shift  = shift % 64;
  for (size_t w = words; w-- > 0;) {
    uint64_t value = 0;
    if (w >= word_shift) {
      value = src[w - word_shift] << bit_shift;
      // x >> 64 is undefined behaviour, so the carry-in is only valid for a nonzero shift
      if (bit_shift != 0 && w > word_shift) {
        value |= src[w - word_shift - 1] >> (64 - bit_shift);
      }
    }
    dst[w] = value;
  }
  const size_t tail = bits % 64;
  if (tail != 0) { dst[words - 1] &= (uint64_t(1) << tail) - 1; }
}

// Bytes this process can reasonably allocate right now. Prefers the kernel's own MemAvailable
// estimate (which counts reclaimable page cache, unlike MemFree), then narrows it by a cgroup
// limit when one applies -- benchmark jobs run in containers, where /proc/meminfo reports the
// host rather than the container.
size_t available_memory_bytes()
{
  auto read_first_number = [](const char* path) -> size_t {
    size_t value = 0;
    if (FILE* file = std::fopen(path, "r")) {
      unsigned long long parsed = 0;
      if (std::fscanf(file, "%llu", &parsed) == 1) { value = parsed; }
      std::fclose(file);
    }
    return value;
  };

  size_t available = 0;
  if (FILE* file = std::fopen("/proc/meminfo", "r")) {
    char line[256];
    while (std::fgets(line, sizeof(line), file) != nullptr) {
      unsigned long long kilobytes = 0;
      if (std::sscanf(line, "MemAvailable: %llu kB", &kilobytes) == 1) {
        available = size_t(kilobytes) * 1024;
        break;
      }
    }
    std::fclose(file);
  }
  if (available == 0) {
    const long pages     = sysconf(_SC_AVPHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) { available = size_t(pages) * size_t(page_size); }
  }

  auto cgroup_headroom = [&](const char* limit_path, const char* usage_path) -> size_t {
    const size_t limit = read_first_number(limit_path);
    // An absent v2 limit reads as the literal "max" and fails to parse; an unset v1 limit shows
    // up as a huge sentinel. Treat anything past a terabyte as no limit at all.
    if (limit == 0 || limit > (size_t(1) << 40)) { return 0; }
    const size_t usage = read_first_number(usage_path);
    return usage < limit ? limit - usage : 0;
  };
  const size_t cgroup_v2 =
    cgroup_headroom("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current");
  const size_t cgroup_v1 = cgroup_headroom("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                                           "/sys/fs/cgroup/memory/memory.usage_in_bytes");
  if (cgroup_v2 != 0) { available = std::min(available, cgroup_v2); }
  if (cgroup_v1 != 0) { available = std::min(available, cgroup_v1); }
  return available;
}

double steady_seconds()
{
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration<double>(now).count();
}

}  // namespace

const char* markshare_status_to_string(markshare_status_t status)
{
  switch (status) {
    case markshare_status_t::NOT_APPLICABLE: return "not applicable";
    case markshare_status_t::OPTIMAL: return "optimal";
    case markshare_status_t::FEASIBLE: return "feasible";
    case markshare_status_t::BOUND_ONLY: return "bound only";
    case markshare_status_t::ABORTED: return "aborted";
  }
  return "unknown";
}

void markshare_build_row_table(const std::vector<markshare_coeff_t>& coefficients,
                               markshare_coeff_t capacity,
                               std::vector<markshare_prefix_t>& f)
{
  const size_t bits  = size_t(capacity) + 1;
  const size_t words = (bits + 63) / 64;

  f.assign(bits, markshare_unreachable);
  f[0] = 0;  // the empty subset reaches a residual of zero

  std::vector<uint64_t> reach(words, 0);
  std::vector<uint64_t> shifted(words, 0);
  reach[0] = 1;

  for (size_t j = 0; j < coefficients.size(); ++j) {
    const markshare_coeff_t a = coefficients[j];
    if (a == 0 || a > capacity) { continue; }

    shift_left_into(reach.data(), shifted.data(), words, size_t(a), bits);

    for (size_t w = 0; w < words; ++w) {
      uint64_t fresh = shifted[w] & ~reach[w];
      reach[w] |= shifted[w];
      while (fresh != 0) {
        const size_t u = w * 64 + size_t(std::countr_zero(fresh));
        f[u]           = markshare_prefix_t(j + 1);
        fresh &= fresh - 1;
      }
    }
  }
}

void markshare_build_joint_table(const std::vector<markshare_coeff_t>& coefficients0,
                                 const std::vector<markshare_coeff_t>& coefficients1,
                                 markshare_coeff_t capacity0,
                                 markshare_coeff_t capacity1,
                                 std::vector<markshare_prefix_t>& f)
{
  const size_t bits1 = size_t(capacity1) + 1;
  // Each u0 block is padded to a whole number of words, so a shift within a block physically
  // cannot alias into the u0 + 1 block. That makes the boundary mask implicit in the padding.
  const size_t stride_words = (bits1 + 63) / 64;
  const size_t blocks       = size_t(capacity0) + 1;

  f.assign(blocks * bits1, markshare_unreachable);
  f[0] = 0;

  std::vector<uint64_t> reach(blocks * stride_words, 0);
  std::vector<uint64_t> shifted(stride_words, 0);
  reach[0] = 1;

  for (size_t j = 0; j < coefficients0.size(); ++j) {
    const markshare_coeff_t a0 = coefficients0[j];
    const markshare_coeff_t a1 = coefficients1[j];
    if (a0 == 0 && a1 == 0) { continue; }
    if (a0 > capacity0 || a1 > capacity1) { continue; }

    // Descending u0 is what keeps this a 0/1 knapsack rather than an unbounded one.
    for (size_t u0 = blocks; u0-- > size_t(a0);) {
      const uint64_t* src = &reach[(u0 - size_t(a0)) * stride_words];
      uint64_t* dst       = &reach[u0 * stride_words];

      // `shifted` must be a real temporary: when a0 is zero, src and dst are the same block.
      shift_left_into(src, shifted.data(), stride_words, size_t(a1), bits1);

      for (size_t w = 0; w < stride_words; ++w) {
        uint64_t fresh = shifted[w] & ~dst[w];
        dst[w] |= shifted[w];
        while (fresh != 0) {
          const size_t u1    = w * 64 + size_t(std::countr_zero(fresh));
          f[u0 * bits1 + u1] = markshare_prefix_t(j + 1);
          fresh &= fresh - 1;
        }
      }
    }
  }
}

template <typename i_t, typename f_t>
void markshare_solver_t<i_t, f_t>::fingerprint_set_t::init(size_t capacity)
{
  size_t bits = 1;
  while ((size_t(1) << bits) < capacity * 2) { ++bits; }
  mask = (size_t(1) << bits) - 1;
  slot.assign(mask + 1, 0);
}

template <typename i_t, typename f_t>
void markshare_solver_t<i_t, f_t>::fingerprint_set_t::insert(uint64_t fingerprint)
{
  // Zero marks an empty slot, so it is the one value that cannot be stored verbatim.
  if (fingerprint == 0) { fingerprint = 1; }
  size_t p = fingerprint & mask;
  for (;;) {
    std::atomic_ref<uint64_t> cell(slot[p]);
    const uint64_t current = cell.load(std::memory_order_relaxed);
    if (current == fingerprint) { return; }
    if (current == 0) {
      uint64_t expected = 0;
      if (cell.compare_exchange_strong(expected, fingerprint, std::memory_order_relaxed)) {
        return;
      }
      if (expected == fingerprint) { return; }
    }
    p = (p + 1) & mask;
  }
}

template <typename i_t, typename f_t>
bool markshare_solver_t<i_t, f_t>::fingerprint_set_t::contains(uint64_t fingerprint) const
{
  if (fingerprint == 0) { fingerprint = 1; }
  size_t p = fingerprint & mask;
  while (slot[p] != 0) {
    if (slot[p] == fingerprint) { return true; }
    p = (p + 1) & mask;
  }
  return false;
}

template <typename i_t, typename f_t>
uint64_t markshare_solver_t<i_t, f_t>::residual_fingerprint(
  const markshare_coeff_t* residual) const
{
  uint64_t h = 1469598103934665603ull;
  for (i_t k = 0; k < model_.m; ++k) {
    h ^= uint64_t(uint32_t(residual[k])) + 0x9e3779b9ull;
    h *= 1099511628211ull;
  }
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdull;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ull;
  return h ^ (h >> 33);
}

template <typename i_t, typename f_t>
i_t markshare_solver_t<i_t, f_t>::choose_hash_depth() const
{
  const i_t n = model_.n;
  // Small models finish in milliseconds without the terminal and would only pay its build cost.
  if (n <= settings_.hash_min_cols) { return 0; }
  // Leaving a fixed number of levels above the terminal is what bounds the remaining tree; each
  // extra level of depth removes another factor of two from it. Then clamp to what memory buys.
  i_t depth = std::min(n / 2 + settings_.hash_depth_offset, settings_.hash_max_depth);

  size_t budget = settings_.max_hash_bytes;
  if (budget == 0) {
    // The table is freed before branch and bound starts, but the rest of the solve still needs
    // room while it is live, so claim only a fraction of what is free.
    const size_t available = available_memory_bytes();
    const i_t divisor      = std::max(settings_.hash_memory_divisor, i_t(1));
    budget                 = std::min(available / divisor, settings_.hash_bytes_cap);
  }

  while (depth > 0) {
    // Open addressing at 50% load: 2^(depth + 1) slots of eight bytes.
    const size_t bytes = (size_t(1) << (depth + 1)) * sizeof(uint64_t);
    if (bytes <= budget) { break; }
    --depth;
  }
  return depth < 8 ? 0 : depth;
}

template <typename i_t, typename f_t>
void markshare_solver_t<i_t, f_t>::build_hash()
{
  hash_depth_ = choose_hash_depth();
  if (hash_depth_ <= 0) { return; }

  const i_t m = model_.m;
  const i_t h = hash_depth_;
  hash_.init(size_t(1) << h);

  // Split the enumerated columns into a prefix that fans out across tasks and a suffix each task
  // walks in Gray code order, so every subset costs one add per row rather than h of them.
  i_t top = 0;
  while ((i_t(1) << top) < 4 * num_threads_ && top < h - 8) { ++top; }
  const i_t low       = h - top;
  const size_t blocks = size_t(1) << top;

#pragma omp taskloop grainsize(1) shared(m, h, top, low)
  for (size_t block = 0; block < blocks; ++block) {
    std::vector<markshare_coeff_t> sum(m, 0);
    for (i_t t = 0; t < top; ++t) {
      if ((block >> t & 1) != 0) {
        const markshare_coeff_t* column = &model_.a_col[size_t(low + t) * m];
        for (i_t k = 0; k < m; ++k) { sum[k] += column[k]; }
      }
    }
    auto record = [&]() {
      for (i_t k = 0; k < m; ++k) {
        if (sum[k] > model_.b[k]) { return; }
      }
      hash_.insert(residual_fingerprint(sum.data()));
    };

    record();
    uint64_t previous = 0;
    for (uint64_t g = 1; g < (uint64_t(1) << low); ++g) {
      const uint64_t code          = g ^ (g >> 1);
      const uint64_t diff          = code ^ previous;
      const i_t j                  = std::countr_zero(diff);
      const markshare_coeff_t* col = &model_.a_col[size_t(j) * m];
      if ((code & diff) != 0) {
        for (i_t k = 0; k < m; ++k) { sum[k] += col[k]; }
      } else {
        for (i_t k = 0; k < m; ++k) { sum[k] -= col[k]; }
      }
      previous = code;
      record();
    }
  }
}

template <typename i_t, typename f_t>
markshare_solver_t<i_t, f_t>::markshare_solver_t(
  const simplex::user_problem_t<i_t, f_t>& user_problem,
  const markshare_settings_t<i_t, f_t>& settings)
  : problem_(user_problem), settings_(settings)
{
}

template <typename i_t, typename f_t>
bool markshare_solver_t<i_t, f_t>::detect()
{
  detected_ = false;
  // Restricted to double on purpose: with float, an exact row scaling of the form
  // integer_multiple / gcd lands around 1e-7 relative error, which straddles the normalization
  // tolerance. The upside on float is zero and the downside is a wrong optimality claim.
  if constexpr (std::is_same_v<f_t, double>) { detected_ = detect_impl(); }
  if (!detected_) { model_ = model_t{}; }
  return detected_;
}

template <typename i_t, typename f_t>
bool markshare_solver_t<i_t, f_t>::detect_impl()
{
  const i_t m        = problem_.num_rows;
  const i_t num_cols = problem_.num_cols;

  // --- cheap structural gates, before anything is allocated -------------------------------
  // Two rows minimum: the joint table is what makes the search tractable, and a single row
  // market split is just a subset sum that presolve already handles.
  if (m < 2 || m > settings_.max_rows) { return false; }
  // Room for several continuous columns per row: running ahead of PaPILO means fixed decoy
  // columns are still present (markshare1 and markshare2 carry one per row).
  if (num_cols < m + 1 || num_cols > settings_.max_core_cols + 4 * settings_.max_rows) {
    return false;
  }
  // A range row is reported with sense 'E' plus a separate range value, so it would otherwise
  // be misread as an equality.
  if (problem_.num_range_rows != 0 || !problem_.range_rows.empty()) { return false; }
  if (problem_.A.nnz() > m * num_cols) { return false; }

  for (i_t k = 0; k < m; ++k) {
    if (problem_.row_sense[k] != 'E' || !std::isfinite(problem_.rhs[k])) {
      CUOPT_LOG_DEBUG("markshare: row %d is not a finite equality", k);
      return false;
    }
  }

  // --- classify every column as a core binary or a row slack -------------------------------
  const f_t int_tol = settings_.integrality_tolerance;
  std::vector<i_t> slack_col(m, -1);
  std::vector<double> row_divisor(m, 0.0);
  std::vector<i_t> core_col;
  std::vector<i_t> pinned_col;
  std::vector<double> fixed_activity(m, 0.0);
  core_col.reserve(num_cols);

  for (i_t j = 0; j < num_cols; ++j) {
    const i_t len = problem_.A.col_length(j);

    // Fixed columns contribute a constant. This runs ahead of cuOpt's trivial presolve, so the
    // decoy columns that markshare1 and markshare2 pin at zero are still here.
    if (problem_.lower[j] == problem_.upper[j] && std::isfinite(problem_.lower[j])) {
      const double value = problem_.lower[j];
      pinned_col.push_back(j);
      if (value != 0.0) {
        for (i_t e = problem_.A.col_start[j]; e < problem_.A.col_start[j + 1]; ++e) {
          fixed_activity[problem_.A.i[e]] += problem_.A.x[e] * value;
        }
      }
      continue;
    }

    if (problem_.var_types[j] != simplex::variable_type_t::CONTINUOUS) {
      // get_host_user_problem collapses every non-continuous type to INTEGER, so a binary has to
      // be recognised by its bounds rather than by its declared type.
      if (problem_.lower[j] != 0.0 || problem_.upper[j] != 1.0) {
        CUOPT_LOG_DEBUG("markshare: integer column %d is not binary", j);
        return false;
      }
      if (std::abs(problem_.objective[j]) > int_tol) {
        CUOPT_LOG_DEBUG("markshare: binary column %d carries objective cost", j);
        return false;
      }
      if (len > m) { return false; }
      core_col.push_back(j);
      continue;
    }

    // Continuous columns must be the per-row slack, and nothing else.
    if (len != 1) {
      CUOPT_LOG_DEBUG("markshare: continuous column %d is not a singleton", j);
      return false;
    }
    const i_t entry  = problem_.A.col_start[j];
    const i_t k      = problem_.A.i[entry];
    const double gam = problem_.A.x[entry];
    if (gam == 0.0) { return false; }
    // A strictly positive cost. A maximisation model arrives with negated costs and is rejected
    // here, which is what we want.
    if (!(problem_.objective[j] > int_tol)) {
      CUOPT_LOG_DEBUG("markshare: continuous column %d has non-positive cost", j);
      return false;
    }
    if (problem_.lower[j] != 0.0) { return false; }
    // Two slacks in one row would be the two sided market split form. Reading that as one sided
    // forbids negative residuals and could "prove" an optimum above the true one.
    if (slack_col[k] >= 0) {
      CUOPT_LOG_DEBUG("markshare: row %d has more than one slack", k);
      return false;
    }
    slack_col[k]   = j;
    row_divisor[k] = gam;
  }

  const i_t core_count = core_col.size();
  if (core_count < 1 || core_count > settings_.max_core_cols) { return false; }
  // Every column must be accounted for as a core binary, a row slack, or a fixed column.
  if (core_count + m + i_t(pinned_col.size()) != num_cols) { return false; }
  for (i_t k = 0; k < m; ++k) {
    if (slack_col[k] < 0) {
      CUOPT_LOG_DEBUG("markshare: row %d has no slack", k);
      return false;
    }
  }

  // All slack costs must agree: the objective has to be a positive multiple of the total slack,
  // otherwise the objective levels are not evenly spaced.
  const f_t slack_cost = problem_.objective[slack_col[0]];
  for (i_t k = 1; k < m; ++k) {
    if (std::abs(problem_.objective[slack_col[k]] - slack_cost) > int_tol) {
      CUOPT_LOG_DEBUG("markshare: slack costs are not uniform");
      return false;
    }
  }

  // The slack bound must not bind before the residual does.
  for (i_t k = 0; k < m; ++k) {
    const double upper = problem_.upper[slack_col[k]];
    if (!(std::isinf(upper) && upper > 0.0)) {
      const double implied = problem_.rhs[k] / row_divisor[k];
      if (upper < implied) {
        CUOPT_LOG_DEBUG("markshare: slack of row %d is bounded above", k);
        return false;
      }
    }
  }

  // --- normalize each row by its own slack coefficient -------------------------------------
  // Dividing row k through by gamma_k makes the normalized slack coefficient exactly one, so the
  // slack value equals the residual. This also absorbs any MIP row scaling factor exactly, which
  // dividing by the coefficient gcd would not: row scaling leaves the *slack* coefficient as a
  // possibly non-integer rational.
  const double exact_tol = settings_.exactness_tolerance;
  auto normalize         = [&](double value, double gamma, markshare_coeff_t& out) -> bool {
    const double q      = value / gamma;
    const double q_int  = std::round(q);
    if (std::abs(q - q_int) > int_tol * std::max(1.0, std::abs(q))) { return false; }
    if (std::abs(q_int * gamma - value) > exact_tol * std::max(1.0, std::abs(value))) {
      return false;
    }
    if (std::abs(q_int) > settings_.max_normalized_rhs) { return false; }
    out = markshare_coeff_t(q_int);
    return true;
  };

  std::vector<markshare_coeff_t> a_row(size_t(m) * core_count, 0);
  std::vector<markshare_coeff_t> b(m, 0);
  for (i_t k = 0; k < m; ++k) {
    // Fixed columns have already been folded out of the right hand side.
    if (!normalize(problem_.rhs[k] - fixed_activity[k], row_divisor[k], b[k])) {
      CUOPT_LOG_DEBUG("markshare: rhs of row %d does not normalize to an integer", k);
      return false;
    }
  }
  for (i_t p = 0; p < core_count; ++p) {
    const i_t j = core_col[p];
    for (i_t e = problem_.A.col_start[j]; e < problem_.A.col_start[j + 1]; ++e) {
      const i_t k = problem_.A.i[e];
      markshare_coeff_t normalized;
      if (!normalize(problem_.A.x[e], row_divisor[k], normalized)) {
        CUOPT_LOG_DEBUG("markshare: entry (%d, %d) does not normalize to an integer", k, j);
        return false;
      }
      a_row[size_t(k) * core_count + p] = normalized;
    }
  }

  // --- complement all-negative columns, then require a non-negative model -------------------
  std::vector<uint8_t> flipped(core_count, 0);
  for (i_t p = 0; p < core_count; ++p) {
    bool any_negative = false;
    bool any_positive = false;
    for (i_t k = 0; k < m; ++k) {
      const markshare_coeff_t v = a_row[size_t(k) * core_count + p];
      any_negative |= v < 0;
      any_positive |= v > 0;
    }
    if (!any_negative) { continue; }
    if (any_positive) {
      CUOPT_LOG_DEBUG("markshare: core column %d has mixed signs", core_col[p]);
      return false;
    }
    // x = 1 - x': the coefficient negates and its old value moves to the right hand side.
    flipped[p] = 1;
    for (i_t k = 0; k < m; ++k) {
      markshare_coeff_t& v = a_row[size_t(k) * core_count + p];
      b[k] -= v;
      v = -v;
    }
  }
  for (i_t k = 0; k < m; ++k) {
    if (b[k] < 0 || b[k] > settings_.max_normalized_rhs) {
      CUOPT_LOG_DEBUG("markshare: normalized rhs of row %d is out of range", k);
      return false;
    }
  }

  // --- drop all-zero columns, then order by column sum --------------------------------------
  // All-zero core columns join the already fixed columns: both are simply written at their lower
  // bound during reconstruction, and keeping them would double the search space for nothing.
  std::vector<i_t> kept;
  std::vector<int64_t> col_sum(core_count, 0);
  for (i_t p = 0; p < core_count; ++p) {
    int64_t sum = 0;
    for (i_t k = 0; k < m; ++k) { sum += a_row[size_t(k) * core_count + p]; }
    col_sum[p] = sum;
    if (sum == 0) {
      pinned_col.push_back(core_col[p]);
    } else {
      kept.push_back(p);
    }
  }
  if (kept.empty()) { return false; }

  // Ascending by column sum. The enumeration runs backwards, so it decides the largest
  // coefficients first while the tables' prefixes hold the smallest -- that is what makes both
  // the remaining-capacity prune and the reachability prune bite at shallow depth. Measured on
  // 40 columns this is the difference between milliseconds and not terminating.
  std::stable_sort(
    kept.begin(), kept.end(), [&](i_t x, i_t y) { return col_sum[x] < col_sum[y]; });

  const i_t n = kept.size();
  if (n >= markshare_unreachable) { return false; }
  // Beyond this the enumeration cannot finish even with the meet-in-the-middle terminal, and the
  // budget is the whole remaining solve, so hand the model back rather than spending it all.
  if (n > settings_.max_search_cols) {
    CUOPT_LOG_DEBUG("markshare: %d core columns is beyond the tractable range", n);
    return false;
  }

  model_.m           = m;
  model_.n           = n;
  model_.slack_col   = std::move(slack_col);
  model_.row_divisor = std::move(row_divisor);
  model_.pinned_col  = std::move(pinned_col);
  model_.b           = std::move(b);
  model_.slack_cost  = slack_cost;
  model_.core_col.resize(n);
  model_.flipped.resize(n);
  model_.a_row.assign(size_t(m) * n, 0);
  model_.a_col.assign(size_t(n) * m, 0);
  for (i_t p = 0; p < n; ++p) {
    const i_t src      = kept[p];
    model_.core_col[p] = core_col[src];
    model_.flipped[p]  = flipped[src];
    for (i_t k = 0; k < m; ++k) {
      const markshare_coeff_t v         = a_row[size_t(k) * core_count + src];
      model_.a_row[size_t(k) * n + p]   = v;
      model_.a_col[size_t(p) * m + k]   = v;
    }
  }

  model_.prefix_max.assign(size_t(m) * (n + 1), 0);
  model_.row_gcd.assign(m, 0);
  for (i_t k = 0; k < m; ++k) {
    markshare_coeff_t running = 0;
    markshare_coeff_t divisor = 0;
    for (i_t p = 0; p < n; ++p) {
      const markshare_coeff_t v                    = model_.a_row[size_t(k) * n + p];
      running                                      += v;
      model_.prefix_max[size_t(k) * (n + 1) + p + 1] = running;
      divisor                                       = std::gcd(divisor, v);
    }
    model_.row_gcd[k] = divisor;
    if (running < model_.b[k]) {
      CUOPT_LOG_DEBUG("markshare: row %d cannot reach its rhs", k);
      return false;
    }
  }

  // --- pick the joint pair and check the table budget ---------------------------------------
  int64_t best_cells = -1;
  for (i_t k0 = 0; k0 < m; ++k0) {
    for (i_t k1 = k0 + 1; k1 < m; ++k1) {
      const int64_t cells = (int64_t(model_.b[k0]) + 1) * (int64_t(model_.b[k1]) + 1);
      if (best_cells < 0 || cells < best_cells) {
        best_cells   = cells;
        joint_row0_  = k0;
        joint_row1_  = k1;
      }
    }
  }
  const size_t joint_bytes = size_t(best_cells) * sizeof(markshare_prefix_t);
  if (joint_bytes > settings_.max_table_bytes) {
    CUOPT_LOG_DEBUG("markshare: joint table would need %zu bytes", joint_bytes);
    return false;
  }

  CUOPT_LOG_INFO("%s",
                 std::format("Markshare structure detected: {} rows, {} binaries, rhs max {}, "
                             "joint rows ({}, {}), joint table {:.1f} MB",
                             model_.m,
                             model_.n,
                             *std::max_element(model_.b.begin(), model_.b.end()),
                             joint_row0_,
                             joint_row1_,
                             joint_bytes / (1024.0 * 1024.0))
                   .c_str());
  return true;
}

template <typename i_t, typename f_t>
void markshare_solver_t<i_t, f_t>::build_tables()
{
  const i_t m = model_.m;
  const i_t n = model_.n;

  row_tables_.resize(m);
  extra_rows_.clear();
  std::vector<markshare_coeff_t> coefficients(n);
  for (i_t k = 0; k < m; ++k) {
    for (i_t p = 0; p < n; ++p) { coefficients[p] = model_.a_row[size_t(k) * n + p]; }
    markshare_build_row_table(coefficients, model_.b[k], row_tables_[k]);
    if (k != joint_row0_ && k != joint_row1_) { extra_rows_.push_back(k); }
  }

  std::vector<markshare_coeff_t> c0(n), c1(n);
  for (i_t p = 0; p < n; ++p) {
    c0[p] = model_.a_row[size_t(joint_row0_) * n + p];
    c1[p] = model_.a_row[size_t(joint_row1_) * n + p];
  }
  markshare_build_joint_table(c0, c1, model_.b[joint_row0_], model_.b[joint_row1_], joint_);
  joint_stride_ = size_t(model_.b[joint_row1_]) + 1;

  context_.resize(n, m);
  value_.assign(n, 0);
  target_.assign(m, 0);
}

template <typename i_t, typename f_t>
typename markshare_solver_t<i_t, f_t>::dfs_result_t markshare_solver_t<i_t, f_t>::run_dfs_from(
  dfs_context_t& ctx,
  i_t start_depth,
  const markshare_coeff_t* start_residual,
  const std::atomic<bool>* stop,
  i_t terminal_depth) const
{
  const i_t m = model_.m;
  const i_t n = model_.n;

  std::copy(start_residual, start_residual + m, ctx.residual.begin() + size_t(start_depth) * m);
  ctx.branch[start_depth] = 0;
  i_t j                   = start_depth;
  int64_t next_check      = ctx.nodes + 4096;

  for (;;) {
    if (j == terminal_depth) {
      // At depth zero the remaining-capacity prune has already forced every residual to exactly
      // zero, so reaching this point is a solution with no further check needed.
      if (terminal_depth == 0) { return dfs_result_t::FOUND; }
      // Otherwise ask the meet-in-the-middle table whether the columns below can supply the
      // residual exactly. A fingerprint collision surfaces as a sub-search that finds nothing,
      // which simply resumes the enumeration -- it can never hide a solution.
      const markshare_coeff_t* residual = &ctx.residual[size_t(j) * m];
      if (hash_.contains(residual_fingerprint(residual))) {
        const std::vector<markshare_coeff_t> below(residual, residual + m);
        if (run_dfs_from(ctx, terminal_depth, below.data(), stop, 0) == dfs_result_t::FOUND) {
          return dfs_result_t::FOUND;
        }
      }
      ++j;
      if (j > start_depth) { return dfs_result_t::EXHAUSTED; }
      continue;
    }
    if (ctx.branch[j] == 2) {
      ++j;
      if (j > start_depth) { return dfs_result_t::EXHAUSTED; }
      continue;
    }

    // Try one first: it shrinks the residual faster, so the capacity prune bites sooner.
    const uint8_t v = 1 - ctx.branch[j];
    ++ctx.branch[j];
    ++ctx.nodes;
    if (ctx.nodes >= next_check) {
      next_check = ctx.nodes + 4096;
      if (stop != nullptr && stop->load(std::memory_order_relaxed)) {
        return dfs_result_t::BUDGET;
      }
      if (std::isfinite(deadline_) && steady_seconds() > deadline_) {
        return dfs_result_t::BUDGET;
      }
    }

    const i_t p                       = j - 1;
    const markshare_coeff_t* column   = &model_.a_col[size_t(p) * m];
    const markshare_coeff_t* previous = &ctx.residual[size_t(j) * m];
    markshare_coeff_t* current        = &ctx.residual[size_t(p) * m];

    // Prune 1 and 2 fused into one pass: negative residual, or a residual larger than the
    // remaining columns can possibly supply.
    bool pruned = false;
    for (i_t k = 0; k < m; ++k) {
      const markshare_coeff_t left = previous[k] - (v != 0 ? column[k] : 0);
      if (left < 0 || left > model_.prefix_max[size_t(k) * (n + 1) + p]) {
        pruned = true;
        break;
      }
      current[k] = left;
    }
    // Prune 3: the rows outside the joint pair. Their tables are a couple of KB and stay in L1.
    if (!pruned) {
      for (i_t k : extra_rows_) {
        if (row_tables_[k][size_t(current[k])] > p) {
          pruned = true;
          break;
        }
      }
    }
    // Prune 4: the joint table. Strongest, but also the one lookup that misses cache, so it goes
    // last.
    if (!pruned && joint_at(current[joint_row0_], current[joint_row1_]) > p) { pruned = true; }

    if (!pruned) {
      ctx.value[p]  = v;
      ctx.branch[p] = 0;
      j             = p;
    }
  }
}

template <typename i_t, typename f_t>
void markshare_solver_t<i_t, f_t>::collect_seeds(const std::vector<markshare_coeff_t>& target,
                                                 i_t depth,
                                                 std::vector<seed_t>& seeds)
{
  const i_t m          = model_.m;
  const i_t n          = model_.n;
  const i_t stop_depth = n - depth;

  std::vector<markshare_coeff_t> residual(size_t(n + 1) * m, 0);
  std::vector<uint8_t> branch(n + 1, 0);
  std::vector<uint8_t> value(n, 0);
  std::copy(target.begin(), target.end(), residual.begin() + size_t(n) * m);

  i_t j = n;
  for (;;) {
    if (j == stop_depth) {
      seed_t seed;
      seed.value = value;
      seed.residual.assign(residual.begin() + size_t(j) * m,
                           residual.begin() + size_t(j) * m + m);
      seeds.push_back(std::move(seed));
      ++j;
      if (j > n) { return; }
      continue;
    }
    if (branch[j] == 2) {
      ++j;
      if (j > n) { return; }
      continue;
    }

    const uint8_t v = 1 - branch[j];
    ++branch[j];

    const i_t p                       = j - 1;
    const markshare_coeff_t* column   = &model_.a_col[size_t(p) * m];
    const markshare_coeff_t* previous = &residual[size_t(j) * m];
    markshare_coeff_t* current        = &residual[size_t(p) * m];

    bool pruned = false;
    for (i_t k = 0; k < m; ++k) {
      const markshare_coeff_t left = previous[k] - (v != 0 ? column[k] : 0);
      if (left < 0 || left > model_.prefix_max[size_t(k) * (n + 1) + p]) {
        pruned = true;
        break;
      }
      current[k] = left;
    }
    if (!pruned) {
      for (i_t k : extra_rows_) {
        if (row_tables_[k][size_t(current[k])] > p) {
          pruned = true;
          break;
        }
      }
    }
    if (!pruned && joint_at(current[joint_row0_], current[joint_row1_]) > p) { pruned = true; }

    if (!pruned) {
      value[p]  = v;
      branch[p] = 0;
      j         = p;
    }
  }
}

template <typename i_t, typename f_t>
typename markshare_solver_t<i_t, f_t>::dfs_result_t markshare_solver_t<i_t, f_t>::run_dfs(
  const std::vector<markshare_coeff_t>& target)
{
  const i_t n = model_.n;

  // Serial for small models: seeding and task overhead would dominate a search that finishes in
  // microseconds anyway.
  if (num_threads_ < 2 || n < 16 || n - 1 <= hash_depth_) {
    context_.branch.assign(n + 1, 0);
    const int64_t before  = context_.nodes;
    const dfs_result_t rc = run_dfs_from(context_, n, target.data(), nullptr, hash_depth_);
    nodes_ += context_.nodes - before;
    if (rc == dfs_result_t::FOUND) { value_ = context_.value; }
    return rc;
  }

  // Split the trailing columns into independent subtrees. Subtree sizes are wildly uneven, so
  // aim for several tasks per thread and let the scheduler balance them.
  std::vector<seed_t> seeds;
  i_t depth = 1;
  while (depth < n - 1) {
    seeds.clear();
    collect_seeds(target, depth, seeds);
    if (seeds.empty()) { return dfs_result_t::EXHAUSTED; }
    if (i_t(seeds.size()) >= 8 * num_threads_) { break; }
    ++depth;
  }
  if (seeds.empty()) { return dfs_result_t::EXHAUSTED; }

  const i_t start_depth = n - depth;
  std::atomic<bool> stop{false};
  std::atomic<bool> found{false};
  std::atomic<bool> budget{false};
  std::atomic<int64_t> node_total{0};
  const size_t seed_count = seeds.size();

#pragma omp taskloop grainsize(1) shared(seeds, stop, found, budget, node_total)
  for (size_t s = 0; s < seed_count; ++s) {
    if (!stop.load(std::memory_order_relaxed)) {
      dfs_context_t ctx;
      ctx.resize(model_.n, model_.m);
      std::copy(seeds[s].value.begin(), seeds[s].value.end(), ctx.value.begin());
      const dfs_result_t rc = run_dfs_from(ctx, start_depth, seeds[s].residual.data(), &stop, hash_depth_);
      node_total.fetch_add(ctx.nodes, std::memory_order_relaxed);
      if (rc == dfs_result_t::FOUND) {
        bool expected = false;
        // First finder wins; the rest are told to stop.
        if (found.compare_exchange_strong(expected, true)) { value_ = ctx.value; }
        stop.store(true, std::memory_order_relaxed);
      } else if (rc == dfs_result_t::BUDGET) {
        budget.store(true, std::memory_order_relaxed);
        stop.store(true, std::memory_order_relaxed);
      }
    }
  }

  nodes_ += node_total.load(std::memory_order_relaxed);
  if (found.load(std::memory_order_relaxed)) { return dfs_result_t::FOUND; }
  if (budget.load(std::memory_order_relaxed)) { return dfs_result_t::BUDGET; }
  return dfs_result_t::EXHAUSTED;
}

template <typename i_t, typename f_t>
bool markshare_solver_t<i_t, f_t>::enumerate_level(i_t level,
                                                   std::vector<markshare_coeff_t>& slack,
                                                   i_t index,
                                                   bool& found)
{
  const i_t m = model_.m;
  if (index == m - 1) {
    slack[index] = level;

    for (i_t k = 0; k < m; ++k) {
      if (slack[k] > model_.b[k]) { return false; }
      target_[k] = model_.b[k] - slack[k];
    }
    // Three O(m) rejects that kill most target vectors without entering the DFS at all.
    for (i_t k = 0; k < m; ++k) {
      if (model_.row_gcd[k] > 0 && (target_[k] % model_.row_gcd[k]) != 0) { return false; }
      if (row_tables_[k][size_t(target_[k])] > model_.n) { return false; }
    }
    if (joint_at(target_[joint_row0_], target_[joint_row1_]) > model_.n) { return false; }

    ++targets_;
    const dfs_result_t result = run_dfs(target_);
    if (result == dfs_result_t::FOUND) {
      found = true;
      return true;
    }
    if (result == dfs_result_t::BUDGET) {
      budget_exhausted_ = true;
      return true;
    }
    return false;
  }

  for (i_t v = 0; v <= level; ++v) {
    slack[index] = v;
    if (enumerate_level(level - v, slack, index + 1, found)) { return true; }
  }
  return false;
}

template <typename i_t, typename f_t>
bool markshare_solver_t<i_t, f_t>::reconstruct(std::vector<f_t>& solution, f_t& objective) const
{
  const i_t m        = model_.m;
  const i_t n        = model_.n;
  const i_t num_cols = problem_.num_cols;

  solution.assign(num_cols, f_t{0});
  for (i_t p = 0; p < n; ++p) {
    const uint8_t v            = model_.flipped[p] != 0 ? 1 - value_[p] : value_[p];
    solution[model_.core_col[p]] = v;
  }
  for (i_t col : model_.pinned_col) { solution[col] = problem_.lower[col]; }

  // Recompute each slack from the original row rather than from the level distribution, so a
  // normalization mistake shows up here instead of hiding.
  std::vector<double> activity(m, 0.0);
  for (i_t j = 0; j < num_cols; ++j) {
    const double x = solution[j];
    if (x == 0.0) { continue; }
    for (i_t e = problem_.A.col_start[j]; e < problem_.A.col_start[j + 1]; ++e) {
      activity[problem_.A.i[e]] += problem_.A.x[e] * x;
    }
  }
  for (i_t k = 0; k < m; ++k) {
    const double slack = (problem_.rhs[k] - activity[k]) / model_.row_divisor[k];
    if (slack < -settings_.integrality_tolerance) {
      CUOPT_LOG_ERROR("markshare: reconstructed slack of row %d is negative", k);
      return false;
    }
    solution[model_.slack_col[k]] = std::max(slack, 0.0);
  }

  // Independent verification against the untouched user problem. This is the last line of
  // defence against every detection assumption above.
  std::vector<double> check(m, 0.0);
  for (i_t j = 0; j < num_cols; ++j) {
    const double x = solution[j];
    if (x < problem_.lower[j] - settings_.integrality_tolerance ||
        x > problem_.upper[j] + settings_.integrality_tolerance) {
      CUOPT_LOG_ERROR("markshare: reconstructed column %d violates its bounds", j);
      return false;
    }
    if (problem_.var_types[j] != simplex::variable_type_t::CONTINUOUS &&
        std::abs(x - std::round(x)) > settings_.integrality_tolerance) {
      CUOPT_LOG_ERROR("markshare: reconstructed column %d is fractional", j);
      return false;
    }
    if (x == 0.0) { continue; }
    for (i_t e = problem_.A.col_start[j]; e < problem_.A.col_start[j + 1]; ++e) {
      check[problem_.A.i[e]] += problem_.A.x[e] * x;
    }
  }
  for (i_t k = 0; k < m; ++k) {
    const double tolerance = 1e-6 * std::max(1.0, std::abs(double(problem_.rhs[k])));
    if (std::abs(check[k] - problem_.rhs[k]) > tolerance) {
      CUOPT_LOG_ERROR("markshare: reconstructed row %d is violated", k);
      return false;
    }
  }

  double total = 0.0;
  for (i_t j = 0; j < num_cols; ++j) { total += problem_.objective[j] * solution[j]; }
  objective = total;
  return true;
}

template <typename i_t, typename f_t>
markshare_result_t<i_t, f_t> markshare_solver_t<i_t, f_t>::solve()
{
  markshare_result_t<i_t, f_t> result;
  if (!detected_) { return result; }

  const double started = steady_seconds();
  deadline_ = std::isfinite(settings_.time_limit) ? started + settings_.time_limit
                                                  : std::numeric_limits<double>::infinity();
  nodes_            = 0;
  targets_          = 0;
  budget_exhausted_ = false;
  // We run under `omp masked` inside the solver's parallel region, so the rest of the team is
  // parked at the barrier and available to pick up tasks.
  num_threads_ = omp_get_num_threads();

  build_tables();
  build_hash();
  if (hash_depth_ > 0) {
    CUOPT_LOG_INFO("%s",
                   std::format("Markshare meet-in-the-middle terminal at depth {} ({:.1f} MB)",
                               hash_depth_,
                               hash_.bytes() / (1024.0 * 1024.0))
                     .c_str());
  }

  std::vector<markshare_coeff_t> slack(model_.m, 0);
  i_t first_incomplete = -1;
  i_t solution_level   = -1;

  for (i_t level = 0; level <= settings_.max_level; ++level) {
    bool found = false;
    enumerate_level(level, slack, 0, found);
    if (found) {
      solution_level = level;
      break;
    }
    if (budget_exhausted_) {
      first_incomplete = level;
      break;
    }
    result.levels_exhausted = level + 1;
  }

  result.nodes       = nodes_;
  result.targets     = targets_;
  result.search_time = steady_seconds() - started;
  // Exhausting levels 0..L-1 proves that no feasible point has a total slack below L. The slacks
  // are implied integer, so the levels are spaced exactly one apart and the bound is exact.
  result.proven_lower_bound = model_.slack_cost * result.levels_exhausted;

  if (solution_level >= 0) {
    f_t objective = 0;
    if (!reconstruct(result.solution, objective)) {
      result.solution.clear();
      result.status = markshare_status_t::NOT_APPLICABLE;
      return result;
    }
    result.objective = objective;
    // A solution at level T is optimal only if every level below it was fully exhausted.
    result.status = first_incomplete < 0 ? markshare_status_t::OPTIMAL
                                         : markshare_status_t::FEASIBLE;
  } else if (budget_exhausted_) {
    result.status = result.levels_exhausted > 0 ? markshare_status_t::BOUND_ONLY
                                                : markshare_status_t::ABORTED;
  } else {
    result.status = markshare_status_t::BOUND_ONLY;
  }

  CUOPT_LOG_INFO("%s",
                 std::format("Markshare search {}: {} levels exhausted, {} targets, {} nodes, "
                             "{:.2f} s",
                             markshare_status_to_string(result.status),
                             result.levels_exhausted,
                             result.targets,
                             result.nodes,
                             result.search_time)
                   .c_str());
  return result;
}

template <typename i_t, typename f_t>
markshare_result_t<i_t, f_t> markshare_solver_t<i_t, f_t>::try_solve(
  const simplex::user_problem_t<i_t, f_t>& user_problem,
  const markshare_settings_t<i_t, f_t>& settings)
{
  markshare_solver_t<i_t, f_t> solver(user_problem, settings);
  if (!solver.detect()) { return markshare_result_t<i_t, f_t>{}; }
  return solver.solve();
}

#if MIP_INSTANTIATE_FLOAT
template class markshare_solver_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class markshare_solver_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip

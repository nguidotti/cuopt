/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <stdint.h>
#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

// Copied from raft/PCGenerator (rng_device.cuh). We cannot use raft implementation
// on the CPU (.cpp file) since the raft header includes CUDA code.
// The original code is from https://www.pcg-random.org/.
namespace cuopt {
class pcgenerator_t {
 public:
  static constexpr uint64_t default_seed   = 0x853c49e6748fea9bULL;
  static constexpr uint64_t default_stream = 0xda3e39cb94b95bdbULL;

  /**
   * @brief Initializes the PCG generator.
   * @param seed        Generator state seed.
   * @param subsequence Selects one of 2^64 independent subsequences. Use distinct values per
   *                    thread to guarantee non-overlapping streams in parallel contexts.
   * @param offset      Number of outputs to skip ahead before the first draw.
   */
  pcgenerator_t(const uint64_t seed        = default_seed,
                const uint64_t subsequence = default_stream,
                uint64_t offset            = 0)
  {
    set_seed(seed, subsequence, offset);
  }

  /**
   * @brief Re-seeds the generator.
   * @param seed        Generator state seed.
   * @param subsequence Selects one of 2^64 independent subsequences.
   * @param offset      Number of outputs to skip ahead before the first draw.
   */
  void set_seed(uint64_t seed, const uint64_t subsequence = default_stream, uint64_t offset = 0)
  {
    state  = uint64_t(0);
    stream = (subsequence << 1u) | 1u;
    uint32_t discard;
    next(discard);
    state += seed;
    next(discard);
    skipahead(offset);
  }

  /**
   * @brief Advances the generator state by @p offset steps in O(log offset) time.
   *
   * Uses the closed-form LCG jump described in "Random Number Generation with Arbitrary Strides"
   * (F. B. Brown, https://mcnp.lanl.gov/pdf_files/anl-rn-arb-stride.pdf).
   */
  void skipahead(uint64_t offset)
  {
    uint64_t G = 1;
    uint64_t h = 6364136223846793005ULL;
    uint64_t C = 0;
    uint64_t f = stream;
    while (offset) {
      if (offset & 1) {
        G = G * h;
        C = C * h + f;
      }
      f = f * (h + 1);
      h = h * h;
      offset >>= 1;
    }
    state = state * G + C;
  }

  /**
   * @returns the next uniformly distributed 32-bit unsigned integer.
   */
  uint32_t next_u32()
  {
    uint32_t ret;
    uint64_t oldstate   = state;
    state               = oldstate * 6364136223846793005ULL + stream;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot        = oldstate >> 59u;
    ret                 = (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
    return ret;
  }

  /**
   * @returns the next uniformly distributed 64-bit unsigned integer.
   */
  uint64_t next_u64()
  {
    uint64_t ret;
    uint32_t a, b;
    a   = next_u32();
    b   = next_u32();
    ret = uint64_t(a) | (uint64_t(b) << 32);
    return ret;
  }

  /**
   * @returns the next uniformly distributed non-negative 32-bit signed integer in [0,
   * INT32_MAX].
   */
  int32_t next_i32()
  {
    int32_t ret;
    uint32_t val;
    val = next_u32();
    ret = int32_t(val & 0x7fffffff);
    return ret;
  }

  /**
   * @returns the next uniformly distributed non-negative 64-bit signed integer in [0,
   * INT64_MAX].
   */
  int64_t next_i64()
  {
    int64_t ret;
    uint64_t val;
    val = next_u64();
    ret = int64_t(val & 0x7fffffffffffffff);
    return ret;
  }

  /**
   * @returns a uniformly distributed float in [0, 1).
   */
  float next_float() { return (next_u32() >> 8) * 0x1.0p-24; }

  /**
   * @returns a uniformly distributed double in [0, 1).
   */
  double next_double() { return (next_u64() >> 11) * 0x1.0p-53; }

  /**
   * @returns the next random value of type @p T.
   */
  template <typename T>
  T next()
  {
    T val;
    next(val);
    return val;
  }

  void next(uint32_t& ret) { ret = next_u32(); }
  void next(uint64_t& ret) { ret = next_u64(); }
  void next(int32_t& ret) { ret = next_i32(); }
  void next(int64_t& ret) { ret = next_i64(); }
  void next(float& ret) { ret = next_float(); }
  void next(double& ret) { ret = next_double(); }

  /**
   * @brief Draws a sample from a uniform distribution over `[low, high)`.
   *
   * May have a slight bias toward some values due to floating-point scaling.
   */
  template <typename T>
  T uniform(T low, T high)
  {
    double val = next_double();
    T range    = high - low;
    return low + (val * range);
  }

  /**
   * @brief Shuffles @p seq in-place using the Fisher-Yates algorithm.
   */
  template <typename T>
  void shuffle(std::vector<T>& seq)
  {
    if (seq.empty()) { return; }
    for (size_t i = 0; i < seq.size() - 1; ++i) {
      size_t j = uniform(i, seq.size());
      if (j != i) std::swap(seq[i], seq[j]);
    }
  }

  /**
   * @brief Draw a random number from a normal distribution N(mu, sigma)
   * via Box-Muller.
   */
  template <typename T>
  T normal(T mu = T(0), T sigma = T(1))
  {
    T u1 = T{1} - next<T>();  // (0, 1], avoiding log(0).
    T u2 = next<T>();
    T x  = std::sqrt(-2 * std::log(u1)) * std::cos(2 * std::numbers::pi_v<T> * u2);
    return mu + sigma * x;
  }

  /**
   * @brief Draw two random numbers from a normal distribution N(mu, sigma)
   * via Box-Muller.
   */
  template <typename T>
  std::pair<T, T> normal2(T mu = T(0), T sigma = T(1))
  {
    T u1 = T{1} - next<T>();  // (0, 1], avoiding log(0).
    T u2 = next<T>();
    T x  = std::sqrt(-2 * std::log(u1)) * std::cos(2 * std::numbers::pi_v<T> * u2);
    T y  = std::sqrt(-2 * std::log(u1)) * std::sin(2 * std::numbers::pi_v<T> * u2);
    return {mu + sigma * x, mu + sigma * y};
  }

  /**
   * @brief Draw a random number from a split normal distribution:
   * f(x; c, \sigma_L, \sigma_R) =
   * \begin{cases}
   * \sqrt{\frac{2}{\pi (\sigma_L + \sigma_R)^2}} \exp\!\left(-\dfrac{(x-c)^2}{2\sigma_L^2}\right),
   * & L\leq x < c,\\
   * \sqrt{\frac{2}{\pi (\sigma_L + \sigma_R)^2}} \exp\!\left(-\dfrac{(x-c)^2}{2\sigma_R^2}\right),
   * & c\leq x \leq U,
   * \end{cases}
   */
  template <typename T>
  T split_normal(T c, T sigma_r, T sigma_l)
  {
    T u1 = T{1} - next<T>();  // (0, 1], avoiding log(0).
    T u2 = next<T>();
    T z  = std::abs(std::sqrt(-2 * std::log(u1)) * std::cos(2 * std::numbers::pi_v<T> * u2));

    T p_left  = sigma_l / (sigma_l + sigma_r);
    bool left = next<T>() < p_left;
    return left ? c - sigma_l * z : c + sigma_r * z;
  }

 private:
  uint64_t state;
  uint64_t stream;
};
}  // namespace cuopt

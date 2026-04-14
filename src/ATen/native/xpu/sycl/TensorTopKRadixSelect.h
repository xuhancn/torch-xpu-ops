/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// Radix-select TopK implementation for Intel XPU (SYCL).
//
// Algorithm overview:
//   This file implements a radix-select based TopK that avoids full sorting.
//   Instead of sorting all N elements (O(N log N)), it processes the radix
//   bits from MSB to LSB, at each pass counting how many elements fall into
//   each bucket. Elements in "selected" buckets (those entirely within top-K)
//   are emitted immediately; only the "active" bucket (straddling the K
//   boundary) continues to the next radix pass. This achieves O(N * B)
//   complexity where B is the number of radix bits (32 for float).
//
// Three kernel paths handle different (nelements, K) regimes:
//
//   1. RadixTopKKernel (single-tile): For nelements <= 4096. All data fits
//      in one work-group's registers (GROUP_SIZE * KEYS_PER_THREAD elements).
//      Uses packed 2×uint16 counters in shared local memory (SLM) for
//      zero-contention bucket counting with subgroup-shuffle prefix sums.
//
//   2. RadixTopKLargeKernel (multi-tile): For nelements > 4096 when K > 16.
//      Processes data in tiles of PROCESSING_LENGTH (4096), running radix
//      select on each tile to find local top-K, then merging with the next
//      tile until all input is consumed. Requires temp buffers for
//      intermediate top-K results.
//
//   3. HeapScanKernel (heap-select): For nelements > 4096 and K <= 16.
//      Each thread scans a contiguous block of input, maintaining a sorted
//      K-element array in registers (insertion sort). The GS*K candidates
//      are then reduced via a single-tile radix select. This avoids the
//      multi-tile merge overhead for the common LLM case (large vocab, tiny K).
//
// Key data structures:
//   - RadixTraits<T>: Converts floating-point/integer types to unsigned
//     integers that sort in the same order (IEEE 754 bit-flip trick).
//   - aligned_vector<T, N>: Enables vectorized 4-wide memory loads.
//   - RankStorage: Union of packed counters for ranking and exchange buffers
//     for compacting active elements between radix passes.

#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdint>
#include <limits>

namespace at {
namespace native {
namespace xpu {

// ======================== Utilities ========================

// Compile-time log2 for power-of-two constants.
template <int N>
struct Log2 {
  static constexpr int VALUE = 1 + Log2<N / 2>::VALUE;
};
template <>
struct Log2<1> {
  static constexpr int VALUE = 0;
};

// Aligned vector type for vectorized 4-wide loads/stores.
// alignas ensures the compiler can emit single wide load instructions.
template <typename T, int N>
struct alignas(sizeof(T) * N) aligned_vector {
  T val[N];
};

// ======================== Key Traits ========================
// RadixTraits converts each scalar type to an unsigned integer representation
// that preserves sort order. For floating-point types, this uses the IEEE 754
// bit-flip trick: flip all bits if negative, else flip only the sign bit.
// This maps the floating-point total order to unsigned integer order.
// For integer types, simply flip the sign bit to convert from signed to
// unsigned order.

template <typename T>
struct RadixTraits;

template <>
struct RadixTraits<float> {
  using UintType = uint32_t;
  static constexpr int BITS = 32;
  static inline UintType to_radix(float v) {
    UintType x;
    std::memcpy(&x, &v, sizeof(x));
    UintType mask = (x & 0x80000000u) ? 0xffffffffu : 0x80000000u;
    return x ^ mask;
  }
  static inline float from_radix(UintType v) {
    UintType mask = (v & 0x80000000u) ? 0x80000000u : 0xffffffffu;
    v = v ^ mask;
    float r;
    std::memcpy(&r, &v, sizeof(r));
    return r;
  }
  static inline UintType padding_key(bool descending) {
    return descending ? UintType(0) : ~UintType(0);
  }
};

template <>
struct RadixTraits<double> {
  using UintType = uint64_t;
  static constexpr int BITS = 64;
  static inline UintType to_radix(double v) {
    UintType x;
    std::memcpy(&x, &v, sizeof(x));
    UintType mask = (x >> 63) ? ~UintType(0) : (UintType(1) << 63);
    return x ^ mask;
  }
  static inline double from_radix(UintType v) {
    UintType mask = (v >> 63) ? (UintType(1) << 63) : ~UintType(0);
    v = v ^ mask;
    double r;
    std::memcpy(&r, &v, sizeof(r));
    return r;
  }
  static inline UintType padding_key(bool descending) {
    return descending ? UintType(0) : ~UintType(0);
  }
};

template <>
struct RadixTraits<at::Half> {
  using UintType = uint16_t;
  static constexpr int BITS = 16;
  static inline UintType to_radix(at::Half v) {
    UintType x;
    std::memcpy(&x, &v, sizeof(x));
    UintType mask = (x & 0x8000u) ? 0xffffu : 0x8000u;
    return x ^ mask;
  }
  static inline at::Half from_radix(UintType v) {
    UintType mask = (v & 0x8000u) ? 0x8000u : 0xffffu;
    v = v ^ mask;
    at::Half r;
    std::memcpy(&r, &v, sizeof(r));
    return r;
  }
  static inline UintType padding_key(bool descending) {
    return descending ? UintType(0) : ~UintType(0);
  }
};

template <>
struct RadixTraits<at::BFloat16> {
  using UintType = uint16_t;
  static constexpr int BITS = 16;
  static inline UintType to_radix(at::BFloat16 v) {
    UintType x;
    std::memcpy(&x, &v, sizeof(x));
    UintType mask = (x & 0x8000u) ? 0xffffu : 0x8000u;
    return x ^ mask;
  }
  static inline at::BFloat16 from_radix(UintType v) {
    UintType mask = (v & 0x8000u) ? 0x8000u : 0xffffu;
    v = v ^ mask;
    at::BFloat16 r;
    std::memcpy(&r, &v, sizeof(r));
    return r;
  }
  static inline UintType padding_key(bool descending) {
    return descending ? UintType(0) : ~UintType(0);
  }
};

template <>
struct RadixTraits<int32_t> {
  using UintType = uint32_t;
  static constexpr int BITS = 32;
  static inline UintType to_radix(int32_t v) {
    return static_cast<UintType>(v) ^ (UintType(1) << 31);
  }
  static inline int32_t from_radix(UintType v) {
    return static_cast<int32_t>(v ^ (UintType(1) << 31));
  }
  static inline UintType padding_key(bool descending) {
    return descending ? UintType(0) : ~UintType(0);
  }
};

template <>
struct RadixTraits<int64_t> {
  using UintType = uint64_t;
  static constexpr int BITS = 64;
  static inline UintType to_radix(int64_t v) {
    return static_cast<UintType>(v) ^ (UintType(1) << 63);
  }
  static inline int64_t from_radix(UintType v) {
    return static_cast<int64_t>(v ^ (UintType(1) << 63));
  }
  static inline UintType padding_key(bool descending) {
    return descending ? UintType(0) : ~UintType(0);
  }
};

template <>
struct RadixTraits<int16_t> {
  using UintType = uint16_t;
  static constexpr int BITS = 16;
  static inline UintType to_radix(int16_t v) {
    return static_cast<UintType>(v) ^ (UintType(1) << 15);
  }
  static inline int16_t from_radix(UintType v) {
    return static_cast<int16_t>(v ^ (UintType(1) << 15));
  }
  static inline UintType padding_key(bool descending) {
    return descending ? UintType(0) : ~UintType(0);
  }
};

template <>
struct RadixTraits<int8_t> {
  using UintType = uint8_t;
  static constexpr int BITS = 8;
  static inline UintType to_radix(int8_t v) {
    return static_cast<UintType>(v) ^ (UintType(1) << 7);
  }
  static inline int8_t from_radix(UintType v) {
    return static_cast<int8_t>(v ^ (UintType(1) << 7));
  }
  static inline UintType padding_key(bool descending) {
    return descending ? UintType(0) : ~UintType(0);
  }
};

template <>
struct RadixTraits<uint8_t> {
  using UintType = uint8_t;
  static constexpr int BITS = 8;
  static inline UintType to_radix(uint8_t v) {
    return v;
  }
  static inline uint8_t from_radix(UintType v) {
    return v;
  }
  static inline UintType padding_key(bool descending) {
    return descending ? UintType(0) : ~UintType(0);
  }
};

// ======================== Subgroup prefix sum ========================
// Computes inclusive and exclusive prefix sums within a subgroup using
// shuffle operations. This is used for the packed-counter ranking step
// to compute global offsets without atomics.

template <typename T, int STEPS>
inline void subgroup_cumsum(
    sycl::sub_group& sg,
    int sgid,
    T input,
    T& inclusive_sum,
    T& exclusive_sum) {
  inclusive_sum = input;
#pragma unroll
  for (int i = 0, offset = 1; i < STEPS; ++i, offset <<= 1) {
    T temp = sycl::shift_group_right(sg, inclusive_sum, offset);
    if (sgid >= offset)
      inclusive_sum += temp;
  }
  exclusive_sum = inclusive_sum - input;
}

// Group-level exclusive prefix sum over packed counters in SLM.
// Three-phase algorithm:
//   1. Each thread sums its COUNTER_LANES values (intra-thread scan)
//   2. Subgroup-level prefix sum via shuffle (inter-thread within subgroup)
//   3. Sequential scan across subgroup totals (inter-subgroup)
// Returns the total sum across all threads (used to determine bucket sizes).
template <typename T, int COUNTER_LANES, int GROUP_SIZE, int SUBGROUP_SIZE>
inline T group_exclusive_cumsum(T* storage, sycl::nd_item<1>& item) {
  static constexpr int NUM_SUBGROUPS = GROUP_SIZE / SUBGROUP_SIZE;
  static constexpr int SUBGROUP_SCAN_STEPS = Log2<SUBGROUP_SIZE>::VALUE;

  int lid = item.get_local_linear_id();
  auto sg = item.get_sub_group();
  int subgroup_local_id = sg.get_local_id()[0];
  int subgroup_id = sg.get_group_id()[0];
  int lane_temp_values[COUNTER_LANES];

  auto storage_lanes = storage + lid * COUNTER_LANES;
  T lane_all_sum = 0;
#pragma unroll
  for (int lane = 0; lane < COUNTER_LANES; ++lane) {
    lane_temp_values[lane] = lane_all_sum;
    lane_all_sum += storage_lanes[lane];
  }

  T subgroup_inclusive_sum, subgroup_exclusive_sum;
  subgroup_cumsum<T, SUBGROUP_SCAN_STEPS>(
      sg,
      subgroup_local_id,
      lane_all_sum,
      subgroup_inclusive_sum,
      subgroup_exclusive_sum);
  sycl::group_barrier(item.get_group());

  if (subgroup_local_id == (SUBGROUP_SIZE - 1))
    storage[subgroup_id] = subgroup_inclusive_sum;
  sycl::group_barrier(item.get_group());

  T group_all_sum = 0, group_exclusive_sum = 0;
#pragma unroll
  for (int i = 0; i < NUM_SUBGROUPS; ++i) {
    if (subgroup_id == i)
      group_exclusive_sum = group_all_sum;
    group_all_sum += storage[i];
  }
  sycl::group_barrier(item.get_group());

  subgroup_exclusive_sum += group_exclusive_sum;
#pragma unroll
  for (int lane = 0; lane < COUNTER_LANES; ++lane) {
    storage_lanes[lane] = subgroup_exclusive_sum + lane_temp_values[lane];
  }
  sycl::group_barrier(item.get_group());

  return group_all_sum;
}

// ======================== Radix TopK Kernel (single tile)
// ========================
//
// Single work-group radix select for nelements <= PROCESSING_LENGTH (4096).
// Each work-group handles one "segment" (one row of the input tensor).
//
// Algorithm per radix pass (4 bits at a time, MSB to LSB):
//   1. Each thread counts its elements into 16 radix buckets using packed
//      2×uint16-in-uint32 counters (RankStorage). This packing halves
//      SLM usage and eliminates bank conflicts.
//   2. Exclusive prefix sum over all counters gives global ranks.
//   3. Scan bucket totals to find the "pivot bucket" — the one that
//      straddles the K boundary. Elements in buckets before the pivot
//      are immediately written to output (they're in the top-K).
//   4. Elements in the pivot bucket are compacted via SLM exchange for
//      the next radix pass. Elements in buckets after the pivot are
//      discarded (not in top-K).
//   5. Repeat until all radix bits are processed or K elements are found.

template <
    typename scalar_t,
    bool IS_DESCENDING,
    int GROUP_SIZE,
    int KEYS_PER_THREAD,
    int SUBGROUP_SIZE = 32>
struct RadixTopKKernel {
  using Traits = RadixTraits<scalar_t>;
  using UintType = typename Traits::UintType;
  using DigitT = uint16_t;
  using CounterT = uint32_t;

  static constexpr int RADIX_BITS = 4; // Process 4 bits per pass (16 buckets)
  static constexpr int RADIX_BUCKETS = 1 << RADIX_BITS;
  static constexpr int PROCESSING_LENGTH = GROUP_SIZE * KEYS_PER_THREAD;
  // Pack 2 × uint16 counters into each uint32 to halve SLM usage
  static constexpr int PACKING_RATIO = sizeof(CounterT) / sizeof(DigitT);
  static constexpr int COUNTER_LANES = RADIX_BUCKETS / PACKING_RATIO;
  static constexpr int LOG_COUNTER_LANES = Log2<COUNTER_LANES>::VALUE;
  static constexpr int DIGIT_BITS = sizeof(DigitT) * 8;
  static constexpr int DIGIT_MASK = (1 << DIGIT_BITS) - 1;

  // Shared local memory layout (union to reuse SLM across phases):
  //   rank_storage: packed counters for radix ranking
  //   exchange_keys/vals: compaction buffers between radix passes
  union RankStorage {
    CounterT counters[COUNTER_LANES][GROUP_SIZE];
    CounterT counters_flat[COUNTER_LANES * GROUP_SIZE];
    DigitT buckets[COUNTER_LANES][GROUP_SIZE][PACKING_RATIO];
  };

  union LocalMem {
    RankStorage rank_storage;
    UintType exchange_keys[PROCESSING_LENGTH];
    int64_t exchange_vals[PROCESSING_LENGTH];
  };

  static int local_mem_size() {
    return sizeof(LocalMem);
  }

  // encode_key: Convert scalar to unsigned int for ascending radix sort.
  // For descending order (IS_DESCENDING=true), bit-flip so that ascending
  // radix sort produces descending scalar order.
  static inline UintType encode_key(scalar_t v) {
    UintType u = Traits::to_radix(v);
    return IS_DESCENDING ? ~u : u;
  }
  // decode_key: Inverse of encode_key, converts back to scalar_t.
  static inline scalar_t decode_key(UintType u) {
    return Traits::from_radix(IS_DESCENDING ? ~u : u);
  }
  // padding_key: Value for out-of-bounds elements. Uses max unsigned value
  // so padding sorts to the end (never selected as top-K).
  static inline UintType padding_key() {
    return ~UintType(0);
  }

  const scalar_t* __restrict__ input_;
  scalar_t* __restrict__ values_out_;
  int64_t* __restrict__ indices_out_;
  int nelements_;
  int k_;

  RadixTopKKernel(
      const scalar_t* input,
      scalar_t* values_out,
      int64_t* indices_out,
      int nelements,
      int k)
      : input_(input),
        values_out_(values_out),
        indices_out_(indices_out),
        nelements_(nelements),
        k_(k) {}

  // rank_keys_topk: Core radix ranking step.
  //
  // For each active element, determines its rank (position) within the
  // current radix pass. Uses packed per-thread counters to avoid atomics:
  //   1. Each thread writes its elements' digits into packed counters
  //   2. Group-wide exclusive prefix sum computes global offsets
  //   3. Unpacks the packed counters to get per-element ranks
  //   4. Scans bucket totals (step-outer, lane-inner order) to find:
  //      - out_offset_select: number of elements in "selected" buckets
  //        (entirely within top-K, written to output)
  //      - out_offset_active: total elements in selected + pivot buckets
  //        (pivot bucket continues to next pass)
  static inline void rank_keys_topk(
      sycl::nd_item<1>& item,
      LocalMem& lm,
      int lid,
      UintType* ukeys,
      int* ranks,
      uint32_t active_mask,
      int begin_bit,
      int pass_bits,
      int num_to_select,
      int* out_offset_select,
      int* out_offset_active) {
    DigitT* digit_counters[KEYS_PER_THREAD];

#pragma unroll
    for (int lane = 0; lane < COUNTER_LANES; ++lane) {
      lm.rank_storage.counters[lane][lid] = 0;
    }
    sycl::group_barrier(item.get_group());

#pragma unroll
    for (int i = 0; i < KEYS_PER_THREAD; ++i) {
      ranks[i] = PROCESSING_LENGTH;
      if ((active_mask >> i) & 1) {
        int digit = (ukeys[i] >> begin_bit) & ((1 << pass_bits) - 1);
        int sub_counter = digit >> LOG_COUNTER_LANES;
        int counter_lane = digit & (COUNTER_LANES - 1);
        digit_counters[i] =
            &lm.rank_storage.buckets[counter_lane][lid][sub_counter];
        ranks[i] = *digit_counters[i];
        *digit_counters[i] = ranks[i] + 1;
      }
    }
    sycl::group_barrier(item.get_group());

    CounterT exclusive = group_exclusive_cumsum<
        CounterT,
        COUNTER_LANES,
        GROUP_SIZE,
        SUBGROUP_SIZE>(lm.rank_storage.counters_flat, item);

    int carry = 0;
#pragma unroll
    for (int step = 0; step < PACKING_RATIO; ++step) {
      DigitT cc = (exclusive >> (step * DIGIT_BITS)) & DIGIT_MASK;
      carry += cc;
    }

    CounterT c = 0;
#pragma unroll
    for (int step = 1; step < PACKING_RATIO; ++step) {
      exclusive = exclusive << DIGIT_BITS;
      c += exclusive;
    }
#pragma unroll
    for (int lane = 0; lane < COUNTER_LANES; ++lane) {
      lm.rank_storage.counters[lane][lid] += c;
    }
    sycl::group_barrier(item.get_group());

#pragma unroll
    for (int i = 0; i < KEYS_PER_THREAD; ++i) {
      if ((active_mask >> i) & 1) {
        ranks[i] += *digit_counters[i];
      }
    }
    sycl::group_barrier(item.get_group());

    // Scan bucket totals: step outer, lane inner
    *out_offset_select = 0;
    *out_offset_active = 0;
    int carry_last = 0;
    bool found = false;
#pragma unroll
    for (int step = 0; step < PACKING_RATIO; ++step) {
#pragma unroll
      for (int lane = 0; lane < COUNTER_LANES; ++lane) {
        if (!found) {
          int count = (int)(lm.rank_storage.buckets[lane][0][step]);
          if (count > num_to_select) {
            *out_offset_active = count;
            *out_offset_select = carry_last;
            found = true;
          }
          carry_last = count;
        }
      }
    }
    if (!found) {
      *out_offset_select = carry_last;
      *out_offset_active = carry;
    }
    sycl::group_barrier(item.get_group());
  }

  void operator()(sycl::nd_item<1> item, char* slm_raw) const {
    auto& lm = *reinterpret_cast<LocalMem*>(slm_raw);
    const int lid = item.get_local_id(0);
    const int seg_idx = item.get_group(0);
    const scalar_t* seg_in =
        input_ + seg_idx * static_cast<int64_t>(nelements_);
    scalar_t* seg_vals_out = values_out_ + seg_idx * static_cast<int64_t>(k_);
    int64_t* seg_inds_out = indices_out_ + seg_idx * static_cast<int64_t>(k_);

    UintType ukeys[KEYS_PER_THREAD];
    int64_t indices[KEYS_PER_THREAD];
    int ranks[KEYS_PER_THREAD];

    // Vectorized initial load: use 4-wide aligned loads when possible,
    // fall back to scalar loads for unaligned or short segments.
    {
      int base = lid * KEYS_PER_THREAD;
      if constexpr (KEYS_PER_THREAD == 4) {
        if (base + 3 < nelements_ &&
            (reinterpret_cast<uintptr_t>(seg_in + base) %
                 (sizeof(scalar_t) * 4) ==
             0)) {
          using vec_t = aligned_vector<scalar_t, 4>;
          vec_t v = *reinterpret_cast<const vec_t*>(seg_in + base);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            ukeys[i] = encode_key(v.val[i]);
            indices[i] = base + i;
          }
        } else {
#pragma unroll
          for (int i = 0; i < KEYS_PER_THREAD; ++i) {
            int offset = base + i;
            if (offset < nelements_) {
              ukeys[i] = encode_key(seg_in[offset]);
              indices[i] = offset;
            } else {
              ukeys[i] = padding_key();
              indices[i] = -1;
            }
          }
        }
      } else {
#pragma unroll
        for (int i = 0; i < KEYS_PER_THREAD; ++i) {
          int offset = base + i;
          if (offset < nelements_) {
            ukeys[i] = encode_key(seg_in[offset]);
            indices[i] = offset;
          } else {
            ukeys[i] = padding_key();
            indices[i] = -1;
          }
        }
      }
    }

    uint32_t active_mask = 0;
#pragma unroll
    for (int i = 0; i < KEYS_PER_THREAD; ++i) {
      int offset = lid * KEYS_PER_THREAD + i;
      if (offset < nelements_)
        active_mask |= (1u << i);
    }

    int num_selected = 0;
    int begin_bit = Traits::BITS;

    while (true) {
      int pass_bits = begin_bit;
      if (pass_bits > RADIX_BITS)
        pass_bits = RADIX_BITS;
      begin_bit -= pass_bits;

      int offset_select, offset_active;
      rank_keys_topk(
          item,
          lm,
          lid,
          ukeys,
          ranks,
          active_mask,
          begin_bit,
          pass_bits,
          k_ - num_selected,
          &offset_select,
          &offset_active);

      if (begin_bit == 0)
        offset_select = k_ - num_selected;

      if (offset_select > 0) {
#pragma unroll
        for (int i = 0; i < KEYS_PER_THREAD; ++i) {
          if (ranks[i] < offset_select) {
            seg_vals_out[num_selected + ranks[i]] = decode_key(ukeys[i]);
            seg_inds_out[num_selected + ranks[i]] = indices[i];
          }
        }
      }
      num_selected += offset_select;
      if (num_selected == k_)
        break;

// Exchange: compact active-bucket elements
#pragma unroll
      for (int i = 0; i < KEYS_PER_THREAD; ++i) {
        if (ranks[i] >= offset_select && ranks[i] < offset_active) {
          lm.exchange_keys[ranks[i] - offset_select] = ukeys[i];
        }
      }
      sycl::group_barrier(item.get_group());

      active_mask = 0u;
      int new_length = offset_active - offset_select;
#pragma unroll
      for (int i = 0; i < KEYS_PER_THREAD; ++i) {
        int offset = lid * KEYS_PER_THREAD + i;
        if (offset < new_length) {
          active_mask |= (1u << i);
          ukeys[i] = lm.exchange_keys[offset];
        } else {
          ukeys[i] = padding_key();
        }
      }
      sycl::group_barrier(item.get_group());

#pragma unroll
      for (int i = 0; i < KEYS_PER_THREAD; ++i) {
        if (ranks[i] >= offset_select && ranks[i] < offset_active) {
          lm.exchange_vals[ranks[i] - offset_select] = indices[i];
        }
      }
      sycl::group_barrier(item.get_group());

#pragma unroll
      for (int i = 0; i < KEYS_PER_THREAD; ++i) {
        int offset = lid * KEYS_PER_THREAD + i;
        if (offset < new_length) {
          indices[i] = lm.exchange_vals[offset];
        } else {
          indices[i] = -1;
        }
      }
      sycl::group_barrier(item.get_group());
    }
  }
};

// ======================== Multi-tile Radix TopK ========================
//
// For nelements > PROCESSING_LENGTH (4096) and K > 16.
// Processes input in tiles of PROCESSING_LENGTH elements:
//   1. Load first tile into registers
//   2. Run radix select to find top-K within the tile
//   3. Store top-K to temp buffers
//   4. Merge top-K with next tile of input (top-K occupies first K slots,
//      new elements fill remaining PROCESSING_LENGTH - K slots)
//   5. Repeat until all input consumed
//   6. Final radix select on the last merged tile writes to output
//
// This reuses RadixTopKKernel's rank_keys_topk for each tile's radix passes.

template <
    typename scalar_t,
    bool IS_DESCENDING,
    int GROUP_SIZE,
    int KEYS_PER_THREAD,
    int SUBGROUP_SIZE = 32>
struct RadixTopKLargeKernel {
  using Traits = RadixTraits<scalar_t>;
  using UintType = typename Traits::UintType;
  using SmallKernel = RadixTopKKernel<
      scalar_t,
      IS_DESCENDING,
      GROUP_SIZE,
      KEYS_PER_THREAD,
      SUBGROUP_SIZE>;
  using DigitT = typename SmallKernel::DigitT;
  using CounterT = typename SmallKernel::CounterT;
  using LocalMem = typename SmallKernel::LocalMem;

  static constexpr int PROCESSING_LENGTH = GROUP_SIZE * KEYS_PER_THREAD;
  static int local_mem_size() {
    return SmallKernel::local_mem_size();
  }
  static inline UintType encode_key(scalar_t v) {
    return SmallKernel::encode_key(v);
  }
  static inline scalar_t decode_key(UintType u) {
    return SmallKernel::decode_key(u);
  }
  static inline UintType padding_key() {
    return SmallKernel::padding_key();
  }

  const scalar_t* __restrict__ input_;
  scalar_t* __restrict__ values_out_;
  int64_t* __restrict__ indices_out_;
  int nelements_;
  int k_;
  scalar_t* __restrict__ temp_keys_;
  int64_t* __restrict__ temp_indices_;

  RadixTopKLargeKernel(
      const scalar_t* input,
      scalar_t* values_out,
      int64_t* indices_out,
      int nelements,
      int k,
      scalar_t* temp_keys,
      int64_t* temp_indices)
      : input_(input),
        values_out_(values_out),
        indices_out_(indices_out),
        nelements_(nelements),
        k_(k),
        temp_keys_(temp_keys),
        temp_indices_(temp_indices) {}

  void operator()(sycl::nd_item<1> item, char* slm_raw) const {
    auto& lm = *reinterpret_cast<LocalMem*>(slm_raw);
    const int lid = item.get_local_id(0);
    const int seg_idx = item.get_group(0);
    const scalar_t* seg_in =
        input_ + seg_idx * static_cast<int64_t>(nelements_);
    scalar_t* seg_vals_out = values_out_ + seg_idx * static_cast<int64_t>(k_);
    int64_t* seg_inds_out = indices_out_ + seg_idx * static_cast<int64_t>(k_);
    scalar_t* my_temp_keys =
        temp_keys_ + seg_idx * static_cast<int64_t>(PROCESSING_LENGTH);
    int64_t* my_temp_inds =
        temp_indices_ + seg_idx * static_cast<int64_t>(PROCESSING_LENGTH);

    UintType ukeys[KEYS_PER_THREAD];
    int64_t indices[KEYS_PER_THREAD];
    int ranks[KEYS_PER_THREAD];

    // Load first tile (vectorized when aligned)
    int tile_size =
        nelements_ < PROCESSING_LENGTH ? nelements_ : PROCESSING_LENGTH;
    {
      int base = lid * KEYS_PER_THREAD;
      if constexpr (KEYS_PER_THREAD == 4) {
        if (base + 3 < tile_size &&
            (reinterpret_cast<uintptr_t>(seg_in + base) %
                 (sizeof(scalar_t) * 4) ==
             0)) {
          using vec_t = aligned_vector<scalar_t, 4>;
          vec_t v = *reinterpret_cast<const vec_t*>(seg_in + base);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            ukeys[i] = encode_key(v.val[i]);
            indices[i] = base + i;
          }
        } else {
#pragma unroll
          for (int i = 0; i < KEYS_PER_THREAD; ++i) {
            int offset = base + i;
            if (offset < tile_size) {
              ukeys[i] = encode_key(seg_in[offset]);
              indices[i] = offset;
            } else {
              ukeys[i] = padding_key();
              indices[i] = -1;
            }
          }
        }
      } else {
#pragma unroll
        for (int i = 0; i < KEYS_PER_THREAD; ++i) {
          int offset = base + i;
          if (offset < tile_size) {
            ukeys[i] = encode_key(seg_in[offset]);
            indices[i] = offset;
          } else {
            ukeys[i] = padding_key();
            indices[i] = -1;
          }
        }
      }
    }
    int input_consumed = tile_size;

    // Process subsequent tiles: topk current -> merge with next chunk
    while (input_consumed < nelements_) {
      uint32_t active_mask = 0;
#pragma unroll
      for (int i = 0; i < KEYS_PER_THREAD; ++i) {
        int offset = lid * KEYS_PER_THREAD + i;
        if (offset < (input_consumed < PROCESSING_LENGTH ? input_consumed
                                                         : PROCESSING_LENGTH))
          active_mask |= (1u << i);
      }

      int num_selected = 0;
      int begin_bit = Traits::BITS;
      while (true) {
        int pass_bits = begin_bit;
        if (pass_bits > SmallKernel::RADIX_BITS)
          pass_bits = SmallKernel::RADIX_BITS;
        begin_bit -= pass_bits;

        int offset_select, offset_active;
        SmallKernel::rank_keys_topk(
            item,
            lm,
            lid,
            ukeys,
            ranks,
            active_mask,
            begin_bit,
            pass_bits,
            k_ - num_selected,
            &offset_select,
            &offset_active);

        if (begin_bit == 0)
          offset_select = k_ - num_selected;

        if (offset_select > 0) {
#pragma unroll
          for (int i = 0; i < KEYS_PER_THREAD; ++i) {
            if (ranks[i] < offset_select) {
              my_temp_keys[num_selected + ranks[i]] = decode_key(ukeys[i]);
              my_temp_inds[num_selected + ranks[i]] = indices[i];
            }
          }
        }
        num_selected += offset_select;
        if (num_selected == k_)
          break;

// Exchange keys
#pragma unroll
        for (int i = 0; i < KEYS_PER_THREAD; ++i) {
          if (ranks[i] >= offset_select && ranks[i] < offset_active)
            lm.exchange_keys[ranks[i] - offset_select] = ukeys[i];
        }
        sycl::group_barrier(item.get_group());
        active_mask = 0u;
        int new_length = offset_active - offset_select;
#pragma unroll
        for (int i = 0; i < KEYS_PER_THREAD; ++i) {
          int off = lid * KEYS_PER_THREAD + i;
          if (off < new_length) {
            active_mask |= (1u << i);
            ukeys[i] = lm.exchange_keys[off];
          } else {
            ukeys[i] = padding_key();
          }
        }
        sycl::group_barrier(item.get_group());

// Exchange indices
#pragma unroll
        for (int i = 0; i < KEYS_PER_THREAD; ++i) {
          if (ranks[i] >= offset_select && ranks[i] < offset_active)
            lm.exchange_vals[ranks[i] - offset_select] = indices[i];
        }
        sycl::group_barrier(item.get_group());
#pragma unroll
        for (int i = 0; i < KEYS_PER_THREAD; ++i) {
          int off = lid * KEYS_PER_THREAD + i;
          if (off < new_length)
            indices[i] = lm.exchange_vals[off];
          else
            indices[i] = -1;
        }
        sycl::group_barrier(item.get_group());
      }

      // Merge: load top-k from temp + new elements from input
      int new_slots = PROCESSING_LENGTH - k_;
      int remaining_input = nelements_ - input_consumed;
      int new_elements =
          remaining_input < new_slots ? remaining_input : new_slots;
      int merge_size = k_ + new_elements;

#pragma unroll
      for (int i = 0; i < KEYS_PER_THREAD; ++i) {
        int offset = lid * KEYS_PER_THREAD + i;
        if (offset < k_) {
          ukeys[i] = encode_key(my_temp_keys[offset]);
          indices[i] = my_temp_inds[offset];
        } else if (offset < merge_size) {
          int src = input_consumed + (offset - k_);
          ukeys[i] = encode_key(seg_in[src]);
          indices[i] = src;
        } else {
          ukeys[i] = padding_key();
          indices[i] = -1;
        }
      }
      input_consumed += new_elements;
    }

    // Final radix select
    uint32_t active_mask = 0;
#pragma unroll
    for (int i = 0; i < KEYS_PER_THREAD; ++i) {
      int offset = lid * KEYS_PER_THREAD + i;
      if (offset <
          (nelements_ < PROCESSING_LENGTH ? nelements_ : PROCESSING_LENGTH))
        active_mask |= (1u << i);
    }

    int num_selected = 0;
    int begin_bit = Traits::BITS;
    while (true) {
      int pass_bits = begin_bit;
      if (pass_bits > SmallKernel::RADIX_BITS)
        pass_bits = SmallKernel::RADIX_BITS;
      begin_bit -= pass_bits;

      int offset_select, offset_active;
      SmallKernel::rank_keys_topk(
          item,
          lm,
          lid,
          ukeys,
          ranks,
          active_mask,
          begin_bit,
          pass_bits,
          k_ - num_selected,
          &offset_select,
          &offset_active);

      if (begin_bit == 0)
        offset_select = k_ - num_selected;

      if (offset_select > 0) {
#pragma unroll
        for (int i = 0; i < KEYS_PER_THREAD; ++i) {
          if (ranks[i] < offset_select) {
            seg_vals_out[num_selected + ranks[i]] = decode_key(ukeys[i]);
            seg_inds_out[num_selected + ranks[i]] = indices[i];
          }
        }
      }
      num_selected += offset_select;
      if (num_selected == k_)
        break;

#pragma unroll
      for (int i = 0; i < KEYS_PER_THREAD; ++i) {
        if (ranks[i] >= offset_select && ranks[i] < offset_active)
          lm.exchange_keys[ranks[i] - offset_select] = ukeys[i];
      }
      sycl::group_barrier(item.get_group());
      active_mask = 0u;
      int new_length = offset_active - offset_select;
#pragma unroll
      for (int i = 0; i < KEYS_PER_THREAD; ++i) {
        int off = lid * KEYS_PER_THREAD + i;
        if (off < new_length) {
          active_mask |= (1u << i);
          ukeys[i] = lm.exchange_keys[off];
        } else {
          ukeys[i] = padding_key();
        }
      }
      sycl::group_barrier(item.get_group());

#pragma unroll
      for (int i = 0; i < KEYS_PER_THREAD; ++i) {
        if (ranks[i] >= offset_select && ranks[i] < offset_active)
          lm.exchange_vals[ranks[i] - offset_select] = indices[i];
      }
      sycl::group_barrier(item.get_group());
#pragma unroll
      for (int i = 0; i < KEYS_PER_THREAD; ++i) {
        int off = lid * KEYS_PER_THREAD + i;
        if (off < new_length)
          indices[i] = lm.exchange_vals[off];
        else
          indices[i] = -1;
      }
      sycl::group_barrier(item.get_group());
    }
  }
};

// ======================== Heap Scan Kernel ========================
//
// Two-phase approach for K <= 16 with large dimensions:
//
// Phase 1 (this kernel): Each thread scans a contiguous block of
//   nelements/GROUP_SIZE input values, maintaining a sorted K-element
//   array in registers via insertion sort. When a new value beats the
//   current K-th best, it's inserted in sorted position. This is
//   efficient because K is tiny (<=16) so the insertion is just a few
//   register shifts.
//
//   Output: GROUP_SIZE * K candidates in temp buffers (one K-element
//   sorted list per thread).
//
// Phase 2 (caller invokes RadixTopKKernel on reduced data): A single-tile
//   radix select on the GROUP_SIZE * K candidates to find the global top-K.
//   Since GROUP_SIZE * K <= 16384, this fits in one tile.
//
// This avoids the multi-tile merge overhead and is optimal for the common
// LLM use case (large vocab dimension, K=1..16 for beam search/sampling).

template <typename scalar_t, bool IS_DESCENDING, int GROUP_SIZE, int MAX_K = 16>
struct HeapScanKernel {
  using Traits = RadixTraits<scalar_t>;
  using UintType = typename Traits::UintType;

  static inline UintType encode_key(scalar_t v) {
    UintType u = Traits::to_radix(v);
    return IS_DESCENDING ? ~u : u;
  }
  static inline scalar_t decode_key(UintType u) {
    return Traits::from_radix(IS_DESCENDING ? ~u : u);
  }

  const scalar_t* __restrict__ input_;
  scalar_t* __restrict__ temp_keys_;
  int64_t* __restrict__ temp_indices_;
  int nelements_;
  int k_;

  HeapScanKernel(
      const scalar_t* input,
      scalar_t* temp_keys,
      int64_t* temp_indices,
      int nelements,
      int k)
      : input_(input),
        temp_keys_(temp_keys),
        temp_indices_(temp_indices),
        nelements_(nelements),
        k_(k) {}

  void operator()(sycl::nd_item<1> item) const {
    const int lid = item.get_local_id(0);
    const int seg_idx = item.get_group(0);
    const scalar_t* seg_in =
        input_ + seg_idx * static_cast<int64_t>(nelements_);

    const int reduced_size = GROUP_SIZE * k_;
    scalar_t* seg_temp_keys =
        temp_keys_ + seg_idx * static_cast<int64_t>(reduced_size);
    int64_t* seg_temp_inds =
        temp_indices_ + seg_idx * static_cast<int64_t>(reduced_size);

    // Thread-local top-K sorted ascending by encoded key
    UintType top_keys[MAX_K];
    int64_t top_indices[MAX_K];
    const UintType PAD = ~UintType(0);
#pragma unroll
    for (int i = 0; i < MAX_K; ++i) {
      top_keys[i] = PAD;
      top_indices[i] = -1;
    }

    const int k = k_;

    // Blocked access: each thread processes a contiguous chunk
    const int block_size = (nelements_ + GROUP_SIZE - 1) / GROUP_SIZE;
    int start = lid * block_size;
    int end = start + block_size;
    if (end > nelements_)
      end = nelements_;

    int i = start;

    // Vec4 scan for types >= 2 bytes
    if constexpr (sizeof(scalar_t) >= 2) {
      constexpr int VEC = 4;
      using vec_t = aligned_vector<scalar_t, VEC>;
      constexpr size_t ALIGN_MASK = sizeof(scalar_t) * VEC - 1;

      // Scalar prefix until aligned
      for (; i < end &&
           (reinterpret_cast<uintptr_t>(seg_in + i) & ALIGN_MASK) != 0;
           ++i) {
        UintType key = encode_key(seg_in[i]);
        if (key < top_keys[k - 1]) {
          int pos = k - 1;
          while (pos > 0 && key < top_keys[pos - 1]) {
            top_keys[pos] = top_keys[pos - 1];
            top_indices[pos] = top_indices[pos - 1];
            --pos;
          }
          top_keys[pos] = key;
          top_indices[pos] = i;
        }
      }

      // Vec4 main loop
      for (; i + VEC - 1 < end; i += VEC) {
        vec_t v = *reinterpret_cast<const vec_t*>(seg_in + i);
#pragma unroll
        for (int j = 0; j < VEC; ++j) {
          UintType key = encode_key(v.val[j]);
          if (key < top_keys[k - 1]) {
            int pos = k - 1;
            while (pos > 0 && key < top_keys[pos - 1]) {
              top_keys[pos] = top_keys[pos - 1];
              top_indices[pos] = top_indices[pos - 1];
              --pos;
            }
            top_keys[pos] = key;
            top_indices[pos] = i + j;
          }
        }
      }
    }

    // Scalar tail
    for (; i < end; ++i) {
      UintType key = encode_key(seg_in[i]);
      if (key < top_keys[k - 1]) {
        int pos = k - 1;
        while (pos > 0 && key < top_keys[pos - 1]) {
          top_keys[pos] = top_keys[pos - 1];
          top_indices[pos] = top_indices[pos - 1];
          --pos;
        }
        top_keys[pos] = key;
        top_indices[pos] = i;
      }
    }

    // Write thread's top-K to temp buffer
    for (int j = 0; j < k; ++j) {
      seg_temp_keys[lid * k + j] = decode_key(top_keys[j]);
      seg_temp_inds[lid * k + j] = top_indices[j];
    }
  }
};

// ======================== Named Kernel Functors ========================
// SYCL requires named functor classes for parallel_for when using
// -fsycl-host-compiler (PyTorch's build system). These thin wrappers
// forward to the actual kernel operator() with the required
// [[intel::reqd_sub_group_size(32)]] attribute.

template <typename scalar_t, bool IS_DESCENDING, int GROUP_SIZE, int MAX_K>
struct HeapScanFunctor {
  using KernelType = HeapScanKernel<scalar_t, IS_DESCENDING, GROUP_SIZE, MAX_K>;
  KernelType kern_;
  HeapScanFunctor(KernelType kern) : kern_(kern) {}
  [[intel::reqd_sub_group_size(32)]] void operator()(
      sycl::nd_item<1> item) const {
    kern_(item);
  }
};

template <
    typename scalar_t,
    bool IS_DESCENDING,
    int GROUP_SIZE,
    int KEYS_PER_THREAD>
struct TopKSmallFunctor {
  using KernelType =
      RadixTopKKernel<scalar_t, IS_DESCENDING, GROUP_SIZE, KEYS_PER_THREAD>;
  KernelType kern_;
  sycl::local_accessor<char, 1> slm_;
  TopKSmallFunctor(KernelType kern, sycl::local_accessor<char, 1> slm)
      : kern_(kern), slm_(slm) {}
  [[intel::reqd_sub_group_size(32)]] void operator()(
      sycl::nd_item<1> item) const {
    kern_(item, slm_.get_multi_ptr<sycl::access::decorated::no>().get());
  }
};

template <
    typename scalar_t,
    bool IS_DESCENDING,
    int GROUP_SIZE,
    int KEYS_PER_THREAD>
struct TopKLargeFunctor {
  using KernelType = RadixTopKLargeKernel<
      scalar_t,
      IS_DESCENDING,
      GROUP_SIZE,
      KEYS_PER_THREAD>;
  KernelType kern_;
  sycl::local_accessor<char, 1> slm_;
  TopKLargeFunctor(KernelType kern, sycl::local_accessor<char, 1> slm)
      : kern_(kern), slm_(slm) {}
  [[intel::reqd_sub_group_size(32)]] void operator()(
      sycl::nd_item<1> item) const {
    kern_(item, slm_.get_multi_ptr<sycl::access::decorated::no>().get());
  }
};

// ======================== Kernel Launchers ========================
// Adaptive GROUP_SIZE dispatch: smaller GROUP_SIZE for small nelements
// to avoid idle threads, larger GROUP_SIZE for large nelements to
// maximize throughput. KEYS_PER_THREAD=4 enables vec4 loads.

template <typename scalar_t, bool IS_DESCENDING, int GS>
void launch_topk_small_gs(
    const scalar_t* input,
    scalar_t* values_out,
    int64_t* indices_out,
    int nsegments,
    int nelements,
    int k,
    sycl::queue& q) {
  constexpr int KPT = 4;
  using Kernel = RadixTopKKernel<scalar_t, IS_DESCENDING, GS, KPT>;
  int slm_size = Kernel::local_mem_size();
  Kernel kern(input, values_out, indices_out, nelements, k);
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<char, 1> slm(sycl::range<1>(slm_size), cgh);
    TopKSmallFunctor<scalar_t, IS_DESCENDING, GS, KPT> functor(kern, slm);
    cgh.parallel_for(sycl::nd_range<1>(nsegments * GS, GS), functor);
  });
}

// Adaptive GROUP_SIZE dispatch for single-tile radix select.
// Each GS handles GS*4 elements, so GS=64 covers up to 256,
// GS=128 covers up to 512, etc. Larger GS has more barrier overhead,
// so we use the smallest GS that covers all elements.
template <typename scalar_t, bool IS_DESCENDING>
void launch_topk_small(
    const scalar_t* input,
    scalar_t* values_out,
    int64_t* indices_out,
    int nsegments,
    int nelements,
    int k,
    sycl::queue& q) {
  if (nelements <= 256)
    launch_topk_small_gs<scalar_t, IS_DESCENDING, 64>(
        input, values_out, indices_out, nsegments, nelements, k, q);
  else if (nelements <= 512)
    launch_topk_small_gs<scalar_t, IS_DESCENDING, 128>(
        input, values_out, indices_out, nsegments, nelements, k, q);
  else if (nelements <= 1024)
    launch_topk_small_gs<scalar_t, IS_DESCENDING, 256>(
        input, values_out, indices_out, nsegments, nelements, k, q);
  else if (nelements <= 2048)
    launch_topk_small_gs<scalar_t, IS_DESCENDING, 512>(
        input, values_out, indices_out, nsegments, nelements, k, q);
  else
    launch_topk_small_gs<scalar_t, IS_DESCENDING, 1024>(
        input, values_out, indices_out, nsegments, nelements, k, q);
}

template <typename scalar_t, bool IS_DESCENDING>
void launch_topk_large(
    const scalar_t* input,
    scalar_t* values_out,
    int64_t* indices_out,
    int nsegments,
    int nelements,
    int k,
    scalar_t* temp_keys,
    int64_t* temp_indices,
    sycl::queue& q) {
  constexpr int GS = 1024;
  constexpr int KPT = 4;
  using Kernel = RadixTopKLargeKernel<scalar_t, IS_DESCENDING, GS, KPT>;
  int slm_size = Kernel::local_mem_size();
  Kernel kern(
      input, values_out, indices_out, nelements, k, temp_keys, temp_indices);
  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<char, 1> slm(sycl::range<1>(slm_size), cgh);
    TopKLargeFunctor<scalar_t, IS_DESCENDING, GS, KPT> functor(kern, slm);
    cgh.parallel_for(sycl::nd_range<1>(nsegments * GS, GS), functor);
  });
}

template <typename scalar_t, bool IS_DESCENDING, int GS>
void launch_heap_scan_gs(
    const scalar_t* input,
    scalar_t* temp_keys,
    int64_t* temp_indices,
    int nsegments,
    int nelements,
    int k,
    sycl::queue& q) {
  using Kernel = HeapScanKernel<scalar_t, IS_DESCENDING, GS, 16>;
  Kernel kern(input, temp_keys, temp_indices, nelements, k);
  HeapScanFunctor<scalar_t, IS_DESCENDING, GS, 16> functor(kern);
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(nsegments * GS, GS), functor);
  });
}

// Adaptive GROUP_SIZE for heap scan: more threads = more parallelism but
// each thread's block is smaller (less data locality). Empirically:
//   K<=4: GS=1024 (each thread scans ~nelements/1024 elements)
//   K<=8: GS=512 (balance scan work vs reduced-data size)
//   K<=16: GS=256 (keeps GROUP_SIZE*K <= 4096 for single-tile phase 2)
template <typename scalar_t, bool IS_DESCENDING>
void launch_heap_scan(
    const scalar_t* input,
    scalar_t* temp_keys,
    int64_t* temp_indices,
    int nsegments,
    int nelements,
    int k,
    sycl::queue& q) {
  if (k <= 4)
    launch_heap_scan_gs<scalar_t, IS_DESCENDING, 1024>(
        input, temp_keys, temp_indices, nsegments, nelements, k, q);
  else if (k <= 8)
    launch_heap_scan_gs<scalar_t, IS_DESCENDING, 512>(
        input, temp_keys, temp_indices, nsegments, nelements, k, q);
  else
    launch_heap_scan_gs<scalar_t, IS_DESCENDING, 256>(
        input, temp_keys, temp_indices, nsegments, nelements, k, q);
}

// ======================== Dispatch Function ========================
//
// Main entry point. Routes to the appropriate kernel based on
// (nelements, k) parameters:
//
//   nelements <= 4096:        Single-tile radix select (fastest)
//   nelements > 4096, K<=16:  Heap-scan + single-tile radix (2 phases)
//   nelements > 4096, K>16:   Multi-tile radix merge
//
// The `largest` parameter controls sort direction via IS_DESCENDING template.
// Output is unsorted — caller handles sorting if needed.

template <typename scalar_t>
void radix_topk_kernel(
    const scalar_t* input,
    scalar_t* values_out,
    int64_t* indices_out,
    int nsegments,
    int nelements,
    int k,
    bool largest,
    sycl::queue& q) {
  constexpr int MAX_SMALL = 4096;

  auto dispatch_descending = [&](auto largest_tag) {
    constexpr bool IS_DESCENDING = decltype(largest_tag)::value;

    if (nelements <= MAX_SMALL) {
      // Path 1: All elements fit in one work-group's registers
      launch_topk_small<scalar_t, IS_DESCENDING>(
          input, values_out, indices_out, nsegments, nelements, k, q);
    } else if (k <= 16 && nelements > (k <= 8 ? MAX_SMALL : 2 * MAX_SMALL)) {
      // Path 2: Heap-select for small K with large dimensions.
      // Phase 1: Each of GS threads scans nelements/GS values, keeping
      // top-K in registers. Produces GS*K candidates.
      int scan_gs = (k <= 4) ? 1024 : (k <= 8) ? 512 : 256;
      int reduced_size = scan_gs * k;

      auto opts = at::TensorOptions()
                      .dtype(c10::CppTypeToScalarType<scalar_t>::value)
                      .device(at::kXPU);
      at::Tensor scan_keys = at::empty({nsegments, reduced_size}, opts);
      at::Tensor scan_indices =
          at::empty({nsegments, reduced_size}, opts.dtype(at::kLong));

      // Phase 1: Each thread scans a block, tracks top-K in registers
      launch_heap_scan<scalar_t, IS_DESCENDING>(
          input,
          scan_keys.data_ptr<scalar_t>(),
          scan_indices.data_ptr<int64_t>(),
          nsegments,
          nelements,
          k,
          q);

      // Phase 2: Single-tile radix select on the GS*K candidates
      at::Tensor phase2_values = at::empty({nsegments, k}, opts);
      at::Tensor phase2_indices =
          at::empty({nsegments, k}, opts.dtype(at::kLong));

      launch_topk_small<scalar_t, IS_DESCENDING>(
          scan_keys.data_ptr<scalar_t>(),
          phase2_values.data_ptr<scalar_t>(),
          phase2_indices.data_ptr<int64_t>(),
          nsegments,
          reduced_size,
          k,
          q);

      // Phase 3: Remap indices from reduced-data space back to original
      // input indices, then copy final results to output.
      at::Tensor remapped_indices = scan_indices.gather(-1, phase2_indices);

      q.memcpy(
          values_out,
          phase2_values.data_ptr<scalar_t>(),
          nsegments * k * sizeof(scalar_t));
      q.memcpy(
          indices_out,
          remapped_indices.data_ptr<int64_t>(),
          nsegments * k * sizeof(int64_t));
      q.wait();
    } else {
      // Path 3: Multi-tile radix merge for large dimensions with K > 16.
      // Processes input in tiles of PL=4096, running radix select per tile.
      constexpr int PL = 1024 * 4;
      auto opts = at::TensorOptions()
                      .dtype(c10::CppTypeToScalarType<scalar_t>::value)
                      .device(at::kXPU);
      at::Tensor temp_keys = at::empty({nsegments, PL}, opts);
      at::Tensor temp_indices =
          at::empty({nsegments, PL}, opts.dtype(at::kLong));

      launch_topk_large<scalar_t, IS_DESCENDING>(
          input,
          values_out,
          indices_out,
          nsegments,
          nelements,
          k,
          temp_keys.data_ptr<scalar_t>(),
          temp_indices.data_ptr<int64_t>(),
          q);
    }
  };

  if (largest) {
    dispatch_descending(std::true_type{});
  } else {
    dispatch_descending(std::false_type{});
  }
}

} // namespace xpu
} // namespace native
} // namespace at

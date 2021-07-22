#ifndef CILKSTL_ALGORITHM_H
#define CILKSTL_ALGORITHM_H

#include <cilk/cilk.h>
#include <cilk/reducer.h>
#include <cilk/reducer_max.h>
#include <cilk/reducer_min.h>
#include <cilk/reducer_opadd.h>

#include <cstdlib>
#include <iostream>
#include <iterator>
#include <vector>

namespace cilkstl {
namespace __parallel {

constexpr int PARTITION_GS = 4096;  // Below this grainsize partition will default to serial
constexpr int PART_SIZE = 64;       // TODO make this a function of array length

/**
 * Serial code to compute a partition inplace using the predicate `p` amongst elements that are
 * index `offset` mod `part_size` in the sequence defined by [`first`, `last`). The `part_size` is
 * computed as the floor of the length of the sequence [first, last) divided by num_parts.
 */
template <class _RandomAccessIterator, class _PredicateFunc>
typename std::iterator_traits<_RandomAccessIterator>::difference_type strided_partition(
    _RandomAccessIterator first, _RandomAccessIterator last, _PredicateFunc p,
    typename std::iterator_traits<_RandomAccessIterator>::difference_type num_parts, int offset) {
  typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type diff_t;
  diff_t range_width = last - first;
  diff_t part_size = (diff_t)(range_width / num_parts);

  _RandomAccessIterator s = first + offset;
  _RandomAccessIterator e = (part_size * num_parts + offset < range_width)
                                ? first + (part_size * num_parts + offset)
                                : first + (part_size * (num_parts - 1) + offset);

  while (s < e) {
    std::swap(*s, *e);
    while (p(*s) && s < e) s += part_size;
    while (!p(*e) && s < e) e -= part_size;
  }

  if (p(*s)) {
    return (diff_t)((s - first) + 1);
  };
  return (diff_t)(s - first);
}

/**
 * Computes a parallel partition by splitting the input sequence into blocks of size `PART_SIZE` and
 * computing a partitioning amongst the ith element in each block in parallel. There's an uncertain
 * region in the middle, and this region is partitioned serially in the end. This algorithm works
 * best when the sizes of the two partitions are similar and the two classes are distributed
 * somewhat randomly. It can exhibit poor performance otherwise.
 */
template <class _RandomAccessIterator, class _PredicateFunc>
_RandomAccessIterator partition(_RandomAccessIterator first, _RandomAccessIterator last,
                                _PredicateFunc p) {
  typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type diff_t;
  diff_t range_width = last - first;
  if (range_width < PARTITION_GS) {
    return std::partition(first, last, p);
  }

  diff_t num_parts = (diff_t)(range_width / PART_SIZE);
  std::vector<diff_t> results(PART_SIZE);
  for (int i = 0; i < PART_SIZE; ++i) {
    results[i] = cilk_spawn strided_partition(first, last, p, num_parts, i);
  }
  cilk_sync;
  diff_t left = *std::min_element(results.begin(), results.end());
  diff_t right = *std::max_element(results.begin(), results.end());
  return std::partition(first + left, first + right, p);
}

}  // namespace __parallel
};  // namespace cilkstl

#endif

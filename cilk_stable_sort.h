#ifndef CILKSTL_STABLE_SORT_H
#define CILKSTL_STABLE_SORT_H

#include <cilk/cilk.h>
#include <cilk/reducer.h>
#include <cilk/reducer_opadd.h>

#include <algorithm>
#include <cstdlib>
#include <iterator>

namespace cilkstl {
namespace __parallel {
namespace __sort {

constexpr int CILKSTL_PARALLEL_CUTOFF =
    4000;  // cutoff below which the sort routine will default to serial execution
constexpr int CILKSTL_PARALLEL_MERGE_CUTOFF =
    1000;  // cutoff below which the merge routine will default to serial execution

/**
 * This file implements a standard parallel mergesort algorithm using a single temporary buffer of
 * equal size to the input sequence
 */

template <class _DataType>
class StableSortBuffer {
 public:
  StableSortBuffer(size_t size) { data_ = (_DataType *)malloc(size * sizeof(_DataType)); }
  ~StableSortBuffer() { delete data_; }
  _DataType *data() { return data_; }

 private:
  _DataType *data_;

  // disallow copies
  StableSortBuffer(const StableSortBuffer &);
  StableSortBuffer &operator=(const StableSortBuffer &);
};

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _RandomAccessIterator3,
          class _CompareFunc>
void serial_merge(_RandomAccessIterator1 as, _RandomAccessIterator1 ae, _RandomAccessIterator2 bs,
                  _RandomAccessIterator2 be, _RandomAccessIterator3 cs, _CompareFunc comp) {
  while (as < ae && bs < be) {
    if (comp(*bs, *as))
      *cs++ = std::move(*bs++);
    else
      *cs++ = std::move(*as++);
  }
  while (as < ae) *cs++ = std::move(*as++);
  while (bs < be) *cs++ = std::move(*bs++);
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2>
void move_contents(_RandomAccessIterator1 first, _RandomAccessIterator1 last,
                   _RandomAccessIterator2 out) {
  typedef typename std::iterator_traits<_RandomAccessIterator1>::difference_type diff1_t;
  typedef typename std::iterator_traits<_RandomAccessIterator2>::difference_type diff2_t;
  typedef typename std::common_type<diff1_t, diff2_t>::type diff_t;
  diff_t range_width = last - first;
  cilk_for(diff_t k = 0; k < range_width; ++k) { *(out + k) = std::move(*(first + k)); }
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _RandomAccessIterator3,
          class _CompareFunc>
void parallel_merge(_RandomAccessIterator1 as, _RandomAccessIterator1 ae, _RandomAccessIterator2 bs,
                    _RandomAccessIterator2 be, _RandomAccessIterator3 cs, _CompareFunc comp) {
  typedef typename std::iterator_traits<_RandomAccessIterator1>::difference_type diff1_t;
  typedef typename std::iterator_traits<_RandomAccessIterator2>::difference_type diff2_t;
  typedef typename std::iterator_traits<_RandomAccessIterator3>::difference_type diff3_t;
  typedef typename std::common_type<diff1_t, diff2_t, diff3_t>::type diff_t;

  diff_t a_range_width = ae - as;
  diff_t b_range_width = be - bs;
  if (a_range_width + b_range_width < CILKSTL_PARALLEL_MERGE_CUTOFF) {
    serial_merge(as, ae, bs, be, cs, comp);
    return;
  }

  diff_t a_delta, b_delta;
  _RandomAccessIterator1 am;
  _RandomAccessIterator2 bm;
  if (a_range_width > b_range_width) {
    a_delta = a_range_width - (ae - as) / 2;
    am = as + a_delta;
    bm = std::lower_bound(bs, be, *(as + a_delta), comp);
    b_delta = bm - bs;
  } else {
    b_delta = (be - bs) / 2;
    bm = bs + b_delta;
    am = std::upper_bound(as, ae, *(bs + b_delta), comp);
    a_delta = am - as;
  }
  cilk_spawn parallel_merge(as, am, bs, bm, cs, comp);
  parallel_merge(am, ae, bm, be, cs + a_delta + b_delta, comp);
  return;
}

template <class _RandomAccessIterator1, class _RandomAccessIterator2, class _CompareFunc>
bool merge_sort(_RandomAccessIterator1 first, _RandomAccessIterator1 last,
                _RandomAccessIterator2 out, _CompareFunc comp) {
  // returns true if output is stored in temporary buffer and false if output remains in original

  typedef typename std::iterator_traits<_RandomAccessIterator1>::difference_type diff1_t;
  typedef typename std::iterator_traits<_RandomAccessIterator2>::difference_type diff2_t;
  typedef typename std::common_type<diff1_t, diff2_t>::type diff_t;
  typedef typename std::iterator_traits<_RandomAccessIterator1>::value_type value_t;
  diff_t range_width = last - first;

  _RandomAccessIterator1 middle = first + range_width / 2;
  _RandomAccessIterator2 out_middle = out + range_width / 2;
  _RandomAccessIterator2 out_last = out + range_width;

  // default to serial stable_sort on small range sizes
  if (range_width / 2 <= CILKSTL_PARALLEL_CUTOFF) {
    std::stable_sort(first, middle, comp);
    std::stable_sort(middle, last, comp);
    parallel_merge(first, middle, middle, last, out, comp);
    return true;  // contents in buffer out
  }

  bool r1 = cilk_spawn merge_sort(first, middle, out, comp);
  bool r2 = merge_sort(middle, last, out_middle, comp);
  cilk_sync;

  if (r1 && r2) {
    parallel_merge(out, out_middle, out_middle, out_last, first, comp);
    return false;
  } else if (!r1 && !r2) {
    parallel_merge(first, middle, middle, last, out, comp);
    return true;
  } else if (r1) {
    move_contents(middle, last, out_middle);
    parallel_merge(out, out_middle, out_middle, out_last, first, comp);
    return false;
  } else {
    move_contents(first, middle, out);
    parallel_merge(out, out_middle, out_middle, out_last, first, comp);
    return false;
  }
}

template <class _RandomAccessIterator, class _CompareFunc>
void stable_sort(_RandomAccessIterator first, _RandomAccessIterator last, _CompareFunc comp) {
  typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type diff_t;
  typedef typename std::iterator_traits<_RandomAccessIterator>::value_type value_t;
  diff_t range_width = last - first;

  if (range_width < CILKSTL_PARALLEL_CUTOFF) {
    std::stable_sort(first, last, comp);
    return;
  }

  StableSortBuffer<value_t> buffer(range_width);
  bool result = merge_sort(first, last, buffer.data(), comp);
  if (result) move_contents(buffer.data(), buffer.data() + range_width, first);
}

}  // namespace __sort
}  // namespace __parallel
};  // namespace cilkstl

#endif

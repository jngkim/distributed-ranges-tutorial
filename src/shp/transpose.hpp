// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <oneapi/mkl/blas.hpp>

namespace fft {

template<typename T1, typename T2>
sycl::event transpose_mkl(size_t m, size_t n,
                      T1 in,  size_t lda, T2 out, size_t ldb,
                      const std::vector<sycl::event>& events = {})
{
  auto &&q = dr::shp::__detail::get_queue_for_pointer(out);
  return oneapi::mkl::blas::row_major::omatcopy(q,oneapi::mkl::transpose::trans, m, n, 1, in.get_raw_pointer(), lda, out.get_raw_pointer(), ldb, events);
}

template<typename T1, typename T2>
sycl::event transpose_base(size_t m, size_t n,
                           T1 in,  size_t lda, T2 out, size_t ldb,
                           const std::vector<sycl::event>& events = {})
{
  auto &&q = dr::shp::__detail::get_queue_for_pointer(out);
  constexpr size_t tile_size = 16;
  const size_t m_max         = ((m + tile_size - 1) / tile_size) * tile_size;
  const size_t n_max         = ((n + tile_size - 1) / tile_size) * tile_size;
  auto in_ = in.get_raw_pointer();

  //write coalsed
  return q.parallel_for(sycl::nd_range<2>{{n_max, m_max}, {tile_size, tile_size}}, events,
      [=](sycl::nd_item<2> item) {
      auto x   = item.get_global_id(0);
      auto y   = item.get_global_id(1);
      if (x < n && y < m)
        out[x*ldb+y] = in_[y*lda+x];
    });
  //read
  //return q.parallel_for(sycl::nd_range<2>{{m_max, n_max}, {tile_size, tile_size}}, events,
  //    [=](sycl::nd_item<2> item) {
  //    auto x   = item.get_global_id(1);
  //    auto y   = item.get_global_id(0);
  //    if (x < n && y < m)
  //      out[x*ldb+y] = in_[y*lda+x];
  //  });
}

template<typename T1, typename T2>
sycl::event transpose(size_t m, size_t n,
                      T1 in,  size_t lda, T2 out, size_t ldb,
                      const std::vector<sycl::event>& events = {})
{
  auto &&q = dr::shp::__detail::get_queue_for_pointer(out);
#ifdef USE_MKL_TRANSPOSE
  return oneapi::mkl::blas::row_major::omatcopy(q,oneapi::mkl::transpose::trans, m, n, 1, in.get_raw_pointer(), lda, out.get_raw_pointer(), ldb, events);
#else
  constexpr size_t tile_size = 16;
  const size_t m_max         = ((m + tile_size - 1) / tile_size) * tile_size;
  const size_t n_max         = ((n + tile_size - 1) / tile_size) * tile_size;
  using temp_t = std::iter_value_t<T1>;
  const auto in_ = in.get_raw_pointer();

  return q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(events);
    sycl::local_accessor<temp_t, 2> tile(sycl::range<2>(tile_size, tile_size + 1), cgh);

    cgh.parallel_for(sycl::nd_range<2>{{m_max, n_max}, {tile_size, tile_size}}, [=](sycl::nd_item<2> item) {
      unsigned x   = item.get_global_id(1);
      unsigned y   = item.get_global_id(0);
      unsigned xth = item.get_local_id(1);
      unsigned yth = item.get_local_id(0);

      if (x < n && y < m)
        tile[yth][xth] = in_[(y)*lda + x];
      item.barrier(sycl::access::fence_space::local_space);

      x = item.get_group(0) * tile_size + xth;
      y = item.get_group(1) * tile_size + yth;
      if (x < m && y < n)
        out[(y)*ldb + x] = tile[xth][yth];
    });
  });
#endif
}

template<typename T1, typename T2>
sycl::event transpose_blocked(size_t m, size_t n,
                              T1 in,  size_t lda, T2 out, size_t ldb,
                              const std::vector<sycl::event>& events = {})
{
  auto &&q = dr::shp::__detail::get_queue_for_pointer(out);

  const size_t tile_size = 32;
  const size_t block_rows = 8;
  const size_t n_tiles = ((m + tile_size - 1) / tile_size);

  using temp_t = std::iter_value_t<T1>;
  const auto in_ = in.get_raw_pointer();

  return q.submit([&](sycl::handler& cgh) {
      cgh.depends_on(events);
      sycl::local_accessor<temp_t, 2> tile(sycl::range<2>(tile_size,tile_size+1), cgh);
      cgh.parallel_for(sycl::nd_range<2>{{n_tiles*block_rows, n_tiles*tile_size}, {block_rows, tile_size}},
        [=](sycl::nd_item<2> item) {
        const unsigned thX = item.get_local_id(1); //threadIdx.x
        const unsigned thY = item.get_local_id(0); //threadIdx.y
        unsigned column = item.get_group(1) * tile_size + thX; //item.get_global_id(1);
        unsigned row    = item.get_group(0) * tile_size + thY;

        for (unsigned j = 0; j < tile_size; j += block_rows)
          tile[thY+j][thX] = in_[(row+j)*lda + column];

        item.barrier(sycl::access::fence_space::local_space);

        column = item.get_group(0)*tile_size + thX;
        if(column<m)
        {
          row = item.get_group(1)*tile_size + thY;
          for (unsigned j = 0; j < tile_size; j += block_rows)
          if(row+j < n) out[(row + j)*ldb + column] = tile[thX][thY + j];
        }
    });
  });
}

template<typename T1, typename T2>
sycl::event copy_2d_mkl(size_t m, size_t n,
                        T1 in,  size_t lda, T2 out, size_t ldb,
                        const std::vector<sycl::event>& events = {})
{
  auto &&q = dr::shp::__detail::get_queue_for_pointer(out);
  return oneapi::mkl::blas::row_major::omatcopy(q,oneapi::mkl::transpose::nontrans, m, n, 1, in.get_raw_pointer(), lda, out.get_raw_pointer(), ldb, events);
}

template<typename T1, typename T2>
sycl::event copy_2d(size_t m, size_t n,
                    T1 in,  size_t lda, T2 out, size_t ldb,
                    const std::vector<sycl::event>& events = {})
{
  auto &&q = dr::shp::__detail::get_queue_for_pointer(out);
  constexpr size_t tile_size = 16;
  const size_t m_max         = ((m + tile_size - 1) / tile_size) * tile_size;
  const size_t n_max         = ((n + tile_size - 1) / tile_size) * tile_size;
  //using temp_t = std::iter_value_t<T1>;
  //auto in_ = in.get_raw_pointer();

  return q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(events);
    cgh.parallel_for(sycl::nd_range<2>{{m_max, n_max}, {tile_size, tile_size}}, [=](sycl::nd_item<2> item) {
      unsigned i   = item.get_global_id(0);
      unsigned j   = item.get_global_id(1);
      if (i < m && j < n)
        out[i*ldb+j] = in[i*lda+j];
    });
  });
}
}


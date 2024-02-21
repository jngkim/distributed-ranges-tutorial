// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"
#include "oneapi/mkl/dfti.hpp"
#include <dr/shp.hpp>
#include <dr/detail/logger.hpp>
#include <fmt/core.h>
#include <latch>
#include <thread>
#include <chrono>
#include <omp.h>

template <rng::forward_range R> auto values_view(R &&m) {
  return m | dr::shp::views::transform([](auto &&e) {
           auto &&[_, v] = e;
           return v;
         });
}

template <typename T>
void init_matrix_3d(dr::shp::distributed_dense_matrix<T> &m) {
  std::size_t lda = m.shape()[1];
  dr::shp::for_each(dr::shp::par_unseq, m, [=](auto &&entry) {
    auto &&[idx, v] = entry;
    v = idx[0]*lda + idx[1];
  });
}

template <typename T>
void init_matrix_3d(dr::shp::distributed_dense_matrix<std::complex<T>> &m) {
  std::size_t lda = m.shape()[1];
  dr::shp::for_each(dr::shp::par_unseq, m, [=](auto &&entry) {
    auto &&[idx, v] = entry;
    v = {idx[0]*lda + idx[1], 1.0};
  });
}

template <typename T1, typename T2>
sycl::event transpose_tile(std::size_t m, std::size_t n, T1 in, std::size_t lda,
                           T2 out, std::size_t ldb,
                           const std::vector<sycl::event> &events = {}) {
  auto &&q = dr::shp::__detail::get_queue_for_pointer(out);
  constexpr std::size_t tile_size = 16;
  const std::size_t m_max = ((m + tile_size - 1) / tile_size) * tile_size;
  const std::size_t n_max = ((n + tile_size - 1) / tile_size) * tile_size;
  using temp_t = std::iter_value_t<T1>;
  const auto in_ = in.get_raw_pointer();

  return q.submit([&](sycl::handler &cgh) {
    cgh.depends_on(events);
    sycl::local_accessor<temp_t, 2> tile(
        sycl::range<2>(tile_size, tile_size + 1), cgh);

    cgh.parallel_for(sycl::nd_range<2>{{m_max, n_max}, {tile_size, tile_size}},
                     [=](sycl::nd_item<2> item) {
                       unsigned x = item.get_global_id(1);
                       unsigned y = item.get_global_id(0);
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
}

template<typename T1, typename T2>
sycl::event pack_tiles(size_t m, size_t n,
                       T1 in,  size_t lda, T2 out, size_t ldb,
                       const std::vector<sycl::event>& events = {})
{
  auto &&q = dr::shp::__detail::get_queue_for_pointer(out);
  constexpr size_t tile_size = 16;
  const size_t m_max         = ((m + tile_size - 1) / tile_size) * tile_size;
  const size_t n_max         = ((n + tile_size - 1) / tile_size) * tile_size;
  //using temp_t = std::iter_value_t<T1>;
  auto in_ = in.get_raw_pointer();

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

template<typename T>
struct TransposeBase {
  std::int64_t m_, n_;
  dr::shp::block_cyclic row_blocks_;
  dr::shp::distributed_dense_matrix<T> in_, out_;

  explicit TransposeBase(std::int64_t m_in)
      : m_(dr::__detail::round_up(m_in, dr::shp::nprocs())), n_(m_ * m_),
        row_blocks_({dr::shp::tile::div, dr::shp::tile::div},
                    {dr::shp::nprocs(), 1}),
        in_({m_, n_}, row_blocks_), out_({n_, m_}, row_blocks_) {
          init_matrix_3d(in_);
        }


  virtual ~TransposeBase() = default;
  virtual void transpose(dr::shp::distributed_dense_matrix<T> &i_mat,
                         dr::shp::distributed_dense_matrix<T> &o_mat) = 0;

  void check() { 
    transpose(in_, out_);
    for (std::size_t i = 0; i < m_; i++)
    {
      for (std::size_t j = 0; j < m_; j++)
      {
        for (std::size_t k = 0; k < m_; k++)
          if(out_[{i * m_ + j , k}] != (i * m_ + j + k * m_ * m_)) 
            fmt::print("Error {} {} {}\n", i, j, k, out_[{i * m_ + j , k}]);
        //fmt::print("\n");
      }
      //fmt::print("\n");
    }
  }
  void compute() { 
    transpose(in_, out_);
  }
};

template<typename T>
struct TransposeSerial: TransposeBase<T> {

  explicit TransposeSerial(std::int64_t m_in): TransposeBase<T>(m_in){}

  void transpose(dr::shp::distributed_dense_matrix<T> &i_mat,
                 dr::shp::distributed_dense_matrix<T> &o_mat) override {
    std::vector<sycl::event> events;
    int ntiles = i_mat.segments().size();
    int lda = i_mat.shape()[1];
    int ldb = o_mat.shape()[1];
    // need to handle offsets better
    int m_local = i_mat.shape()[0] / ntiles;
    int n_local = o_mat.shape()[0] / ntiles;
    for (int i = 0; i < ntiles; i++) {
      for (int j_ = 0; j_ < ntiles; j_++) {
        int j = (j_ + i) % ntiles;
        auto &&send_tile = i_mat.tile({i, 0});
        auto &&recv_tile = o_mat.tile({j, 0});
        auto e = transpose_tile(m_local, n_local, send_tile.data() + j * n_local,
            lda, recv_tile.data() + i * m_local, ldb);
        events.push_back(e);
      }
    }
    sycl::event::wait(events);
  }
};

template<typename T>
struct TransposeThreads: TransposeBase<T> {

  explicit TransposeThreads(std::int64_t m_in): TransposeBase<T>(m_in){}

  void transpose(dr::shp::distributed_dense_matrix<T> &i_mat,
                 dr::shp::distributed_dense_matrix<T> &o_mat) override {
    int ntiles = i_mat.segments().size();
    std::vector<std::jthread> threads;
    for (int i = 0; i < ntiles; ++i) {
      threads.emplace_back([i, ntiles, &i_mat, &o_mat] {
        auto &&atile = i_mat.tile({i, 0});

        int lda = i_mat.shape()[1];
        int ldb = o_mat.shape()[1];
        int m_local = i_mat.shape()[0] / ntiles;
        int n_local = o_mat.shape()[0] / ntiles;

        std::vector<sycl::event> events;
        for (int j_ = 0; j_ < ntiles; j_++) {
          int j = (j_ + i) % ntiles;
          auto &&recv_tile = o_mat.tile({j, 0});
          auto e = transpose_tile(m_local, n_local, atile.data() + j * n_local,
                                  lda, recv_tile.data() + i * m_local, ldb);
          events.push_back(e);
        }
        sycl::event::wait(events);
      });
    }
  }
};

template<typename T>
struct TransposeAll2AllThreads: TransposeBase<T> {

  explicit TransposeAll2AllThreads(std::int64_t m_in): TransposeBase<T>(m_in){}

  void transpose(dr::shp::distributed_dense_matrix<T> &i_mat,
                 dr::shp::distributed_dense_matrix<T> &o_mat) override {
    int ntiles = i_mat.segments().size();
    std::vector<std::jthread> threads;
    std::latch all2all_ready(ntiles);
    for (int i = 0; i < ntiles; ++i) {
      threads.emplace_back([i, ntiles, &i_mat, &o_mat, &all2all_ready] {
        auto &&atile = i_mat.tile({i, 0});

        int lda = i_mat.shape()[1];
        int ldb = o_mat.shape()[1];
        int m_local = i_mat.shape()[0] / ntiles;
        int n_local = o_mat.shape()[0] / ntiles;

        const size_t block_size = m_local * n_local;
        //pack In(m_local, lda) -> Out(nproc, m_local*n_local)
        std::vector<sycl::event> events;
        {
          auto &&packed_tile = o_mat.tile({i, 0});
          for (std::size_t j_ = 0; j_ < ntiles; j_++) {
            std::size_t j = (j_ + i) % std::size_t(ntiles);
            auto e = pack_tiles(m_local, n_local, atile.data() + j * n_local, lda, packed_tile.data() + j * block_size, n_local);
            events.push_back(e);
          }
        }
        sycl::event::wait(events);
        all2all_ready.arrive_and_wait();

        events.clear();
        //copy Out_i(j,blocksize) -> remote In_j(i, blocksize)
        for (std::size_t j_ = 0; j_ < ntiles; j_++) {
          std::size_t j = (j_ + i) % std::size_t(ntiles);
          auto &&send_tile = o_mat.tile({i, 0});
          auto &&recv_tile = i_mat.tile({j, 0});
          auto e = dr::shp::copy_async(send_tile.data() + j*block_size, send_tile.data() + (j+1)*block_size, 
                                       recv_tile.data() + i*block_size);
          events.push_back(e);
        }
        sycl::event::wait(events);
        events.clear();

        //unpack (transpose) In(nproc,blocksize) -> Out(n_local, ldb)
        auto &&btile = o_mat.tile({i, 0}); // local tile (n_local, ldb)
        auto &&unpacked_tile = i_mat.tile({i, 0});
        for (std::size_t j_ = 0; j_ < ntiles; j_++) {
          std::size_t j = (j_ + i) % std::size_t(ntiles);
          auto e = transpose_tile(m_local, n_local, unpacked_tile.data() + j * block_size, n_local, btile.data() + j * m_local, ldb);
          events.push_back(e);
        }
        sycl::event::wait(events);
      });
    }
  }
};


int main(int argc, char *argv[]) {

  cxxopts::Options options_spec(argv[0], "fft3d");
  // clang-format off
  options_spec.add_options()
    ("d, num-devices", "number of sycl devices, 0 uses all available devices", cxxopts::value<std::size_t>()->default_value("0"))
    ("n", "problem size", cxxopts::value<std::size_t>()->default_value("8"))
    ("r,repetitions", "Number of repetitions", cxxopts::value<std::size_t>()->default_value("0"))
    ("log", "enable logging")
    ("logprefix", "appended .RANK.log", cxxopts::value<std::string>()->default_value("dr"))
    ("verbose", "verbose output")
    ("h,help", "Print help");
  // clang-format on

  cxxopts::ParseResult options;
  try {
    options = options_spec.parse(argc, argv);
  } catch (const cxxopts::OptionParseException &e) {
    std::cout << options_spec.help() << "\n";
    exit(1);
  }

  std::unique_ptr<std::ofstream> logfile;
  if (options.count("log")) {
    logfile.reset(new std::ofstream(
        fmt::format("{}.log", options["logprefix"].as<std::string>())));
    dr::drlog.set_file(*logfile);
  }

  // 512^3 up to 3072
  std::size_t dim = options["n"].as<std::size_t>();
  std::size_t nreps = options["repetitions"].as<std::size_t>();

  auto devices = dr::shp::get_numa_devices(sycl::default_selector_v);
  std::size_t ranks = options["num-devices"].as<std::size_t>();
  if (ranks != 0 && ranks < devices.size()) {
    devices.resize(ranks);
  }
  //for (auto device : devices) {
  //  fmt::print("Device: {}\n", device.get_info<sycl::info::device::name>());
  //}
  dr::shp::init(devices);

  omp_set_num_threads(devices.size());

  dim = (dim + ranks-1) / ranks * ranks;

  using real_t = float;
  //using value_t = std::complex<real_t>;
  using value_t = real_t;


#ifdef TRANSPOSE_SERIAL
  TransposeSerial<value_t> test(dim);
#endif
#ifdef TRANSPOSE_THREADS
  TransposeThreads<value_t> test(dim);
#endif
#ifdef TRANSPOSE_ALL2ALL_THREADS
  TransposeAll2AllThreads<value_t> test(dim);
#endif

  if (nreps == 0) {
    test.check();
  } else {
    test.compute();
    double elapsed = 0;
    for (int iter = 0; iter < nreps; ++iter) {
      auto begin = std::chrono::steady_clock::now();
      test.compute();
      auto end = std::chrono::steady_clock::now();
      if(iter) 
        elapsed += std::chrono::duration<double>(end - begin).count();
    }

    std::size_t volume = dim * dim * dim * sizeof(value_t) * 2;
    double t_avg = elapsed / (nreps - 1);
    fmt::print("transpose {0} AvgTime {1:.3f} GB/s {2:.3f}\n", dim, t_avg, volume/t_avg*1e-9);
  }

  return 0;
}


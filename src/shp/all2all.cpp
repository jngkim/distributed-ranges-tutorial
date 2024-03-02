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
void init_matrix_3d_T(dr::shp::distributed_dense_matrix<T> &m) {
  std::size_t lda = m.shape()[0];
  dr::shp::for_each(dr::shp::par_unseq, m, [=](auto &&entry) {
    auto &&[idx, v] = entry;
    v = idx[1]*lda + idx[0];
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

template<typename T>
struct All2AllBase {

  std::int64_t m_, n_;
  dr::shp::block_cyclic row_blocks_;
  dr::shp::distributed_dense_matrix<T> in_, out_;

  explicit All2AllBase(std::int64_t m_in)
      : m_(dr::__detail::round_up(m_in, dr::shp::nprocs())), n_(m_),
        row_blocks_({dr::shp::tile::div, dr::shp::tile::div},
                    {dr::shp::nprocs(), 1}),
        in_({m_, n_}, row_blocks_), out_({n_, m_}, row_blocks_) {
          init_matrix_3d(in_);
        }


  virtual ~All2AllBase() = default;
  virtual void transpose(dr::shp::distributed_dense_matrix<T> &i_mat,
                         dr::shp::distributed_dense_matrix<T> &o_mat) = 0;

  void check() { }

  void compute() { 
    transpose(in_, out_);
  }
};

template<typename T>
struct All2AllBatch: All2AllBase<T> {

  explicit All2AllBatch(std::int64_t m_in): All2AllBase<T>(m_in){}

  void transpose(dr::shp::distributed_dense_matrix<T> &i_mat,
                 dr::shp::distributed_dense_matrix<T> &o_mat) override {

    int ntiles = i_mat.segments().size();
    std::vector<sycl::event> events;
    int m_local = i_mat.shape()[0] / ntiles;
    int n_local = o_mat.shape()[0] / ntiles;
    const size_t block_size = m_local * n_local;
    for (int i = 0; i < ntiles; ++i) {
      for (std::size_t j_ = 0; j_ < ntiles; j_++) {
        std::size_t j = (j_ + i) % std::size_t(ntiles);
        auto &&send_tile = o_mat.tile({i, 0});
        auto &&recv_tile = i_mat.tile({j, 0});
        auto e = dr::shp::copy_async(send_tile.data() + j*block_size, send_tile.data() + (j+1)*block_size, 
            recv_tile.data() + i*block_size);
        events.emplace_back(e);
      }
    }
    sycl::event::wait(events);
  }
};

template<typename T>
struct All2AllPut: All2AllBase<T> {

  explicit All2AllPut(std::int64_t m_in): All2AllBase<T>(m_in){}

  void transpose(dr::shp::distributed_dense_matrix<T> &i_mat,
                 dr::shp::distributed_dense_matrix<T> &o_mat) override {

    int ntiles = i_mat.segments().size();
    std::vector<sycl::event> events;
    int m_local = i_mat.shape()[0] / ntiles;
    int n_local = o_mat.shape()[0] / ntiles;
    const size_t block_size = m_local * n_local;
    for (int i = 0; i < ntiles; ++i) {
      auto &&q = dr::shp::__detail::queue(i);
      for (std::size_t j_ = 0; j_ < ntiles; j_++) {
        std::size_t j = (j_ + i) % std::size_t(ntiles);
        auto &&send_tile = o_mat.tile({i, 0});
        auto &&recv_tile = i_mat.tile({j, 0});
        auto e = q.memcpy(
            dr::shp::__detail::local(recv_tile).data() + i*block_size, 
            dr::shp::__detail::local(send_tile).data() + j*block_size, 
            block_size * sizeof(T));
        events.emplace_back(e);
      }
    }
    sycl::event::wait(events);
  }
};

template<typename T>
struct All2AllGet: All2AllBase<T> {

  explicit All2AllGet(std::int64_t m_in): All2AllBase<T>(m_in){}

  void transpose(dr::shp::distributed_dense_matrix<T> &i_mat,
                 dr::shp::distributed_dense_matrix<T> &o_mat) override {

    int ntiles = i_mat.segments().size();
    std::vector<sycl::event> events;
    int m_local = i_mat.shape()[0] / ntiles;
    int n_local = o_mat.shape()[0] / ntiles;
    const size_t block_size = m_local * n_local;
    for (int i = 0; i < ntiles; ++i) {
      for (std::size_t j_ = 0; j_ < ntiles; j_++) {
        std::size_t j = (j_ + i) % std::size_t(ntiles);
        auto &&q = dr::shp::__detail::queue(j);
        auto &&send_tile = o_mat.tile({i, 0});
        auto &&recv_tile = i_mat.tile({j, 0});
        auto e = q.memcpy(
            dr::shp::__detail::local(recv_tile).data() + i*block_size, 
            dr::shp::__detail::local(send_tile).data() + j*block_size, 
            block_size * sizeof(T));
        events.emplace_back(e);
      }
    }
    sycl::event::wait(events);
  }
};


template<typename T>
struct All2AllPutOMP: All2AllBase<T> {

  explicit All2AllPutOMP(std::int64_t m_in): All2AllBase<T>(m_in){}

  void transpose(dr::shp::distributed_dense_matrix<T> &i_mat,
                 dr::shp::distributed_dense_matrix<T> &o_mat) override {

#pragma omp parallel
    {
      int i = omp_get_thread_num();
      int ntiles = i_mat.segments().size();
      std::vector<sycl::event> events;
      int m_local = i_mat.shape()[0] / ntiles;
      int n_local = o_mat.shape()[0] / ntiles;
      const size_t block_size = m_local * n_local;
      auto &&q = dr::shp::__detail::queue(i);
      for (std::size_t j_ = 0; j_ < ntiles; j_++) {
        std::size_t j = (j_ + i) % std::size_t(ntiles);
        auto &&send_tile = o_mat.tile({i, 0});
        auto &&recv_tile = i_mat.tile({j, 0});
        auto e = q.memcpy(
            dr::shp::__detail::local(recv_tile).data() + i*block_size, 
            dr::shp::__detail::local(send_tile).data() + j*block_size, 
            block_size * sizeof(T));
        events.emplace_back(e);
      }
      sycl::event::wait(events);
    }
  }
};

int main(int argc, char *argv[]) {

  cxxopts::Options options_spec(argv[0], "fft3d");
  // clang-format off
  options_spec.add_options()
    ("d, num-devices", "number of sycl devices, 0 uses all available devices", cxxopts::value<std::size_t>()->default_value("0"))
    ("n", "problem size", cxxopts::value<std::size_t>()->default_value("8"))
    ("p, parallel", "host 0:serial, 1:threads, 2:openmp", cxxopts::value<int>()->default_value("0"))
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

  int p_mode = options["p"].as<int>();

  auto devices = dr::shp::get_numa_devices(sycl::default_selector_v);
  std::size_t ranks = options["num-devices"].as<std::size_t>();
  if (ranks != 0 && ranks < devices.size()) {
    devices.resize(ranks);
  }
  //for (auto device : devices) {
  //  fmt::print("Device: {}\n", device.get_info<sycl::info::device::name>());
  //}
  dr::shp::init(devices);

  dim = (dim + ranks-1) / ranks * ranks;

  using real_t = double;
  using value_t = real_t;

  std::string tag;
  All2AllBase<value_t> *test = nullptr;

  switch(p_mode){
    case 0:
      tag = "A2A_shp";
      test = new All2AllBatch<value_t>(dim);
      break;
    case 1:
      tag = "A2A_put";
      test = new All2AllPut<value_t>(dim);
      break;
    case 2:
      tag = "A2A_GetR";
      test = new All2AllGet<value_t>(dim);
      break;
    case 3:
      tag = "A2A_OMP";
      omp_set_num_threads(ranks);
      test = new All2AllPutOMP<value_t>(dim);
      break;
    default:
      std::cerr << "Not a valid transpose method. Select 0-4" << std::endl;
      return 0;
  }

  if (nreps == 0) {
    test->check();
  } else {
    test->compute();
    double elapsed = 0;
    for (int iter = 0; iter < nreps; ++iter) {
      auto begin = std::chrono::steady_clock::now();
      test->compute();
      auto end = std::chrono::steady_clock::now();
      if(iter) 
        elapsed += std::chrono::duration<double>(end - begin).count();
    }

    std::size_t volume = dim * dim * sizeof(value_t);
    //volume per exchange
    double vol = static_cast<double>(volume/ranks/ranks);

    auto project_time = [vol,ranks]() {
      double bw_loc = 1055;
      double bw_mdfi = 183; //220;
      double bw_xe = 19;    //56/2
      double t = vol * (ranks+1) /bw_loc; // write(ranks) + read(1)
      if(ranks > 1)
        t += vol * ( (ranks - 2)/bw_xe + ranks/2/bw_mdfi);
      return t*1e-9;
    };

    double t_avg = elapsed / (nreps - 1);
    double t_proj = project_time();
    fmt::print("{3} {0} {4} AvgTime {1:.6f} {5:.6f} GB/s {2:.3f}\n",
        dim, t_avg, 2 * volume/t_avg*1e-9, tag, ranks, t_proj);
  }

  delete test;

  return 0;
}


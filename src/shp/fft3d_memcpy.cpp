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

#define STANDALONE_BENCHMARK
#ifndef STANDALONE_BENCHMARK

#include "../common/dr_bench.hpp"

#endif

template <rng::forward_range R> auto values_view(R &&m) {
  return m | dr::shp::views::transform([](auto &&e) {
           auto &&[_, v] = e;
           return v;
         });
}

using real_t = float;
using value_t = std::complex<real_t>;
enum {USE_SERIAL, USE_THREADS, USE_OPENMP};

template <typename T>
void init_matrix_3d(dr::shp::distributed_dense_matrix<std::complex<T>> &m,
                    int N) {
  const int N1 = N;
  const int N2 = m.shape()[1] / N;
  const int N3 = m.shape()[0];
  const int H1 = -1;
  const int H2 = -2;
  const int H3 = -3;
  constexpr T TWOPI = 6.2831853071795864769;

  auto moda = [](int K, int L, int M) { return (T)(((long long)K * L) % M); };

  const T norm = T(1) / (N3 * N2 * N1);
  dr::shp::for_each(dr::shp::par_unseq, m, [=](auto &&entry) {
    auto &&[idx, v] = entry;
    int n3 = idx[0];
    int n2 = idx[1] / N1;
    int n1 = idx[1] % N1;
    T phase = TWOPI * (moda(n1, H1, N1) / N1 + moda(n2, H2, N2) / N2 +
                       moda(n3, H3, N3) / N3);
    v = {std::cos(phase) * norm, std::sin(phase) * norm};
  });
}

sycl::event mkl_dft(auto aplan_, auto data_, bool forward, const std::vector<sycl::event>& events = {}) {
  if (forward) {
    return oneapi::mkl::dft::compute_forward(*aplan_, data_, events);
  }
  return oneapi::mkl::dft::compute_backward(*aplan_, data_, events);
}

template <typename T> struct dft_precision {
  static const oneapi::mkl::dft::precision value =
      oneapi::mkl::dft::precision::SINGLE;
};

template <> struct dft_precision<double> {
  static const oneapi::mkl::dft::precision value =
      oneapi::mkl::dft::precision::DOUBLE;
};

template <typename T> class distributed_fft {
  std::int64_t m_, n_;
  dr::shp::block_cyclic row_blocks_;
  dr::shp::distributed_dense_matrix<value_t> in_, out_;

  using fft_plan_t =
      oneapi::mkl::dft::descriptor<dft_precision<T>::value,
                                   oneapi::mkl::dft::domain::COMPLEX>;

  std::vector<fft_plan_t *> fft_plans;

  void fft_omp(dr::shp::distributed_dense_matrix<std::complex<T>> &i_mat,
                   dr::shp::distributed_dense_matrix<std::complex<T>> &o_mat,
                   bool forward) {
    int ntiles = i_mat.segments().size();
    std::size_t mask_p = 0, mask_u = 1;
    if (!forward)
      std::swap(mask_p, mask_u);

#pragma omp parallel 
    {
        int i = omp_get_thread_num();
        auto plan_0 = fft_plans[2 * i + mask_p];
        auto plan_1 = fft_plans[2 * i + mask_u];

        auto &&atile = i_mat.tile({i, 0});
        mkl_dft(plan_0, dr::shp::__detail::local(atile).data(), forward).wait();

#pragma omp barrier

        auto &&btile = o_mat.tile({i, 0});
        auto &&q = dr::shp::__detail::queue(i);
        auto e = q.memcpy(dr::shp::__detail::local(btile).data(), dr::shp::__detail::local(atile).data(), atile.size()*sizeof(std::complex<T>));
        mkl_dft(plan_1, dr::shp::__detail::local(btile).data(), forward, {e}).wait();
    }
  }

  void fft_threads(dr::shp::distributed_dense_matrix<std::complex<T>> &i_mat,
                   dr::shp::distributed_dense_matrix<std::complex<T>> &o_mat,
                   bool forward) {
    int ntiles = i_mat.segments().size();
    std::size_t mask_p = 0, mask_u = 1;
    if (!forward)
      std::swap(mask_p, mask_u);

    std::vector<std::jthread> threads;
    std::latch fft_phase_done(ntiles);

    for (int i = 0; i < ntiles; ++i) {
      threads.emplace_back([i, ntiles, forward,
                            plan_0 = fft_plans[2 * i + mask_p],
                            plan_1 = fft_plans[2 * i + mask_u], &fft_phase_done,
                            &i_mat, &o_mat] {
        auto &&atile = i_mat.tile({i, 0});
        mkl_dft(plan_0, dr::shp::__detail::local(atile).data(), forward).wait();

        fft_phase_done.arrive_and_wait();

        auto &&btile = o_mat.tile({i, 0});
        auto &&q = dr::shp::__detail::queue(i);
        auto e = q.memcpy(dr::shp::__detail::local(btile).data(), dr::shp::__detail::local(atile).data(), atile.size()*sizeof(std::complex<T>));
        mkl_dft(plan_1, dr::shp::__detail::local(btile).data(), forward, {e}).wait();
      });
    }
  }

  void fft_serial(dr::shp::distributed_dense_matrix<std::complex<T>> &i_mat,
                  dr::shp::distributed_dense_matrix<std::complex<T>> &o_mat,
                  bool forward) {

    std::size_t mask_p = 0, mask_u = 1;
    if (!forward)
      std::swap(mask_p, mask_u);

    std::vector<sycl::event> events;
    int nprocs = i_mat.segments().size();
    for (int i = 0; i < nprocs; i++) {
      auto &&atile = i_mat.tile({i, 0});
      auto e = oneapi::mkl::dft::compute_forward(
          *fft_plans[2 * i + mask_p], dr::shp::__detail::local(atile).data());
      events.push_back(e);
    }
    sycl::event::wait(events);
    events.clear();

    for (int i = 0; i < nprocs; i++) {
      auto &&q = dr::shp::__detail::queue(i);
      auto &&atile = i_mat.tile({i, 0});
      auto &&btile = o_mat.tile({i, 0});
      auto e = q.memcpy(dr::shp::__detail::local(btile).data(), dr::shp::__detail::local(atile).data(), atile.size()*sizeof(std::complex<T>));
      events.push_back(e);
    }
    sycl::event::wait(events);
    events.clear();

    for (int i = 0; i < nprocs; i++) {
      auto &&atile = o_mat.tile({i, 0});
      auto e = oneapi::mkl::dft::compute_forward(
          *fft_plans[2 * i + mask_u], dr::shp::__detail::local(atile).data());
      events.push_back(e);
    }
    sycl::event::wait(events);
  }

public:
  explicit distributed_fft(std::int64_t m_in)
      : m_(dr::__detail::round_up(m_in, dr::shp::nprocs())), n_(m_ * m_),
        row_blocks_({dr::shp::tile::div, dr::shp::tile::div},
                    {dr::shp::nprocs(), 1}),
        in_({m_, n_}, row_blocks_), out_({n_, m_}, row_blocks_) {
    auto nprocs = dr::shp::nprocs();
    fft_plans.reserve(2 * nprocs);

    int m_local = m_ / nprocs;
    for (int i = 0; i < nprocs; i++) {
      auto &&q = dr::shp::__detail::queue(i);
      fft_plan_t *desc = new fft_plan_t({m_, m_});
      desc->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                      m_local);
      desc->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, m_ * m_);
      desc->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, m_ * m_);
      desc->set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                      (1.0 / (m_ * m_ * m_)));
      // show_plan("yz", desc);
      desc->commit(q);
      fft_plans.emplace_back(desc);

      desc = new fft_plan_t(m_);
      desc->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                      (m_ * m_) / nprocs);
      desc->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, m_);
      desc->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, m_);
      // show_plan("x", desc);
      desc->commit(q);
      fft_plans.emplace_back(desc);
    }

    fmt::print("Initializing...\n");
    init_matrix_3d(in_, m_);
  }

  ~distributed_fft() {
    int i = fft_plans.size() - 1;
    while (i >= 0) {
      delete fft_plans[i];
      --i;
    }
  }

  void compute(int host_mode) {
    if(host_mode == USE_THREADS) {
      fft_threads(in_, out_, true);
      fft_threads(out_, in_, false);
    }
    if(host_mode == USE_OPENMP) {
      fft_omp(in_, out_, true);
      fft_omp(out_, in_, false);
    }
    else {
      fft_serial(in_, out_, true);
      fft_serial(out_, in_, false);
    }
  }

};

#ifdef STANDALONE_BENCHMARK

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

  omp_set_num_threads(devices.size());

  dim = (dim + ranks-1) / ranks * ranks;
  distributed_fft<real_t> fft3d(dim);

  {
    fft3d.compute(p_mode);
    double elapsed = 0;
    for (int iter = 0; iter < nreps; ++iter) {
      auto begin = std::chrono::steady_clock::now();
      fft3d.compute(p_mode);
      auto end = std::chrono::steady_clock::now();
      if(iter) 
        elapsed += std::chrono::duration<double>(end - begin).count();
    }

    std::size_t volume = dim * dim * dim;
    std::size_t fft_flops =  2 * static_cast<std::size_t>(5. * volume * std::log2(static_cast<double>(volume)));
    double t_avg = (nreps>1)? elapsed / (nreps - 1): 1e-9;
    fmt::print("fft3d-memcpy-{3} {0} {4} AvgTime {1:.3f} GFLOPS {2:.3f}\n", dim, t_avg, fft_flops/t_avg*1e-9, p_mode, ranks);

  }

  return 0;
}

#else

static void FFT3D_DR(benchmark::State &state) {
  std::size_t dim = check_results ? 8 : 768;

  std::size_t ranks = dr::shp::nprocs();
  dim = (dim + ranks - 1) / ranks * ranks; //make multiples of num devices

  distributed_fft<real_t> fft3d(dim);

  fft3d.compute();

  std::size_t volume = dim * dim * dim;
  std::size_t fft_flops =  2 * static_cast<std::size_t>(5. * volume * std::log2(static_cast<double>(volume)));

  Stats stats(state, 2 * sizeof(real_t) * volume, 4 * sizeof(real_t) * volume, fft_flops);

  for (auto _ : state) {
    for (std::size_t i = 0; i < default_repetitions; i++) {
      stats.rep();
      fft3d.compute();
    }
  }
}

DR_BENCHMARK(FFT3D_DR);

#endif // STANDALONE_BENCHMARK

// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include "cxxopts.hpp"
#include "oneapi/mkl/dfti.hpp"
#include <dr/shp.hpp>
#include <fmt/core.h>
#include <cmath>
#include <omp.h>
#include "transpose.hpp"

template <rng::forward_range R> auto values_view(R &&m) {
  return m | dr::shp::views::transform([](auto &&e) {
           auto &&[_, v] = e;
           return v;
         });
}

namespace fft {

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

template <typename T> struct dft_precision {
  static const oneapi::mkl::dft::precision value =
      oneapi::mkl::dft::precision::SINGLE;
};

template <> struct dft_precision<double> {
  static const oneapi::mkl::dft::precision value =
      oneapi::mkl::dft::precision::DOUBLE;
};

template <typename T> class distributed_fft {
  using fft_plan_t =
      oneapi::mkl::dft::descriptor<dft_precision<T>::value,
                                   oneapi::mkl::dft::domain::COMPLEX>;
  std::vector<fft_plan_t *> fft_yz_plans;
  std::vector<fft_plan_t *> fft_x_plans;

  void fft_impl(dr::shp::distributed_dense_matrix<std::complex<T>> &i_mat, 
              dr::shp::distributed_dense_matrix<std::complex<T>> &o_mat,
              bool forward) {
#pragma omp parallel 
    {
      int i = omp_get_thread_num();
      int nprocs = i_mat.segments().size();
      fft_plan_t* first_plan = (forward)? fft_yz_plans[i] : fft_x_plans[i];
      fft_plan_t* second_plan = (forward)? fft_x_plans[i] : fft_yz_plans[i];

      int lda = i_mat.shape()[1];
      int ldb = o_mat.shape()[1];
      int m_local = i_mat.shape()[0] / nprocs;
      int n_local = o_mat.shape()[0] / nprocs;

      auto &&atile = i_mat.tile({i, 0}); // local tile (m_local, lda)
      if(forward)
        oneapi::mkl::dft::compute_forward(*first_plan, dr::shp::__detail::local(atile).data()).wait();
      else
        oneapi::mkl::dft::compute_backward(*first_plan, dr::shp::__detail::local(atile).data()).wait();

      const size_t block_size = m_local * n_local;
      //pack In(m_local, lda) -> Out(nproc, m_local*n_local)
      std::vector<sycl::event> events;
      {
        auto &&packed_tile = o_mat.tile({i, 0});
        auto e0 = fft::transpose(m_local, n_local, atile.data(), lda, packed_tile.data(), ldb);
        events.emplace_back(e0);
        for (std::size_t j_ = 1; j_ < nprocs; j_++) {
          std::size_t j = (j_ + i) % std::size_t(nprocs);
          auto e = fft::copy_2d(m_local, n_local, atile.data() + j * n_local, lda, packed_tile.data() + j * block_size, n_local);
          events.emplace_back(e);
        }
      }
      sycl::event::wait(events);
      events.clear();
#pragma omp barrier
      //copy Out_i(j,blocksize) -> remote In_j(i, blocksize)
      for (std::size_t j_ = 1; j_ < nprocs; j_++) {
        std::size_t j = (j_ + i) % std::size_t(nprocs);
        auto &&send_tile = o_mat.tile({i, 0});
        auto &&recv_tile = i_mat.tile({j, 0});
        auto e = dr::shp::copy_async(send_tile.data() + j*block_size, send_tile.data() + (j+1)*block_size, 
            recv_tile.data() + i*block_size);
        events.emplace_back(e);
      }
      sycl::event::wait(events);
      events.clear();

      //unpack (transpose) In(nproc,blocksize) -> Out(n_local, ldb)
      auto &&btile = o_mat.tile({i, 0}); // local tile (n_local, ldb)
      for (std::size_t j_ = 1; j_ < nprocs; j_++) {
        std::size_t j = (j_ + i) % std::size_t(nprocs);
        auto &&unpacked_tile = i_mat.tile({i, 0});
        auto e = fft::transpose(m_local, n_local, unpacked_tile.data() + j * block_size, n_local, btile.data() + j * m_local, ldb);
        events.emplace_back(e);
      }

      //FFT on Out in-place
      if(forward)
        oneapi::mkl::dft::compute_forward(*second_plan, dr::shp::__detail::local(btile).data(), events).wait();
      else
        oneapi::mkl::dft::compute_backward(*second_plan, dr::shp::__detail::local(btile).data(), events).wait();
    }
}


public:
  explicit distributed_fft(std::int64_t m, int nprocs) {
    fft_yz_plans.resize(nprocs, nullptr);
    fft_x_plans.resize(nprocs, nullptr);

    int m_local = m / nprocs;
#pragma omp parallel for
    for (int i = 0; i < nprocs; i++) {
      auto &&q = dr::shp::__detail::queue(i);
      fft_plan_t *desc = new fft_plan_t({m, m});
      desc->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                      m_local);
      desc->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, m * m);
      desc->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, m * m);
      desc->set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                      (1.0 / (m * m * m)));
      desc->commit(q);
      fft_yz_plans[i] = desc;

      desc = new fft_plan_t(m);
      desc->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                      (m * m) / nprocs);
      desc->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, m);
      desc->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, m);
      desc->commit(q);
      fft_x_plans[i] = desc;
    }
  }

  ~distributed_fft() {
    //mkl_free_buffers();
    int i = fft_yz_plans.size() - 1;
    while (i >= 0) {
      delete fft_x_plans[i];
      delete fft_yz_plans[i];
      --i;
    }
  }

  void
  compute_forward(dr::shp::distributed_dense_matrix<std::complex<T>> &i_mat,
                  dr::shp::distributed_dense_matrix<std::complex<T>> &o_mat) {
    fft_impl(i_mat, o_mat, true);
  }

  void
  compute_backward(dr::shp::distributed_dense_matrix<std::complex<T>> &i_mat,
                   dr::shp::distributed_dense_matrix<std::complex<T>> &o_mat) {
    fft_impl(i_mat, o_mat, false);
  }
};

} // namespace fft

int main(int argc, char **argv) {

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

  // 512^3 up to 3072
  std::size_t dim = options["n"].as<std::size_t>();
  std::size_t nreps = options["repetitions"].as<std::size_t>();

  auto devices = dr::shp::get_numa_devices(sycl::default_selector_v);
  std::size_t nprocs = options["num-devices"].as<std::size_t>();
  if (nprocs != 0 && nprocs < devices.size()) {
    devices.resize(nprocs);
  }
  //for (auto device : devices) {
  //  fmt::print("Device: {}\n", device.get_info<sycl::info::device::name>());
  //}
  dr::shp::init(devices);

  std::size_t m_local = (dim + nprocs - 1) / nprocs;
  std::size_t m = nprocs * m_local;
  std::size_t n = m * m;

  using real_t = float;
  using value_t = std::complex<real_t>;
  if (n * m_local * sizeof(value_t) * 1e-9 > 18.0) {
    fmt::print("Too big: reduce problem size  \n");
    return 0;
  }

  omp_set_num_threads(nprocs);

  fmt::print("Dims {}^3 -> {}^3, Transfer size {} GB \n", dim, m,
             sizeof(value_t) * m * n * 1e-9);

  fmt::print("Allocating...\n");
  dr::shp::block_cyclic row_blocks({dr::shp::tile::div, dr::shp::tile::div},
                                   {dr::shp::nprocs(), 1});

  dr::shp::distributed_dense_matrix<value_t> i_mat({m, n}, row_blocks);
  dr::shp::distributed_dense_matrix<value_t> o_mat({n, m}, row_blocks);
  fmt::print("Creating plans...\n");
  fft::distributed_fft<real_t> fft3d(m, nprocs);

  fmt::print("Initializing... {} threads\n",omp_get_max_threads());
  fft::init_matrix_3d(i_mat, m);

  if (nreps == 0) { // debug
    dr::shp::distributed_dense_matrix<value_t> t_mat({m, n}, row_blocks);
    fft::init_matrix_3d(t_mat, m);

    fft3d.compute_forward(i_mat, o_mat);
    fft3d.compute_backward(o_mat, i_mat);
    auto sub_view =
        dr::shp::views::zip(values_view(i_mat), values_view(t_mat)) |
        dr::shp::views::transform([](auto &&e) {
          auto &&[value, ref] = e;
          return value - ref;
        });
    auto diff_sum = dr::shp::reduce(dr::shp::par_unseq, sub_view, value_t{});
    fmt::print("Difference {} {} \n", diff_sum.real(), diff_sum.imag());
  }
  else
  {
    std::vector<double> for_timers(nreps);
    std::vector<double> back_timers(nreps);

    for (int iter = 0; iter < nreps; ++iter) {
      auto begin = std::chrono::high_resolution_clock::now();
      fft3d.compute_forward(i_mat, o_mat);
      auto end = std::chrono::high_resolution_clock::now();
      for_timers[iter] = std::chrono::duration<double>(end - begin).count();

      begin = std::chrono::high_resolution_clock::now();
      fft3d.compute_backward(o_mat, i_mat);
      end = std::chrono::high_resolution_clock::now();
      back_timers[iter] = std::chrono::duration<double>(end - begin).count();
    }

    std::sort(for_timers.begin(), for_timers.end());
    std::sort(back_timers.begin(), back_timers.end());

    double volume = static_cast<double>(m * m * m);
    double fft_gbytes = 2.0 * volume  *  sizeof(value_t) * 1e-9;
    double fft_gflops = 5. * volume * std::log2(volume) *  1e-9;

    double duration = (for_timers[nreps/2] + back_timers[nreps/2]);
    fmt::print("Type n tiles GFLOPS GB/sec secs\n");
    fmt::print("{0} {1} {2} {3:.3f} sec/call {4:.3f} GFLOPS {5:.3f} GB/s\n", 
        "FFT3D_copy_omp", dim, nprocs, duration, 2*fft_gflops/duration, 2*fft_gbytes/duration);
  }


  return 0;
}

// SPDX-FileCopyrightText: Intel Corporation
//
// SPDX-License-Identifier: BSD-3-Clause

#include <dr/mhp.hpp>
#include <fmt/core.h>

namespace mhp = dr::mhp;

/* The example simulates the elementary 1-d cellular automaton. Description of
 * what the automaton is and how it works can be found at
 * https://en.wikipedia.org/wiki/Elementary_cellular_automaton
 * Visulisation of the automaton work is available
 * https://elife-asu.github.io/wss-modules/modules/1-1d-cellular-automata
 * (credit: Emergence team @ Arizona State University)*/

constexpr std::size_t asize = 60;
constexpr uint8_t ca_rule = 28;
constexpr std::size_t steps = 90;

auto newvalue = [](auto &&p) {
  auto &[in, out] = p;
  auto v = &in;

  uint8_t pattern = 4 * v[-1] + 2 * v[0] + v[1];
  out = (ca_rule >> pattern) % 2;
};

int main(int argc, char **argv) {
#ifdef SYCL_LANGUAGE_VERSION
  mhp::init(sycl::default_selector_v);
#else
  mhp::init();
#endif

  /* the algorithm will reach to 1-cell neigbourhood, possibly located in
   * different node */
  auto dist = dr::mhp::distribution().halo(1);

  /* size of the vector include cells for 0's at the beginning and end of
   * automaton - no special conditions are neccessary */
  mhp::distributed_vector<uint8_t> automaton1(asize + 2, 0, dist);
  mhp::distributed_vector<uint8_t> automaton2(asize + 2, 0, dist);

  /* the algorithm works on subrange - the extra cells are nedded only for
   * calculation of new values at the edges, but are never set */
  auto in = rng::subrange(automaton1.begin() + 1, automaton1.end() - 1);
  auto out = rng::subrange(automaton2.begin() + 1, automaton2.end() - 1);

  /* initial value of the automaton - set this in a way that suits you */
  in[0] = 1;

  /* print the first value of the automaton */
  if (mhp::rank() == 0)
    fmt::print("{}\n", in);

  for (std::size_t s = 0; s < steps; s++) {
    /* exchange the halo values in case they were set and need to be updated
     * also in neighbouring nodes */
    dr::mhp::halo(in).exchange();

    /* the proper loop calculating new values of automaton's cells */
    mhp::for_each(mhp::views::zip(in, out), set_cell);

    /* swap input and output vectors, to keep automaton state always in the same
     * variable */
    std::swap(in, out);

    /* now print the state of the automaton */
    if (mhp::rank() == 0)
      fmt::print("{}\n", in);
  }
}
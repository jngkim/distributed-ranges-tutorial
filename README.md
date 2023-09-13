# distributed-ranges-tutorial

Getting-started resources for oneAPI's distributed-ranges library

## 1. Introduction

The distributed-ranges (dr) library is a replacement of some data structure, container and algorithms of C++20 Standard Template Library. It is designed to run on HPC clusters, consisting of multiple CPUs and GPUs. It takes advantage of parallel processing in distributed memory model and MPI communication as well as parallel processing in shared memory model. The interface is similar to the one used in std and std::ranges namespaces, however inevitable differences exist.
If you are familiar with C++ template libraries, and particularly std::ranges (C++20) or ranges-v3 (C++11 - C++17), the tutorial will help you to start using distributed-ranges, along with understanding the necessary differences between dr and other libraries.

## 2. Download and install

There is no official installer for dr. Currently, there are two ways to start work with distributed-ranges.

If you want to start as a user, and your developement environment is connected to the Internet, we encourage you to clone this [distributed-ranges-tutorial](https://github.com/intel/distributed-ranges-tutorial) and modify examples provided. The cmake files provided in the skeleton repo will download the dr library as a source code (using cmake) and build the examples, there is no need for separate install.

If you want to use distributed-ranges offline, or the default configuration is not appropriate for you, just clone the [official distributed-ranges repo](https://github.com/oneapi-src/distributed-ranges/) and use the content of *./include* directory in your project.

If you want to contribute to distributed-ranges, please also go to [distributed-ranges repo](https://github.com/oneapi-src/distributed-ranges/)

## 3. Description

### 3.1 Namespaces

General namespace used in the library is dr::
For program using a single node with shared memory available for multiple CPUs and one or more GPUs, data structures and algoritms from dr::shp:: namespace are provided.
For distributed memory model, use the dr::mhp:: namespace.

### 3.2 Data structures

Basic data structure in dr library is distributed_vector. The elements of the vector are distributed over available nodes. In particular, segments of dr::mhp::distributed_vector are located in memory of different nodes (mpi processes). Still, global view of the distributed_vector is uniform, with contigous indices.
(pictures from Ben's presentations - segments and global view)

#### 3.2.1 Halo concept

When implementing your algorithm over the distributed vector, you have to be aware of its segmented internal structure. The issue occurs, when your algorithm refers to neighbouring cells of the current one, and your local loop reaches begin or end of the segment. In this moment neighbouring cells are located in physical memory of another node!
To support the situation, the concept of halo is provided. Halo is an area, where content of egde elements of neighbouring segment is copied. Also changes in halo are copied to cells in appropriate segment, to maintain consistency of the whole vector. The concept of halo is described in details in Example 4.2

### 3.3 Algorithms

## 4. Examples (explained)

### 4.1 Hello world

The first program presents declaration of *distributed_vector<>* and reveals its distribution over MPI nodes.
It uses dr::mhp namespace and requires MPI to be installed in the target environment.
The example should be run with mpirun -n *N* ./hello_world, where *N* is a number of your MPI processes.

This example comes from [distributed-ranges repo](https://github.com/oneapi-src/distributed-ranges/)
from the file ./examples/mhp/hello_world.cpp


    // SPDX-FileCopyrightText: Intel Corporation
    //
    // SPDX-License-Identifier: BSD-3-Clause

    #include <dr/mhp.hpp>
    #include <fmt/core.h>

    namespace mhp = dr::mhp;

    int main(int argc, char **argv) {
    #ifdef SYCL_LANGUAGE_VERSION
      mhp::init(sycl::default_selector_v);
    #else
      mhp::init();
    #endif

        {   
            fmt::print("Hello, World! Distributed ranges is running on rank {} / {} on host {}\n",
                        mhp::rank(), mhp::nprocs(), mhp::hostname());

            std::size_t n = 1000;

            mhp::distributed_vector<int> v(n);

            if (mhp::rank() == 0) {
                auto &&segments = v.segments();

                fmt::print("Created distributed_vector of size {} with {} segments.\n",
                            v.size(), segments.size());

                std::size_t segment_id = 0;
                for (auto &&segment : segments) {
                    fmt::print("Rank {} owns segment {}, which is size {}\n",
                                dr::ranges::rank(segment), segment_id, segment.size());
                    ++segment_id;
                }
            }
        }

        mhp::finalize();
        return 0;
    }

### 4.2 Elementary cellular automaton (TBD)

* 1-d example

* halo explained
Problem is described [here](https://en.wikipedia.org/wiki/Elementary_cellular_automaton/) and [here](https://elife-asu.github.io/wss-modules/modules/1-1d-cellular-automata/)

### 4.3 PI calculation (TBD)

* Monte Carlo method

* 2-d, using mdspan

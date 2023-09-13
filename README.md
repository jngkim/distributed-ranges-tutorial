# Introduction

The distributed-ranges (dr) library is a replacement for some data structures, containers, and algorithms of the C++20 Standard Template Library. It is designed to run on HPC clusters consisting of multiple CPUs and GPUs. It takes advantage of parallel processing and MPI communication in distributed memory model as well as parallel processing in shared memory model. The interface is similar to that of std and std::ranges namespaces with some unavoidable differences.
If you are familiar with the C++ Template Libraries, and in particular std::ranges (C++20) or ranges-v3 (C++11 -- C++17), switching to dr will be straightforward, but this tutorial will help you get started even if you have never used them. However, we assume that you are familiar with C++, at least in the C++11 standard (C++20 is recommended).

# Prerequisites

The distributed-ranges library can be used on any system with a working SYCL or g++ compiler. Intel's DPC++ (implementation of the SYCL standard) is recommended. g++ v. 10, 11 or 12 is also supported, but GPU usage is not possible.
Distributed-ranges depends on MPI and MKL libraries. 
DPC++, oneMKL and oneMPI are part of the oneAPI suite. It is recommended to install oneAPI before downloading distributed-ranges.

# Getting started

Currently, there are two ways to start work with distributed-ranges.

If you want to start as a user, and your development environment is connected to the Internet, we encourage you to clone this [distributed-ranges-tutorial](https://github.com/intel/distributed-ranges-tutorial) repository and modify examples provided. The cmake files provided in the skeleton repo will download the dr library as a source code (using cmake) and build the examples, there is no need for separate install.

- In Linux system (bash shell) download distributed-ranges-tutorial from GitHub and build with the following commands 
	```shell
	git clone https://github.com/mateuszpn/distributed-ranges-tutorial
	cd distributed-ranges-tutorial
	CXX=icpx CC=icx cmake -B build
	cmake --build build
	```
- If you want to contribute to distributed-ranges or go through more advanced examples, please go to original distributed-ranges [GitHub repository](https://github.com/oneapi-src/distributed-ranges/)
	```shell
	git clone https://github.com/oneapi-src/distributed-ranges
	cd distributed-ranges
	CXX=icpx CC=icx cmake -B build -DENABLE_SYCL=ON
	cmake --build build -j
	```
	If you have a compiler different than DPC++, change CXX and CC values respectively.

- In case your environment is not configured properly or you just prefer a hassle-free code exploration you can use Docker.
	```shell
	git clone https://github.com/mateuszpn/distributed-ranges-tutorial
	cd distributed-ranges-tutorial
	docker run -it -v $(pwd):/custom-directory-name -u root docker.io/intel/oneapi:latest /bin/bash
	cd custom-directory-name 
	CXX=icpx CC=icx cmake -B build -DENABLE_SYCL=ON
	cmake --build build -j
	```
	where 'custom-directory-name' stands for the name of a directory containing local repo data on a docker volume

# 1. Description

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

4. Examples 
-------------------------
### 4.1 Hello world 
(this example comes from the [distributed-ranges repo](https://github.com/oneapi-src/distributed-ranges/))

Source file: [./src/hello_world.cpp](src/hello_world.cpp)
	
The first program presents declaration of distributed_vector<> and reveals its distribution over MPI nodes. The example is intended to run in distributed environment, with MPI-based communication. We discuss now parts of the code

We use dr::mhp namespace, which contains algorithms and data structures with MPI support.
```cpp
	namespace mhp = dr::mhp;
```
mhp::rank(), mhp::nprocs(), mhp::hostname() are wrappers of MPI functions, allowing us to get number (rank) of particular MPI process, total number of MPI processes (MPI default communicator size) and host name.
```cpp
	fmt::print("Hello, World! Distributed ranges is running on rank {} / {} on host {}\n",
				mhp::rank(), mhp::nprocs(), mhp::hostname());
```

Declaration of distributed vector (corresponding to std::vector). Distributed vector is basic data structure for distributed ranges. It is automatically divided into segments, which are distributed over MPI processes. 
Please note the declaration is very simple - it's distribution over available nodes is fully automatic.
```cpp
	mhp::distributed_vector<int> v(n);
```
Execution of some code can be limited to particular node.
```cpp
	if (mhp::rank() == 0) {
```
If necessary, you can reach the information about physical distribution of data. However, as you will see in next examples, it is not always necessary.
```cpp
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
``` 


	
### 4.2 1-D vector example
#### Elementary cellular automaton

ECA problem is described in [wikipedia](https://en.wikipedia.org/wiki/Elementary_cellular_automaton) and visualized in [ASU team](https://elife-asu.github.io/wss-modules/modules/1-1d-cellular-automata/) web page.

You can find the example in the [./src/cellular_automaton.cpp](src/cellular_automaton.cpp) file. The code is explained in comments inside.

(Some halo description - TBD)

### 4.3 Simple 2-3 operation 
#### Find a pattern in the randomly filled array.



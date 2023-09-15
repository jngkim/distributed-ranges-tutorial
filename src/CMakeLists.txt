# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause


add_executable(example1 example1.cpp)

target_compile_definitions(example1 INTERFACE DR_FORMAT)
target_link_libraries(example1 DR::mpi fmt::fmt)

add_executable(example2 example2.cpp)

target_compile_definitions(example2 INTERFACE DR_FORMAT)
target_link_libraries(example2 DR::mpi fmt::fmt)

add_executable(example3 example3.cpp)

target_compile_definitions(example3 INTERFACE DR_FORMAT)
target_link_libraries(example3 DR::mpi fmt::fmt)

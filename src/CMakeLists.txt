# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause


add_executable(hello_world hello_world.cpp)

target_compile_definitions(hello_world INTERFACE DR_FORMAT)
target_link_libraries(hello_world DR::mpi fmt::fmt)

add_executable(cellular_automaton cellular_automaton.cpp)

target_compile_definitions(cellular_automaton INTERFACE DR_FORMAT)
target_link_libraries(cellular_automaton DR::mpi fmt::fmt)

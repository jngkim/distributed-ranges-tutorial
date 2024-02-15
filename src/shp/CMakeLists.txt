# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

function(add_shp_example example_name)
  add_executable(${example_name} ${example_name}.cpp)
  target_compile_definitions(${example_name} INTERFACE DR_FORMAT)
  target_compile_options(${example_name} PRIVATE "-fopenmp")
  target_link_options(${example_name} PRIVATE "-fopenmp")
  target_link_libraries(${example_name} DR::shp fmt::fmt cxxopts)
endfunction()

add_shp_example(dot_product)
add_shp_example(fft3d_omp)
add_shp_example(fft3d_copy_omp)
add_shp_example(fft3d)

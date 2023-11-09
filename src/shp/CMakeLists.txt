# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause


add_executable(dot_product dot_product.cpp)

target_compile_definitions(dot_product INTERFACE DR_FORMAT)
target_link_libraries(dot_product DR::shp fmt::fmt)

# Copyright 2026 Hongpei Li
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

project(PreFOSConsumer LANGUAGES C)

find_package(PreFOS 0.1 CONFIG REQUIRED)

add_executable(prefos_consumer main.c)
target_link_libraries(prefos_consumer PRIVATE PreFOS::PreFOS)

enable_testing()
add_test(NAME prefos_consumer COMMAND prefos_consumer)

# Copyright 2026 Hongpei Li
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

project(PreFOS VERSION 0.1.0 LANGUAGES C)

include(GNUInstallDirs)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(PREFOS_BUILD_TESTING "Build the PreFOS test suite" OFF)
option(PREFOS_ENABLE_CUDA "Build the experimental CUDA propagation backend" OFF)
option(PREFOS_ENABLE_COVERAGE "Enable coverage instrumentation" OFF)

if(PREFOS_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "PREFOS_ENABLE_CUDA=ON requires an NVIDIA CUDA compiler. "
      "Configure with PREFOS_ENABLE_CUDA=OFF for a CPU-only build."
    )
  endif()
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.2)
    message(FATAL_ERROR "PreFOS CUDA propagation requires CUDA 11.2 or newer")
  endif()
  message(STATUS "PreFOS CUDA propagation backend: enabled")
else()
  message(STATUS "PreFOS CUDA propagation backend: disabled (CPU fallback)")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug Release MinSizeRel RelWithDebInfo
  )
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_EXTENSIONS OFF)

set(PREFOS_SOURCES
  src/common/ParallelRowDetection.c
  src/common/ParallelRowReduction.c
  src/prefos/PREFOS_Presolver.c
  src/prefos/cones/PREFOS_ExponentialCone.c
  src/prefos/cones/PREFOS_PositiveSemidefiniteCone.c
  src/prefos/cones/PREFOS_PowerCone.c
  src/prefos/core/PREFOS_AffineConeCompaction.c
  src/prefos/core/PREFOS_AffineConeFaces.c
  src/prefos/core/PREFOS_MatrixCompaction.c
  src/prefos/core/PREFOS_Postsolve.c
  src/prefos/core/PREFOS_Problem.c
  src/prefos/core/PREFOS_Timer.c
  src/prefos/explorers/PREFOS_AffineConeAggregation.c
  src/prefos/explorers/PREFOS_AffineConeBounds.c
  src/prefos/explorers/PREFOS_AffineConePropagation.c
  src/prefos/explorers/PREFOS_AffineFaceElimination.c
  src/prefos/explorers/PREFOS_AffineFaceRows.c
  src/prefos/explorers/PREFOS_AffineFaceSubstitution.c
  src/prefos/explorers/PREFOS_ConeActivity.c
  src/prefos/explorers/PREFOS_ConePropagation.c
  src/prefos/explorers/PREFOS_CudaBackend.c
  src/prefos/explorers/PREFOS_BoundedDoubletons.c
  src/prefos/explorers/PREFOS_ColumnFixing.c
  src/prefos/explorers/PREFOS_ColumnReductions.c
  src/prefos/explorers/PREFOS_ColumnSubstitution.c
  src/prefos/explorers/PREFOS_ColumnWorkspace.c
  src/prefos/explorers/PREFOS_FreeColumnSubstitution.c
  src/prefos/explorers/PREFOS_LinearPropagation.c
  src/prefos/explorers/PREFOS_ParallelColumns.c
  src/prefos/explorers/PREFOS_ParallelRows.c
  src/prefos/explorers/PREFOS_RedundantBounds.c
  src/prefos/explorers/PREFOS_SDPStructureAnalysis.c
  src/prefos/explorers/PREFOS_SingletonColumns.c
)

if(PREFOS_ENABLE_CUDA)
  list(APPEND PREFOS_SOURCES
    src/prefos/explorers/PREFOS_CudaAffineConePropagation.cu
    src/prefos/explorers/PREFOS_CudaColumnWorkspace.cu
    src/prefos/explorers/PREFOS_CudaConeActivity.cu
    src/prefos/explorers/PREFOS_CudaConePropagation.cu
    src/prefos/explorers/PREFOS_CudaLinearPropagation.cu
    src/prefos/explorers/PREFOS_CudaMatrixCompaction.cu
    src/prefos/explorers/PREFOS_CudaParallelColumns.cu
    src/prefos/explorers/PREFOS_CudaParallelRows.cu
  )
else()
  list(APPEND PREFOS_SOURCES
    src/prefos/explorers/PREFOS_CudaLinearPropagationStub.c
  )
endif()

add_library(PreFOS ${PREFOS_SOURCES})
add_library(PreFOS::PreFOS ALIAS PreFOS)

target_compile_features(PreFOS PUBLIC c_std_99)
target_include_directories(PreFOS
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/prefos
    ${CMAKE_CURRENT_SOURCE_DIR}/include/common
)

set_target_properties(PreFOS PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  C_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(PreFOS
    PRIVATE PREFOS_BUILDING_SHARED_LIBRARY=1
    INTERFACE PREFOS_USING_SHARED_LIBRARY=1
  )
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_compile_definitions(PreFOS PRIVATE _POSIX_C_SOURCE=200809L)
endif()

if(PREFOS_ENABLE_CUDA)
  find_package(Threads REQUIRED)
  set_target_properties(PreFOS PROPERTIES
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
  )
  target_compile_definitions(PreFOS PRIVATE PREFOS_HAS_CUDA=1)
  target_link_libraries(PreFOS PRIVATE Threads::Threads)
endif()

if(MSVC)
  target_compile_options(PreFOS PRIVATE /W4)
else()
  target_compile_options(PreFOS PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Wpedantic;-Wconversion>
    $<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Debug>>:-g;-O0>
    $<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Release>>:-O3;-DNDEBUG>
    $<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:RelWithDebInfo>>:-O2;-g;-DNDEBUG>
    $<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:MinSizeRel>>:-Os;-DNDEBUG>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G;-O0>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3;-DNDEBUG>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:-O2;-lineinfo;-DNDEBUG>
  )
endif()

if(UNIX)
  target_link_libraries(PreFOS PRIVATE m)
endif()

if(PREFOS_ENABLE_COVERAGE)
  if(NOT CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "Coverage instrumentation requires GCC or Clang")
  endif()
  target_compile_options(PreFOS PRIVATE -O0 -g --coverage)
  target_link_options(PreFOS PRIVATE --coverage)
endif()

if(PREFOS_BUILD_TESTING)
  enable_language(CXX)
  include(CTest)
  enable_testing()

  add_executable(prefos_tests tests/test_prefos.c)
  target_link_libraries(prefos_tests PRIVATE PreFOS::PreFOS)
  add_test(NAME prefos_tests COMMAND prefos_tests)

  add_executable(prefos_column_reduction_tests tests/test_column_reductions.c)
  target_link_libraries(prefos_column_reduction_tests PRIVATE PreFOS::PreFOS)
  add_test(NAME prefos_column_reduction_tests COMMAND prefos_column_reduction_tests)

  add_executable(prefos_cpp_tests tests/test_prefos_cpp.cpp)
  target_compile_features(prefos_cpp_tests PRIVATE cxx_std_11)
  target_link_libraries(prefos_cpp_tests PRIVATE PreFOS::PreFOS)
  add_test(NAME prefos_cpp_tests COMMAND prefos_cpp_tests)

  if(PREFOS_ENABLE_COVERAGE)
    target_compile_options(prefos_tests PRIVATE -O0 -g --coverage)
    target_link_options(prefos_tests PRIVATE --coverage)
    target_compile_options(prefos_column_reduction_tests PRIVATE -O0 -g --coverage)
    target_link_options(prefos_column_reduction_tests PRIVATE --coverage)
  endif()
endif()

install(TARGETS PreFOS
  EXPORT PreFOSTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/PreFOS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE NOTICE DESTINATION ${CMAKE_INSTALL_DATADIR}/PreFOS)

install(EXPORT PreFOSTargets
  FILE PreFOSTargets.cmake
  NAMESPACE PreFOS::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PreFOS
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/PreFOSConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/PreFOSConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PreFOS
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/PreFOSConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/PreFOSConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/PreFOSConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PreFOS
)

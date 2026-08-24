# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.14)
project(ailee_video_temporal_provenance VERSION 7.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include directories
include_directories(include)

# C++ Core Library
add_library(ailee_video_temporal_provenance SHARED
    src/ailee_video_temporal_provenance.cpp
)

target_include_directories(ailee_video_temporal_provenance PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Enable CTest
enable_testing()

# Unit Test Binary
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ailee_video_temporal_provenance.cpp")
    add_executable(test_ailee_video_temporal_provenance
        tests/test_ailee_video_temporal_provenance.cpp
    )
    target_link_libraries(test_ailee_video_temporal_provenance PRIVATE ailee_video_temporal_provenance)
    add_test(NAME TestAileeVideoTemporalProvenance COMMAND test_ailee_video_temporal_provenance)
endif()

# Demo Binary
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/ailee_video_temporal_demo.cpp")
    add_executable(ailee_video_temporal_demo
        examples/ailee_video_temporal_demo.cpp
    )
    target_link_libraries(ailee_video_temporal_demo PRIVATE ailee_video_temporal_provenance)
endif()

cmake_minimum_required(VERSION 3.20)

project(stable_diffusion)

set(CMAKE_CXX_STANDARD 11)

if(DEFINED ENV{ARTIFACTS_DIR})
  set(ARTIFACTS_DIR "$ENV{ARTIFACTS_DIR}")
else()
  message(WARNING "ARTIFACTS_DIR env var not defined, abort")
endif()

set(OpenCV_DIR "${ARTIFACTS_DIR}/opencv/build/x64/vc15/lib" CACHE STRING "Look for the OpenCVConfig.cmake file by OpenCV_DIR variables")
set(ncnn_DIR "${ARTIFACTS_DIR}/ncnn/build/install/lib/cmake/ncnn" CACHE STRING "Look for the ncnnConfig.cmake file by ncnn_DIR variables")

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

add_executable(testbed 
  main.cpp
  prompt_slover.cpp
  diffusion_slover.cpp
  decoder_slover.cpp
  prompt_slover.h
  diffusion_slover.h
  decoder_slover.h
)

target_link_libraries(testbed
  ncnn
  ${OpenCV_LIBS}
)

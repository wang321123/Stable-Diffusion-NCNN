cmake_minimum_required(VERSION 3.20)

project(stable_diffusion)

set(CMAKE_CXX_STANDARD 11)

### Look for the OpenCVConfig.cmake file by OpenCV_DIR variables ###
set(OpenCV_DIR "D:/wzp2022/Libary/opencv4.6/opencv/build/x64/vc15/lib") 

### Look for the ncnnConfig.cmake file by ncnn_DIR variables
set(ncnn_DIR "D:/wzp2022/github/ncnn/build/install/lib/cmake/ncnn")

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

add_executable(testbed 
    vs2019_opencv-mobile_ncnn-dll_demo.cpp
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

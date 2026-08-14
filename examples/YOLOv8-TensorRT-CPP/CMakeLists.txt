cmake_minimum_required(VERSION 3.16)

project(YOLOv8TensorRTInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Prefer the CUDA installation selected by CUDA_HOME when it is set.  This
# keeps CMake from accidentally mixing headers and libraries from another
# CUDA installation on machines with multiple toolkits installed.
if(DEFINED ENV{CUDA_HOME} AND NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_HOME}" CACHE PATH "CUDA toolkit root")
endif()

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc dnn)

set(TensorRT_ROOT "" CACHE PATH "TensorRT installation prefix")

# The project can use either TensorRT_ROOT or the environment variables from
# the supplied x86 installation.  The latter are useful when TensorRT is not
# installed system-wide.
if(NOT TensorRT_ROOT AND DEFINED ENV{TensorRT_Inc})
    get_filename_component(TensorRT_ROOT "$ENV{TensorRT_Inc}" DIRECTORY)
endif()

set(TensorRT_INCLUDE_HINTS
    ${TensorRT_ROOT}
    $ENV{TensorRT_Inc})
set(TensorRT_LIBRARY_HINTS
    ${TensorRT_ROOT}
    $ENV{TensorRT_Lib})

find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h NvOnnxParser.h
    HINTS ${TensorRT_INCLUDE_HINTS} $ENV{TensorRT_ROOT}
    PATHS /usr /usr/local /opt/tensorrt
    PATH_SUFFIXES include include/x86_64-linux-gnu)

find_library(TensorRT_NVINFER_LIBRARY
    NAMES nvinfer
    HINTS ${TensorRT_LIBRARY_HINTS} $ENV{TensorRT_ROOT}
    PATHS /usr /usr/local /opt/tensorrt
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)

find_library(TensorRT_ONNX_PARSER_LIBRARY
    NAMES nvonnxparser
    HINTS ${TensorRT_LIBRARY_HINTS} $ENV{TensorRT_ROOT}
    PATHS /usr /usr/local /opt/tensorrt
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)

find_library(TensorRT_PLUGIN_LIBRARY
    NAMES nvinfer_plugin
    HINTS ${TensorRT_LIBRARY_HINTS} $ENV{TensorRT_ROOT}
    PATHS /usr /usr/local /opt/tensorrt
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)

if(NOT TensorRT_INCLUDE_DIR OR NOT TensorRT_NVINFER_LIBRARY OR
   NOT TensorRT_ONNX_PARSER_LIBRARY OR NOT TensorRT_PLUGIN_LIBRARY)
    message(FATAL_ERROR
        "TensorRT C++ headers/libraries were not found. Set TensorRT_ROOT to a TensorRT installation "
        "containing include/NvInfer.h and lib/libnvinfer.so.")
endif()

add_executable(yolov8_tensorrt
    main.cpp
    tensorrt_yolov8.cpp
    tensorrt_yolov8.h)

target_include_directories(yolov8_tensorrt PRIVATE
    ${TensorRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(yolov8_tensorrt PRIVATE
    ${CUDA_CUDART_LIBRARY}
    ${TensorRT_NVINFER_LIBRARY}
    ${TensorRT_ONNX_PARSER_LIBRARY}
    ${TensorRT_PLUGIN_LIBRARY}
    ${OpenCV_LIBS})

# Make the executable find CUDA and TensorRT without requiring LD_LIBRARY_PATH
# every time it is launched.  The explicit environment paths are optional.
if(DEFINED ENV{TensorRT_Lib})
    target_link_directories(yolov8_tensorrt PRIVATE "$ENV{TensorRT_Lib}")
endif()
if(DEFINED ENV{CUDA_Lib})
    target_link_directories(yolov8_tensorrt PRIVATE "$ENV{CUDA_Lib}")
endif()
set_target_properties(yolov8_tensorrt PROPERTIES
    BUILD_RPATH "$ENV{TensorRT_Lib};$ENV{CUDA_Lib}"
    INSTALL_RPATH "$ENV{TensorRT_Lib};$ENV{CUDA_Lib}")

# TensorRT's nvinfer library loads nvinfer_plugin indirectly.  DT_RUNPATH is
# not searched for indirect dependencies by the dynamic loader, so use the
# older DT_RPATH tag here to make the self-contained TensorRT installation
# work without requiring LD_LIBRARY_PATH at runtime.
target_link_options(yolov8_tensorrt PRIVATE "-Wl,--disable-new-dtags")

target_compile_options(yolov8_tensorrt PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)

message(STATUS "TensorRT include: ${TensorRT_INCLUDE_DIR}")
message(STATUS "TensorRT nvinfer: ${TensorRT_NVINFER_LIBRARY}")
message(STATUS "TensorRT ONNX parser: ${TensorRT_ONNX_PARSER_LIBRARY}")
message(STATUS "TensorRT plugin: ${TensorRT_PLUGIN_LIBRARY}")

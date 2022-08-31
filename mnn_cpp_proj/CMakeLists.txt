cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(wenet_mnn VERSION 0.1)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
)

set(CMAKE_VERBOSE_MAKEFILE on)

# utils
add_library(utils STATIC
  utils/mnn_utils.cc
)
# frontend
add_library(frontend STATIC
  frontend/mnn_feature_pipeline.cc
  frontend/fft.cc
)
target_link_libraries(frontend PUBLIC utils)

set(MNN_DIR /data/kzx/work/MNN)
include_directories(${MNN_DIR}/include)
include_directories(${MNN_DIR}/include/MNN)
include_directories(${MNN_DIR}/include/MNN/expr)
include_directories(${MNN_DIR}/tools)
include_directories(${MNN_DIR}/tools/cpp)
include_directories(${MNN_DIR}/source)
include_directories(${MNN_DIR}/source/backend)
include_directories(${MNN_DIR}/source/core)
LINK_DIRECTORIES(${MNN_DIR}/build)
LINK_DIRECTORIES(${MNN_DIR}/build/express)

# decoder
add_library(decoder STATIC
  decoder/mnn_decoder_expr.cc
  decoder/ctc_prefix_beam_search.cc
)

target_link_libraries(decoder PUBLIC utils -lMNN)
target_link_libraries(decoder PUBLIC ${MNN_DIR}/build/express/libMNN_Express.so)

# binary
add_executable(mnn_decoder_main bin/mnn_decoder_main.cc)
target_link_libraries(mnn_decoder_main PUBLIC decoder frontend)


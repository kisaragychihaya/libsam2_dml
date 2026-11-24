cmake_minimum_required(VERSION 3.20)
project(sam2_ncnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Backend: pure CPU (default) or Vulkan. Vulkan builds require the Vulkan SDK.
option(SAM2_VULKAN "Build with the ncnn Vulkan backend" OFF)

# ncnn: either a prebuilt install prefix (CI caches it, see
# .github/workflows/build-and-qc.yml) or fetched from source here.
set(NCNN_ROOT "" CACHE PATH "Prebuilt ncnn install prefix (include/ + lib/)")
# 0a2c46bc75 = 2026-08-25 master, the era the converted models were validated
# against (the 20260827 pyncnn wheel). 20260526 mis-computes the deeper
# encoder stages; 20261020+ segfault on the decoder graph.
set(NCNN_TAG "0a2c46bc75987a39d3b044c77385c564e83c0a84" CACHE STRING "Tencent/ncnn git tag used by the FetchContent fallback")

if(NCNN_ROOT)
  add_library(ncnn STATIC IMPORTED GLOBAL)
  unset(NCNN_LIB CACHE)
  find_library(NCNN_LIB NAMES ncnn libncnn PATHS "${NCNN_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
  set_target_properties(ncnn PROPERTIES
    IMPORTED_LOCATION "${NCNN_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${NCNN_ROOT}/include")
else()
  include(FetchContent)
  set(NCNN_VULKAN ${SAM2_VULKAN} CACHE BOOL "" FORCE)
  set(NCNN_SHARED_LIB OFF CACHE BOOL "" FORCE)      # static: everything lands in sam2_ncnn.dll
  set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(ncnn
    GIT_REPOSITORY https://github.com/Tencent/ncnn
    GIT_TAG ${NCNN_TAG}
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(ncnn)
endif()

add_library(sam2_ncnn SHARED
  src/sam2_ncnn.cpp
  include/sam2_ncnn.h
)

if(SAM2_VULKAN)
  find_package(Vulkan REQUIRED)
endif()

target_compile_definitions(sam2_ncnn PRIVATE
  SAM2_NCNN_BUILD=1
  SAM2_VULKAN=$<BOOL:${SAM2_VULKAN}>
  _CRT_SECURE_NO_WARNINGS
  UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(sam2_ncnn PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(sam2_ncnn PRIVATE ncnn)
if(SAM2_VULKAN)
  target_link_libraries(sam2_ncnn PRIVATE Vulkan::Vulkan)
endif()
if(MSVC)
  target_compile_options(sam2_ncnn PRIVATE /W4 /permissive- /utf-8 /EHsc /MP)
  if(NOT SAM2_VULKAN)
    # the CI-cached libncnn is Vulkan-enabled; delay-load the Vulkan loader so
    # a CPU-only DLL still has no runtime dependency on vulkan-1.dll. Local
    # CPU-only ncnn builds have no vulkan imports and skip this.
    find_library(VULKAN_LOADER_LIB NAMES vulkan-1 libvulkan
      HINTS "$ENV{Vulkan_SDK}/Lib" "$ENV{VULKAN_SDK}/Lib")
    if(VULKAN_LOADER_LIB)
      target_link_libraries(sam2_ncnn PRIVATE "${VULKAN_LOADER_LIB}" delayimp)
      target_link_options(sam2_ncnn PRIVATE /DELAYLOAD:vulkan-1.dll)
    endif()
  endif()
endif()

set_target_properties(sam2_ncnn PROPERTIES
  OUTPUT_NAME "sam2_ncnn"
  PREFIX ""
)

# The ncnn runtime is statically linked - the only artifacts are
# sam2_ncnn.dll plus the model files (sam2_*_fixed.ncnn.param/.bin and
# sam2_prompt.bin) resolved at runtime via sam2_encoder_create /
# sam2_decoder_create paths.

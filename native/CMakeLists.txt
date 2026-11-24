cmake_minimum_required(VERSION 3.20)
project(sam2_dml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static MSVC runtime would clash with onnxruntime.dll; keep /MD (the default
# for CMAKE_CXX_FLAGS_RELEASE here) explicitly.
if(MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(SAM2_ORT_EP "directml" CACHE STRING "Execution provider to build against: directml, cuda, tensorrt, cpu")
set_property(CACHE SAM2_ORT_EP PROPERTY STRINGS directml cuda tensorrt cpu)
string(TOLOWER "${SAM2_ORT_EP}" SAM2_ORT_EP)

# cuda and tensorrt ship as provider plugins inside the single Gpu.Windows package,
# so they share one extracted directory; directml and cpu have their own packages.
if(SAM2_ORT_EP STREQUAL "cuda" OR SAM2_ORT_EP STREQUAL "tensorrt")
  set(_ort_pkg "gpu")
else()
  set(_ort_pkg "${SAM2_ORT_EP}")
endif()
set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-${_ort_pkg}" CACHE PATH "Extracted ONNX Runtime package for SAM2_ORT_EP=${SAM2_ORT_EP}")

if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ONNX Runtime headers not found under ${ORT_ROOT}/include - run third_party/fetch_ort.py ${SAM2_ORT_EP} first")
endif()
if(NOT EXISTS "${ORT_ROOT}/lib/onnxruntime.lib")
  message(FATAL_ERROR "onnxruntime.lib not found under ${ORT_ROOT}/lib")
endif()

add_library(sam2_dml SHARED
  src/sam2_dml.cpp
  include/sam2_dml.h
)

string(TOUPPER "${SAM2_ORT_EP}" SAM2_ORT_EP_UPPER)
target_compile_definitions(sam2_dml PRIVATE
  SAM2_DML_BUILD=1 SAM2_EP_${SAM2_ORT_EP_UPPER}=1 UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(sam2_dml PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${ORT_ROOT}/include"
)
target_link_libraries(sam2_dml PRIVATE "${ORT_ROOT}/lib/onnxruntime.lib")
if(SAM2_ORT_EP STREQUAL "directml")
  target_link_libraries(sam2_dml PRIVATE d3d12 dxgi dxguid)
endif()
if(MSVC)
  target_compile_options(sam2_dml PRIVATE /W4 /permissive- /utf-8 /EHsc /MP)
endif()

set_target_properties(sam2_dml PROPERTIES
  OUTPUT_NAME "sam2_dml"
  PREFIX ""
  SUFFIX ".dll"
)

# Copy the runtime DLLs next to the freshly built sam2_dml.dll so the loader
# resolves them without touching PATH. Only the ones the selected EP actually
# needs: the shared-provider bridge + that EP's plugin for cuda/tensorrt.
set(ORT_RUNTIME_DLLS "${ORT_ROOT}/bin/onnxruntime.dll")
if(SAM2_ORT_EP STREQUAL "directml")
  list(APPEND ORT_RUNTIME_DLLS
    "${ORT_ROOT}/bin/DirectML.dll"
    "${ORT_ROOT}/bin/onnxruntime_providers_shared.dll")
elseif(SAM2_ORT_EP STREQUAL "cuda" OR SAM2_ORT_EP STREQUAL "tensorrt")
  list(APPEND ORT_RUNTIME_DLLS
    "${ORT_ROOT}/bin/onnxruntime_providers_shared.dll"
    "${ORT_ROOT}/bin/onnxruntime_providers_${SAM2_ORT_EP}.dll")
endif()
foreach(dll ${ORT_RUNTIME_DLLS})
  if(EXISTS "${dll}")
    add_custom_command(TARGET sam2_dml POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:sam2_dml>"
      COMMENT "Copying ${dll}")
  else()
    message(WARNING "runtime DLL missing: ${dll}")
  endif()
endforeach()

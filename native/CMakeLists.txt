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

set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "Extracted ONNX Runtime DirectML package")

if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ONNX Runtime headers not found under ${ORT_ROOT}/include - run third_party/fetch_ort_dml.py first")
endif()
if(NOT EXISTS "${ORT_ROOT}/lib/onnxruntime.lib")
  message(FATAL_ERROR "onnxruntime.lib not found under ${ORT_ROOT}/lib")
endif()

add_library(sam2_dml SHARED
  src/sam2_dml.cpp
  include/sam2_dml.h
)

target_compile_definitions(sam2_dml PRIVATE SAM2_DML_BUILD=1 UNICODE _UNICODE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(sam2_dml PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${ORT_ROOT}/include"
)
target_link_libraries(sam2_dml PRIVATE "${ORT_ROOT}/lib/onnxruntime.lib")
target_link_libraries(sam2_dml PRIVATE d3d12 dxgi dxguid)
if(MSVC)
  target_compile_options(sam2_dml PRIVATE /W4 /permissive- /utf-8 /EHsc /MP)
endif()

set_target_properties(sam2_dml PROPERTIES
  OUTPUT_NAME "sam2_dml"
  PREFIX ""
  SUFFIX ".dll"
)

# Copy the runtime DLLs next to the freshly built sam2_dml.dll so the loader
# resolves them without touching PATH.
set(ORT_RUNTIME_DLLS
  "${ORT_ROOT}/bin/onnxruntime.dll"
  "${ORT_ROOT}/bin/DirectML.dll"
  "${ORT_ROOT}/bin/onnxruntime_providers_shared.dll"
)
foreach(dll ${ORT_RUNTIME_DLLS})
  if(EXISTS "${dll}")
    add_custom_command(TARGET sam2_dml POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:sam2_dml>"
      COMMENT "Copying ${dll}")
  endif()
endforeach()

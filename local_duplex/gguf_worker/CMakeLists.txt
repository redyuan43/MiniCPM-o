cmake_minimum_required(VERSION 3.21)
project(local_duplex_gguf_worker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)

if(NOT DEFINED LLAMA_CPP_OMNI_DIR)
    set(LLAMA_CPP_OMNI_DIR "${CMAKE_CURRENT_LIST_DIR}/../../third_party/llama.cpp-omni")
endif()

if(NOT EXISTS "${LLAMA_CPP_OMNI_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "LLAMA_CPP_OMNI_DIR is invalid: ${LLAMA_CPP_OMNI_DIR}")
endif()

set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)

add_subdirectory("${LLAMA_CPP_OMNI_DIR}" "${CMAKE_BINARY_DIR}/llama_cpp_omni")
add_subdirectory("${LLAMA_CPP_OMNI_DIR}/tools/omni" "${CMAKE_BINARY_DIR}/llama_cpp_omni_tools_omni")

add_executable(local-duplex-gguf-worker worker_main.cpp)
set_target_properties(local-duplex-gguf-worker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
target_include_directories(local-duplex-gguf-worker PRIVATE
    "${LLAMA_CPP_OMNI_DIR}"
    "${LLAMA_CPP_OMNI_DIR}/common"
    "${LLAMA_CPP_OMNI_DIR}/tools/omni"
    "${LLAMA_CPP_OMNI_DIR}/vendor"
)
target_link_libraries(local-duplex-gguf-worker PRIVATE omni common Threads::Threads)

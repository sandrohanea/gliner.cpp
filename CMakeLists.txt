cmake_minimum_required(VERSION 3.20)
project(gliner_cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()
option(GLINER_FETCH_GGML "Download ggml if no local or installed copy is found" ON)
option(GLINER_FETCH_UTF8PROC "Download utf8proc if no installed copy is found" ON)
set(GLINER_GGML_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml" CACHE PATH "Local GGML checkout")
option(GGML_CUDA "Build gliner for the GGML CUDA backend" OFF)
option(GGML_METAL "Build gliner for the GGML Metal backend" ${APPLE})
option(GLINER_METAL_STRICT_F32 "Use the F32 GGML Metal dispatch fix; disable for faster, lower-precision matrix products" ON)
set(GLINER_BUILD_CUDA "${GGML_CUDA}")
set(GLINER_BUILD_METAL "${GGML_METAL}")
set(GLINER_COMPILED_BACKEND "cpu")
set(GLINER_METAL_PRECISION_MODE "not_applicable")

if(GLINER_BUILD_CUDA AND GLINER_BUILD_METAL)
    message(FATAL_ERROR "Select only one GPU backend: GGML_CUDA or GGML_METAL")
elseif(GLINER_BUILD_METAL AND NOT APPLE)
    message(FATAL_ERROR "GGML_METAL requires an Apple platform; use GGML_CUDA for NVIDIA GPUs")
elseif(GLINER_BUILD_CUDA)
    set(GLINER_COMPILED_BACKEND "cuda")
elseif(GLINER_BUILD_METAL)
    set(GLINER_COMPILED_BACKEND "metal")
endif()
message(STATUS "gliner execution backend: ${GLINER_COMPILED_BACKEND} (build-time selection)")

if(EXISTS "${GLINER_GGML_SOURCE_DIR}/CMakeLists.txt")
    add_subdirectory("${GLINER_GGML_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/ggml")
    set(GLINER_GGML_TARGET ggml ggml-base)
else()
    find_package(ggml CONFIG QUIET)
    if(TARGET ggml::ggml)
        if((GLINER_BUILD_CUDA AND NOT GGML_CUDA) OR (GLINER_BUILD_METAL AND NOT GGML_METAL))
            message(FATAL_ERROR "Installed GGML does not include the requested ${GLINER_COMPILED_BACKEND} backend; rebuild GGML or use GLINER_GGML_SOURCE_DIR")
        endif()
        set(GLINER_GGML_TARGET ggml::ggml)
    else()
        find_path(GGML_INCLUDE_DIR ggml.h)
        find_library(GGML_BASE_LIBRARY ggml-base)
        find_library(GGML_LIBRARY ggml)
        if(GGML_INCLUDE_DIR AND GGML_BASE_LIBRARY AND GGML_LIBRARY)
            if(GLINER_BUILD_CUDA OR GLINER_BUILD_METAL)
                message(FATAL_ERROR "GPU builds require a GGML CMake package or source checkout, not unconfigured GGML libraries")
            endif()
            add_library(gliner_ggml INTERFACE)
            target_include_directories(gliner_ggml INTERFACE "${GGML_INCLUDE_DIR}")
            target_link_libraries(gliner_ggml INTERFACE "${GGML_LIBRARY}" "${GGML_BASE_LIBRARY}")
            set(GLINER_GGML_TARGET gliner_ggml)
        elseif(GLINER_FETCH_GGML)
            include(FetchContent)
            FetchContent_Declare(ggml
                GIT_REPOSITORY https://github.com/ggml-org/ggml.git
                GIT_TAG v0.24.0
                GIT_SHALLOW TRUE)
            FetchContent_MakeAvailable(ggml)
            set(GLINER_GGML_TARGET ggml ggml-base)
        else()
            message(FATAL_ERROR "ggml not found. Install ggml, populate ggml/, or enable GLINER_FETCH_GGML")
        endif()
    endif()
endif()

if(GLINER_BUILD_CUDA)
    include(cmake/GgmlCudaPrecision.cmake)
    gliner_fix_cuda_precision()
elseif(GLINER_BUILD_METAL)
    if(GLINER_METAL_STRICT_F32)
        include(cmake/GgmlMetalPrecision.cmake)
        gliner_fix_metal_precision()
        set(GLINER_METAL_PRECISION_MODE "strict_f32")
    else()
        set(GLINER_METAL_PRECISION_MODE "fast")
        message(STATUS "gliner: fast Metal matrix products enabled; strict F32 encoder parity is not expected")
    endif()
endif()

find_path(UTF8PROC_INCLUDE_DIR utf8proc.h)
find_library(UTF8PROC_LIBRARY NAMES utf8proc_static utf8proc)
if(UTF8PROC_INCLUDE_DIR AND UTF8PROC_LIBRARY)
    add_library(gliner_unicode INTERFACE)
    target_include_directories(gliner_unicode INTERFACE "${UTF8PROC_INCLUDE_DIR}")
    target_link_libraries(gliner_unicode INTERFACE "${UTF8PROC_LIBRARY}")
    if(WIN32 AND UTF8PROC_LIBRARY MATCHES "utf8proc_static")
        target_compile_definitions(gliner_unicode INTERFACE UTF8PROC_STATIC)
    endif()
elseif(GLINER_FETCH_UTF8PROC)
    include(FetchContent)
    set(UTF8PROC_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(utf8proc
        GIT_REPOSITORY https://github.com/JuliaStrings/utf8proc.git
        GIT_TAG a1b99daa2a3393884220264c927a48ba1251a9c6)
    FetchContent_MakeAvailable(utf8proc)
    add_library(gliner_unicode ALIAS utf8proc)
else()
    message(FATAL_ERROR "utf8proc not found. Install utf8proc or enable GLINER_FETCH_UTF8PROC")
endif()

add_library(gliner STATIC
    src/gguf.cpp
    src/classify.cpp
    src/tokenizer.cpp
    src/deberta.cpp)
target_include_directories(gliner PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(gliner PUBLIC ${GLINER_GGML_TARGET} PRIVATE gliner_unicode)
target_compile_definitions(gliner PRIVATE GLINER_COMPILED_BACKEND="${GLINER_COMPILED_BACKEND}")
if(GLINER_BUILD_METAL AND NOT GLINER_METAL_STRICT_F32)
    target_compile_definitions(gliner PRIVATE GLINER_METAL_FAST_MATH=1)
endif()
if(MSVC)
    target_compile_options(gliner PRIVATE /W4 /utf-8)
else()
    target_compile_options(gliner PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(gliner-classify examples/classify.cpp examples/main.cpp)
target_link_libraries(gliner-classify PRIVATE gliner)
add_executable(gliner-bench examples/bench.cpp examples/main.cpp)
target_link_libraries(gliner-bench PRIVATE gliner)
target_compile_definitions(gliner-bench PRIVATE
    GLINER_BENCH_BUILD_TYPE="$<CONFIG>"
    GLINER_BENCH_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
    GLINER_BENCH_METAL_PRECISION_MODE="${GLINER_METAL_PRECISION_MODE}")
if(MINGW)
    target_link_options(gliner-classify PRIVATE -municode)
    target_link_options(gliner-bench PRIVATE -municode)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(gliner-test tests/test_core.c)
    target_link_libraries(gliner-test PRIVATE gliner)
    set_target_properties(gliner-test PROPERTIES LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED YES C_EXTENSIONS OFF)
    add_test(NAME gliner-core COMMAND gliner-test)
    set(GLINER_PARITY_TESTS gliner-core)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME gliner-python-benchmark-unit
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_python_benchmark.py")
        add_test(NAME gliner-cuda-precision-config
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cuda_precision_config.py"
                "${CMAKE_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/GgmlCudaPrecision.cmake")
        add_test(NAME gliner-metal-precision-config
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_metal_precision_config.py"
                "${CMAKE_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/GgmlMetalPrecision.cmake")
        set(GLINER_CONVERSION_OPTIONS "")
        set(GLINER_CONVERSION_TEST gliner-conversion)
        if(GLINER_BUILD_METAL AND NOT GLINER_METAL_STRICT_F32)
            list(APPEND GLINER_CONVERSION_OPTIONS --fast-metal)
            set(GLINER_CONVERSION_TEST gliner-conversion-fast)
            set(GLINER_CONVERSION_LABELS "format;metal")
        else()
            set(GLINER_CONVERSION_LABELS "parity;${GLINER_COMPILED_BACKEND}")
        endif()
        add_test(NAME ${GLINER_CONVERSION_TEST}
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_conversion.py"
                    "${CMAKE_CURRENT_SOURCE_DIR}/convert/convert_hf_to_gguf.py"
                    "$<TARGET_FILE:gliner-classify>"
                    "$<TARGET_FILE:gliner-test>" --expect-backend "${GLINER_COMPILED_BACKEND}"
                    ${GLINER_CONVERSION_OPTIONS})
        set_tests_properties(${GLINER_CONVERSION_TEST} PROPERTIES LABELS "${GLINER_CONVERSION_LABELS}" TIMEOUT 600)
        list(APPEND GLINER_PARITY_TESTS ${GLINER_CONVERSION_TEST})
        add_test(NAME gliner-benchmark
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_benchmark.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/convert/convert_hf_to_gguf.py"
                "$<TARGET_FILE:gliner-bench>" "$<TARGET_FILE:gliner-classify>"
                --expect-backend "${GLINER_COMPILED_BACKEND}")
        set_tests_properties(gliner-benchmark PROPERTIES LABELS "benchmark;${GLINER_COMPILED_BACKEND}" TIMEOUT 600)
        list(APPEND GLINER_PARITY_TESTS gliner-benchmark)
        set(GLINER_PARITY_CHECKPOINT "" CACHE PATH "Optional real checkpoint for live Python parity")
        set(GLINER_PARITY_GGUF "" CACHE FILEPATH "Converted GGUF for live Python parity")
        if(GLINER_PARITY_CHECKPOINT AND GLINER_PARITY_GGUF AND
                GLINER_BUILD_METAL AND NOT GLINER_METAL_STRICT_F32)
            message(STATUS "gliner: strict real-checkpoint parity is not registered for fast Metal mode")
        endif()
        if(GLINER_PARITY_CHECKPOINT AND GLINER_PARITY_GGUF AND
                (NOT GLINER_BUILD_METAL OR GLINER_METAL_STRICT_F32))
            add_test(NAME gliner-real-parity
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_parity.py"
                    --checkpoint "${GLINER_PARITY_CHECKPOINT}" --gguf "${GLINER_PARITY_GGUF}"
                    --binary "$<TARGET_FILE:gliner-classify>" --expect-backend "${GLINER_COMPILED_BACKEND}")
            set_tests_properties(gliner-real-parity PROPERTIES LABELS "parity;model;${GLINER_COMPILED_BACKEND}" TIMEOUT 600)
            list(APPEND GLINER_PARITY_TESTS gliner-real-parity)
        endif()
    endif()
    if(GLINER_COMPILED_BACKEND STREQUAL "cuda")
        # F32 accumulation alone does not disable the pinned GGML cuBLAS TF32 math mode.
        set_tests_properties(${GLINER_PARITY_TESTS} PROPERTIES
            ENVIRONMENT "NVIDIA_TF32_OVERRIDE=0;GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32")
    endif()
endif()

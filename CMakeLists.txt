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

if(EXISTS "${GLINER_GGML_SOURCE_DIR}/CMakeLists.txt")
    add_subdirectory("${GLINER_GGML_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/ggml")
    set(GLINER_GGML_TARGET ggml ggml-base)
else()
    find_package(ggml CONFIG QUIET)
    if(TARGET ggml::ggml)
        set(GLINER_GGML_TARGET ggml::ggml)
    else()
        find_path(GGML_INCLUDE_DIR ggml.h)
        find_library(GGML_BASE_LIBRARY ggml-base)
        find_library(GGML_LIBRARY ggml)
        if(GGML_INCLUDE_DIR AND GGML_BASE_LIBRARY AND GGML_LIBRARY)
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
if(MSVC)
    target_compile_options(gliner PRIVATE /W4 /utf-8)
else()
    target_compile_options(gliner PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(gliner-classify examples/classify.cpp)
target_link_libraries(gliner-classify PRIVATE gliner)
if(MINGW)
    target_link_options(gliner-classify PRIVATE -municode)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(gliner-test tests/test_core.c)
    target_link_libraries(gliner-test PRIVATE gliner)
    set_target_properties(gliner-test PROPERTIES LINKER_LANGUAGE CXX)
    add_test(NAME gliner-core COMMAND gliner-test)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME gliner-conversion
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_conversion.py"
                    "${CMAKE_CURRENT_SOURCE_DIR}/convert/convert_hf_to_gguf.py"
                    "$<TARGET_FILE:gliner-classify>"
                    "$<TARGET_FILE:gliner-test>")
        set(GLINER_PARITY_CHECKPOINT "" CACHE PATH "Optional real checkpoint for live Python parity")
        set(GLINER_PARITY_GGUF "" CACHE FILEPATH "Converted GGUF for live Python parity")
        if(GLINER_PARITY_CHECKPOINT AND GLINER_PARITY_GGUF)
            add_test(NAME gliner-real-parity
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_parity.py"
                    --checkpoint "${GLINER_PARITY_CHECKPOINT}" --gguf "${GLINER_PARITY_GGUF}"
                    --binary "$<TARGET_FILE:gliner-classify>")
            set_tests_properties(gliner-real-parity PROPERTIES LABELS "parity;model" TIMEOUT 600)
        endif()
    endif()
endif()

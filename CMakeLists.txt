cmake_minimum_required(VERSION 3.20)
project(gliner_cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(GLINER_FETCH_GGML "Download ggml if no local or installed copy is found" ON)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
    add_subdirectory(ggml)
    set(GLINER_GGML_TARGET ggml ggml-base)
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

add_library(gliner STATIC
    src/gguf.cpp
    src/classify.cpp
    src/tokenizer.cpp
    src/deberta.cpp)
target_include_directories(gliner PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(gliner PUBLIC ${GLINER_GGML_TARGET})
target_compile_options(gliner PRIVATE -Wall -Wextra -Wpedantic)

add_executable(gliner-classify examples/classify.cpp)
target_link_libraries(gliner-classify PRIVATE gliner)

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
    endif()
endif()

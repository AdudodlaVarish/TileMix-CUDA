cmake_minimum_required(VERSION 3.24)

# Build both the research target and local development target by default.
# Set this before project() so CUDA compiler detection uses the same targets.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 89 CACHE STRING "CUDA architectures")
endif()
project(tilemix_cuda LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(tilemix_fp16 STATIC
    src/matmul.cu
    src/softmax.cu
    src/pv.cu
    src/attention_fused_fp16.cu)
target_include_directories(tilemix_fp16 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(tilemix_fp16 PUBLIC cxx_std_17 cuda_std_17)
target_compile_options(tilemix_fp16 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(attention_smoke src/main.cu)
target_link_libraries(attention_smoke PRIVATE tilemix_fp16)

include(CTest)
if(BUILD_TESTING)
    foreach(name IN ITEMS softmax attention fused_attention)
        add_executable(${name}_test src/tests/${name}_test.cu)
        target_link_libraries(${name}_test PRIVATE tilemix_fp16)
        add_test(NAME ${name}_cpu COMMAND ${name}_test --cpu-only)
        set_tests_properties(${name}_cpu PROPERTIES LABELS cpu TIMEOUT 60)
        add_test(NAME ${name}_gpu COMMAND ${name}_test)
        set_tests_properties(${name}_gpu PROPERTIES
            LABELS gpu RESOURCE_LOCK cuda_device TIMEOUT 120)
    endforeach()
    add_test(NAME attention_smoke_gpu COMMAND attention_smoke)
    set_tests_properties(attention_smoke_gpu PROPERTIES
        LABELS gpu RESOURCE_LOCK cuda_device TIMEOUT 120)

    # Prefer the project oracle environment; an explicit interpreter overrides it.
    # Configuring never creates an environment or installs packages.
    if(NOT DEFINED Python3_EXECUTABLE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
        set(Python3_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
    endif()
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        execute_process(COMMAND "${Python3_EXECUTABLE}" -c
            "import importlib.util, sys; sys.exit(importlib.util.find_spec('torch') is None)"
            RESULT_VARIABLE torch_missing OUTPUT_QUIET ERROR_QUIET)
        if(torch_missing EQUAL 0)
            add_test(NAME torch_reference_cpu
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/reference_attention.py")
            set_tests_properties(torch_reference_cpu PROPERTIES LABELS "cpu;torch" TIMEOUT 60)
            add_test(NAME torch_reference_gpu
                COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/reference_attention.py"
                    --cuda-test $<TARGET_FILE:fused_attention_test>)
            set_tests_properties(torch_reference_gpu PROPERTIES
                LABELS "gpu;torch" RESOURCE_LOCK cuda_device TIMEOUT 120)
        else()
            message(STATUS "PyTorch not found in ${Python3_EXECUTABLE}; optional oracle tests omitted")
        endif()
    endif()

    find_program(COMPUTE_SANITIZER_EXECUTABLE compute-sanitizer)
    if(COMPUTE_SANITIZER_EXECUTABLE)
        foreach(check IN ITEMS memcheck racecheck synccheck)
            foreach(name IN ITEMS softmax attention fused_attention)
                add_test(NAME ${name}_${check}
                    COMMAND "${COMPUTE_SANITIZER_EXECUTABLE}" --tool ${check}
                        --error-exitcode 1 $<TARGET_FILE:${name}_test>)
                set_tests_properties(${name}_${check} PROPERTIES
                    LABELS "gpu;sanitizer;${check}" RESOURCE_LOCK cuda_device TIMEOUT 180)
            endforeach()
        endforeach()
    endif()
endif()

cmake_minimum_required(VERSION 3.18)

project(monte_carlo_option_pricing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(monte_carlo_option_pricing
    src/main.cpp
    src/monte_carlo_cpu.cpp
    src/monte_carlo_cuda.cu
)

target_include_directories(monte_carlo_option_pricing
    PRIVATE
        include
)

target_link_libraries(monte_carlo_option_pricing
    PRIVATE
        CUDA::curand
)

set_target_properties(monte_carlo_option_pricing PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

if(MSVC)
    target_compile_options(monte_carlo_option_pricing
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(monte_carlo_option_pricing
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-pedantic>
    )
endif()

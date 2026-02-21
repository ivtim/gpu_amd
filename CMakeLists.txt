cmake_minimum_required (VERSION 3.10)
project ("GpuAmd" VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_PREFIX_PATH "C:/Program Files/AMD/ROCm/6.4/lib/cmake" CACHE PATH "Prefix path for find_package")
find_package(hip REQUIRED)

add_executable (GpuAmd "main.cpp")
target_link_libraries(GpuAmd PRIVATE hip::device)
target_compile_features(GpuAmd PRIVATE cxx_std_11)

set(TARGET_NAME "GpuAmd")
set_target_properties(${TARGET_NAME} PROPERTIES
    OUTPUT_NAME "${TARGET_NAME}_${PROJECT_VERSION}"
)
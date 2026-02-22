cmake_minimum_required (VERSION 3.10)
project ("GpuAmd" VERSION 1.0.0 LANGUAGES CXX HIP)

set(CMAKE_PREFIX_PATH "C:/Program Files/AMD/ROCm/6.4/lib/cmake" CACHE PATH "Prefix path for find_package")
find_package(hip REQUIRED)
find_package(rocprim REQUIRED)
find_package(rocthrust REQUIRED)

add_subdirectory(common)
add_subdirectory(sha1)
add_subdirectory(thrust)
cmake_minimum_required (VERSION 3.10)
project ("GpuAmd" VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_PREFIX_PATH "C:/Program Files/AMD/ROCm/6.4/lib/cmake" CACHE PATH "Prefix path for find_package")
find_package(hip REQUIRED)

add_subdirectory(common)
add_subdirectory(sha1)
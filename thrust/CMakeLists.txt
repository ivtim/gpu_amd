set_source_files_properties(main.hip PROPERTIES LANGUAGE CXX)
add_executable (GpuAmdThrust "main.hip")
target_link_libraries(GpuAmdThrust
    PRIVATE
        hip::device
        roc::rocthrust
)
target_compile_features(GpuAmdThrust PRIVATE cxx_std_17)

target_include_directories(GpuAmdThrust PRIVATE
    "C:/Program Files/AMD/ROCm/6.4/include"
)

target_compile_definitions(GpuAmdThrust PRIVATE __HIPCC__=1)

target_compile_definitions(GpuAmdThrust PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
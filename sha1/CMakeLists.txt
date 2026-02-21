add_executable (GpuAmdSha1 "main.cpp")
target_link_libraries(GpuAmdSha1 PRIVATE hip::device)
target_compile_features(GpuAmdSha1 PRIVATE cxx_std_11)

set(TARGET_NAME "GpuAmdSha1")
set_target_properties(${TARGET_NAME} PROPERTIES
    OUTPUT_NAME "${TARGET_NAME}_${PROJECT_VERSION}"
)
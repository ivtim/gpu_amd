add_executable (GpuAmdCommon "main.cpp")
target_link_libraries(GpuAmdCommon PRIVATE hip::device)
target_compile_features(GpuAmdCommon PRIVATE cxx_std_11)

set(TARGET_NAME "GpuAmdCommon")
set_target_properties(${TARGET_NAME} PROPERTIES
    OUTPUT_NAME "${TARGET_NAME}_${PROJECT_VERSION}"
)
add_library(MLToolingC SHARED
    "src/compute/tensor.cpp"
)

target_link_libraries(MLToolingC PRIVATE MLTooling)

target_include_directories(MLToolingC
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(MLToolingC PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 23
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

if (WIN32)
    target_compile_definitions(MLToolingC PRIVATE MLTOOLING_C_EXPORTS)
endif()

add_executable(ctest
    "test/compute/tensorTest.c"
)

target_link_libraries(ctest PRIVATE MLToolingC)
target_include_directories(ctest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

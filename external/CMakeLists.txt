set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BUILD_WITHOUT_LAPACK ON CACHE BOOL "" FORCE)
set(BUILD_WITHOUT_LAPACKE ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

set(CMAKE_MESSAGE_LOG_LEVEL ERROR)

add_subdirectory(
    OpenBLAS
    ${CMAKE_BINARY_DIR}/OpenBLAS
)

set(CMAKE_MESSAGE_LOG_LEVEL STATUS)

add_library(MLTooling_dependencies INTERFACE)

target_link_libraries(MLTooling_dependencies
    INTERFACE
        openblas
)

target_include_directories(MLTooling_dependencies
    SYSTEM INTERFACE
        ${CMAKE_BINARY_DIR}/generated
        ${CMAKE_BINARY_DIR}
)

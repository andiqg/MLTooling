cmake_minimum_required(VERSION 3.28)
project(MLTooling VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)


option(MLTOOLING_IDE_FILES "Add headers and source groups for IDEs" OFF)

# Dependencies
add_subdirectory(external)

set(MLTOOLING_SOURCES
    "src/api/compute/Tensor.cpp"
    "src/api/math/Matrix.cpp"
    "src/api/math/Vector.cpp"
    "src/api/models/LinearRegression.cpp"
    "src/data/DataColumn.cpp"
    "test/data/DataFrame.cpp"
    "src/math/matrix.cpp"
    "src/math/kernels/matrixkernels.cpp"
    "src/compute/core/Storage.cpp"
    "src/compute/core/MltArray.cpp"
    "src/compute/Operations.cpp"
    "src/compute/Runtime.cpp"
    "src/compute/Kernels.cpp"
)

add_library(MLTooling ${MLTOOLING_SOURCES})

target_sources(MLTooling
    PUBLIC
        FILE_SET CXX_MODULES FILES
            "modules/Error.cppm"
            "modules/Storage.cppm"
            "modules/Tensor.cppm"
            "modules/SizeArray.cppm"
            "modules/Operations.cppm"
            "modules/MltArray.cppm"
            "modules/Runtime.cppm"
            "modules/Kernels.cppm"
)

if(MLTOOLING_IDE_FILES)
    file(GLOB_RECURSE MLTOOLING_HEADERS CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/include/*.hpp"
        "${PROJECT_SOURCE_DIR}/include/*.h"
    )

    target_sources(MLTooling
        PRIVATE
            ${MLTOOLING_HEADERS}
    )

    source_group(TREE ${PROJECT_SOURCE_DIR}
        FILES
            ${MLTOOLING_SOURCES}
            ${MLTOOLING_HEADERS}
    )

endif()

target_include_directories(MLTooling
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include/public
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include/internal
)

if(NOT MSVC)
    target_compile_options(MLTooling PRIVATE -fPIC)
endif()

if(MSVC)
    target_compile_options(MLTooling PRIVATE
        "$<$<CONFIG:Release>:/O2>"
        "$<$<CONFIG:Release>:/fp:fast>"
    )
else()
    target_compile_options(MLTooling PRIVATE
        "$<$<CONFIG:Release>:-march=native>"
        "$<$<CONFIG:Release>:-ffast-math>"
    )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(MLTooling PRIVATE
        "$<$<CONFIG:Release>:-fopt-info-vec>"
        "$<$<CONFIG:Release>:-fopt-info-vec-missed>"
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(MLTooling PRIVATE
        "$<$<CONFIG:Release>:-Rpass=loop-vectorize>"
        "$<$<CONFIG:Release>:-Rpass-missed=loop-vectorize>"
    )
elseif(MSVC)
    target_compile_options(MLTooling PRIVATE
        "$<$<CONFIG:Release>:/Qvec-report:2>"
    )
endif()

target_link_libraries(MLTooling PRIVATE MLTooling_dependencies)

install(TARGETS MLTooling DESTINATION lib)
install(DIRECTORY include/public DESTINATION include)

add_executable(example_basic examples/example_basic.cpp
)
target_link_libraries(example_basic PRIVATE MLTooling)

message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler ID : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Build Type  : ${CMAKE_BUILD_TYPE}")

add_library(MLTooling_internal INTERFACE)
target_include_directories(MLTooling_internal INTERFACE
    ${PROJECT_SOURCE_DIR}/include/internal
)

add_executable(tests
    "test/test.cpp"
    "test/data/DataFrameTest.cpp"
    "test/math/datastructures/MatrixFloatTest.cpp"
    "test/math/datastructures/MatrixTest.cpp"
    "test/math/datastructures/VectorTest.cpp"
    "test/math/kernels/matrixkernelTest.cpp"
    "test/compute/StorageTest.cpp"
    #"test/compute/backend/cpu/NativeTest.cpp"
    "test/compute/MltArrayTest.cpp"
    "test/compute/operations/OperationsTest.cpp"
)

target_link_libraries(tests PRIVATE MLTooling MLTooling_internal)

add_subdirectory(bindings/c)

#install(TARGETS MLToolingCapi DESTINATION lib)

## Kopieren ins Python-Paket-Verzeichnis nach dem Build
#add_custom_command(TARGET MLTooling_capi POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E make_directory
#        ${CMAKE_CURRENT_SOURCE_DIR}/pymltooling/_lib
#        COMMAND ${CMAKE_COMMAND} -E copy
#        $<TARGET_FILE:MLTooling_capi>
#        ${CMAKE_CURRENT_SOURCE_DIR}/pymltooling/_lib
#)

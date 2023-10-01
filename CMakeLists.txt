cmake_minimum_required(VERSION 3.25)
project(PBTorch)

set(CMAKE_CXX_STANDARD 20)

set(DEP_ROOT ${CMAKE_CURRENT_LIST_DIR}/libs)

set(Torch_ROOT ${DEP_ROOT}/libtorch)

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(PBTorch library.cpp)

set_property(TARGET PBTorch PROPERTY CXX_STANDARD 20)

message(STATUS "PYTHON_LIBRARIES: ${Python_LIBRARIES}")

target_include_directories(PBTorch PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_definitions(PBTorch PRIVATE ${TORCH_DEFINITIONS})
target_link_libraries(PBTorch ${TORCH_LIBRARIES} Python::Python)

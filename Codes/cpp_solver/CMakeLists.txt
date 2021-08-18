cmake_minimum_required(VERSION 3.1)
project(cppsolver)

# add_subdirectory(pybind11)
find_package(PkgConfig REQUIRED)
# LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Find_Package(G2O REQUIRED)
Find_Package(Eigen3 REQUIRED)
Find_Package(pybind11 REQUIRED)
Find_Package(Ceres REQUIRED COMPONENTS SuiteSparse EigenSparse Multithreading LAPACK)
include_directories(${PROJECT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})


pybind11_add_module(cppsolver src/cpp_solver_analytical.cpp)
target_link_libraries(cppsolver PRIVATE ${CERES_LIBRARIES})

# pybind11_add_module(cppsolver src/cpp_solver.cpp)
# target_link_libraries(cppsolver PRIVATE ${CERES_LIBRARIES})
# pybind11_LIBRARIES(cmake_example, ${CERES_LIBRARIES})

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(cppsolver PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})


add_executable(test_eigen src/test_eigen.cpp)
target_link_libraries(cppsolver PRIVATE ${CERES_LIBRARIES})


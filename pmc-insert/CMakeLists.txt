cmake_minimum_required(VERSION 3.15)
project(pmc_c_strict_plus CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_executable(api_detector tool.cpp)
target_link_libraries(api_detector PRIVATE
  clangTooling clangBasic clangASTMatchers clangRewrite clangFrontend
  clangSerialization clangAST clangLex LLVM
)

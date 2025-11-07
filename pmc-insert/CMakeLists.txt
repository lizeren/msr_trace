cmake_minimum_required(VERSION 3.15)
project(pmc_insert CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Always generate compile_commands.json for IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use LLVM 10
set(LLVM_DIR "/usr/lib/llvm-10/lib/cmake/llvm")
set(Clang_DIR "/usr/lib/llvm-10/lib/cmake/clang")

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_executable(api_detector tool_simple.cpp)
target_link_libraries(api_detector PRIVATE
  clangTooling clangBasic clangASTMatchers clangRewrite clangFrontend
  clangSerialization clangAST clangLex LLVM
)

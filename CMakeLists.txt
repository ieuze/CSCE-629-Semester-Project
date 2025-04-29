cmake_minimum_required(VERSION 3.15)
project(cppheight CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # 推荐关闭 C++ 扩展

# --- 查找依赖项 ---

# 1. Eigen3 (通常是 header-only，但 find_package 可确保找到)
#    需要安装 eigen3-devel 或类似包
find_package(Eigen3 3.3 REQUIRED CONFIG) # 使用 CONFIG 模式更可靠

# 2. HiGHS (使用 pkg-config)
#    需要安装 highs-devel 或类似包，并包含 .pc 文件
find_package(PkgConfig REQUIRED)
pkg_check_modules(HIGHS REQUIRED highs>=1.1) # 检查 highs.pc 是否存在且版本足够

# --- 定义可执行文件 ---
add_executable(cppheight cppheight.cpp)

# --- 链接库 ---
target_link_libraries(cppheight PRIVATE
    Eigen3::Eigen       # Eigen3 的 CMake 目标
    ${HIGHS_LIBRARIES}  # pkg-config 找到的 HiGHS 库
)

# --- 包含目录 ---
# Eigen 通常通过 target_link_libraries 处理
# HiGHS 通过 pkg-config 的 HIGHS_INCLUDE_DIRS 处理
target_include_directories(cppheight PRIVATE
    ${HIGHS_INCLUDE_DIRS}
)

# --- 编译选项 ---
# 根据需要调整优化级别和架构特定标志
target_compile_options(cppheight PRIVATE -O3 -march=native)

# --- 安装规则 (可选) ---
# install(TARGETS cppheight DESTINATION bin)

# --- 提示信息 ---
message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIRS}")
message(STATUS "Found HiGHS includes: ${HIGHS_INCLUDE_DIRS}")
message(STATUS "Found HiGHS libraries: ${HIGHS_LIBRARIES}") 
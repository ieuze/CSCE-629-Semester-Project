cmake_minimum_required(VERSION 3.15)
project(lp_verifier CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -------- 1. pkg-config 寻找 HiGHS ----------
find_package(PkgConfig REQUIRED)
pkg_check_modules(HIGHS REQUIRED highs)      # <- 用 highs.pc

# -------- 2. 其他依赖 ----------
find_package(BLAS   REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP)                         # optional

# -------- 3. (lp_verifier 目标已被移除) ----------

# -------- 4. 添加 manual_verify 可执行文件 ----------
add_executable(manual_verify manual_verify.cpp)

# 头文件 (通常由链接库传递，但显式添加确保安全)
target_include_directories(manual_verify PRIVATE
    /usr/include/eigen3
    ${HIGHS_INCLUDE_DIRS}
)

# 库 (与 lp_verifier 相同)
target_link_libraries(manual_verify PRIVATE
    ${HIGHS_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# 编译选项 (与 lp_verifier 相同)
target_compile_options(manual_verify PRIVATE -O3 -march=native)
if(OpenMP_CXX_FOUND)
  target_link_libraries(manual_verify PRIVATE OpenMP::OpenMP_CXX)
  target_compile_options(manual_verify PRIVATE -fopenmp)
endif()

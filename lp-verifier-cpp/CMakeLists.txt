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

# -------- 3. 可执行文件 ----------
add_executable(lp_verifier compute_m_height.cpp)

# 头文件
target_include_directories(lp_verifier PRIVATE
    /usr/include/eigen3
    ${HIGHS_INCLUDE_DIRS}
)

# 库
target_link_libraries(lp_verifier PRIVATE
    ${HIGHS_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# 编译选项
target_compile_options(lp_verifier PRIVATE -O3 -march=native)
if(OpenMP_CXX_FOUND)
  target_link_libraries(lp_verifier PRIVATE OpenMP::OpenMP_CXX)
  target_compile_options(lp_verifier PRIVATE -fopenmp)
endif()

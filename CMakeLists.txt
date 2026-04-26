cmake_minimum_required(VERSION 3.20)
project(dfloat11_tt VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Locate TT_METAL_HOME
# ---------------------------------------------------------------------------
if(NOT DEFINED TT_METAL_HOME)
    if(DEFINED ENV{TT_METAL_HOME})
        set(TT_METAL_HOME $ENV{TT_METAL_HOME})
    else()
        set(TT_METAL_HOME "$ENV{HOME}/tt-metal")
    endif()
endif()
message(STATUS "TT_METAL_HOME = ${TT_METAL_HOME}")

set(TT_BUILD "${TT_METAL_HOME}/build_Release")

# ---------------------------------------------------------------------------
# Locate the Python interpreter inside tt-metal's venv (preferred) or system
# ---------------------------------------------------------------------------
if(NOT DEFINED Python_EXECUTABLE)
    set(Python_EXECUTABLE "${TT_METAL_HOME}/python_env/bin/python3")
endif()
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# ---------------------------------------------------------------------------
# Find tt-metalium and tt-nn
# ---------------------------------------------------------------------------
list(APPEND CMAKE_PREFIX_PATH
    "${TT_BUILD}"
    "${TT_BUILD}/lib/cmake/tt-metalium"
    "${TT_BUILD}/lib/cmake/tt-nn"
    # transitive deps that tt-metalium needs
    "${TT_BUILD}/lib/cmake/umd"
    "${TT_BUILD}/lib/cmake/spdlog"
    "${TT_BUILD}/lib/cmake/libuv"
    "${TT_BUILD}/lib/cmake/flatbuffers"
    "${TT_BUILD}/lib/cmake/CapnProto"
    "${TT_BUILD}/cmake"           # enchantum
    "${TT_BUILD}/_deps/spdlog-build"
    "${TT_BUILD}/_deps/nlohmann_json-build"
    "${TT_BUILD}/_deps/xtl-build"
    "${TT_BUILD}/_deps/xtensor-build"
    "${TT_BUILD}/_deps/enchantum-build"
    "${TT_BUILD}/share/Tracy"
)

# The root-level *-config.cmake files reference broken relative paths.
# Include the actual target files directly, bypassing find_package entirely.
# This is safe: we know exactly where tt-metal was built.

# --- Load transitive deps that Metalium.cmake needs ---
find_package(fmt          REQUIRED HINTS "${TT_BUILD}" NO_DEFAULT_PATH)
find_package(nlohmann_json REQUIRED HINTS "${TT_BUILD}" NO_DEFAULT_PATH)
find_package(umd          REQUIRED HINTS "${TT_BUILD}/lib/cmake/umd" NO_DEFAULT_PATH)
find_package(spdlog       REQUIRED HINTS "${TT_BUILD}/lib/cmake/spdlog" NO_DEFAULT_PATH)
find_package(Threads      REQUIRED)

# tt-logger
find_package(tt-logger    QUIET HINTS "${TT_BUILD}" "${TT_BUILD}/lib/cmake" NO_DEFAULT_PATH)

# --- Metalium targets ---
include("${TT_BUILD}/lib/cmake/tt-metalium/Metalium.cmake")

# --- xtensor (needed by TT-NN.cmake) ---
find_package(xtl     REQUIRED HINTS "${TT_BUILD}/_deps/xtl-build"     NO_DEFAULT_PATH)
find_package(xtensor REQUIRED HINTS "${TT_BUILD}/_deps/xtensor-build" NO_DEFAULT_PATH)

# --- TT-NN targets ---
include("${TT_BUILD}/lib/cmake/tt-nn/TT-NN.cmake")

# ---------------------------------------------------------------------------
# nanobind (bundled inside tt-metal's CPM cache)
# ---------------------------------------------------------------------------
set(NB_SRC "${TT_METAL_HOME}/.cpmcache/nanobind/8be24e8fbbe8c9cee0b8c553c21d090a5ede3d79")
list(APPEND CMAKE_PREFIX_PATH "${NB_SRC}/cmake")
find_package(nanobind CONFIG REQUIRED HINTS "${NB_SRC}/cmake")

# ---------------------------------------------------------------------------
# DFloat11-TT host library
# ---------------------------------------------------------------------------
add_library(df11_host STATIC
    host/df11_op.cpp
    op/dfloat11_decompress.cpp
)
set_target_properties(df11_host PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(df11_host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TT_BUILD}/include
    ${TT_BUILD}/include/tt-metalium
)
target_link_libraries(df11_host PUBLIC TT::Metalium TTNN::TTNN)

# Kernel source path baked in at compile time for JIT discovery
target_compile_definitions(df11_host PUBLIC
    DF11_KERNEL_PREFIX="${CMAKE_CURRENT_SOURCE_DIR}/kernels/"
)

# ---------------------------------------------------------------------------
# Python extension module
# ---------------------------------------------------------------------------
nanobind_add_module(dfloat11_tt_cpp
    op/pybind.cpp
)
target_link_libraries(dfloat11_tt_cpp PRIVATE df11_host)
set_target_properties(dfloat11_tt_cpp PROPERTIES
    BUILD_RPATH "${TT_BUILD}/lib;${TT_BUILD}/ttnn"
    INSTALL_RPATH "${TT_BUILD}/lib;${TT_BUILD}/ttnn"
)

# ---------------------------------------------------------------------------
# Install .so alongside the Python package
# ---------------------------------------------------------------------------
install(TARGETS dfloat11_tt_cpp
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}
)

message(STATUS "DFloat11-TT kernels: ${CMAKE_CURRENT_SOURCE_DIR}/kernels/")
message(STATUS "Kernels compiled on-the-fly by tt-metal JIT.")

cmake_minimum_required(VERSION 3.18)
project(GmxQeqReann LANGUAGES CXX)
set(GMX_QEQ_NVTX_LIBRARY "" CACHE FILEPATH "Optional NVTX shared library from a Python CUDA distribution")
set(GMX_QEQ_NVTX_INCLUDE_DIR "" CACHE PATH "Matching NVTX headers")
if(GMX_QEQ_NVTX_LIBRARY AND NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt SHARED IMPORTED)
    set_target_properties(CUDA::nvToolsExt PROPERTIES IMPORTED_LOCATION "${GMX_QEQ_NVTX_LIBRARY}")
    if(GMX_QEQ_NVTX_INCLUDE_DIR)
        set_target_properties(CUDA::nvToolsExt PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${GMX_QEQ_NVTX_INCLUDE_DIR}")
    endif()
endif()
if(GMX_QEQ_NVTX_LIBRARY)
    # Torch 2.4 also searches this variable separately from CUDA::nvToolsExt.
    set(LIBNVTOOLSEXT "${GMX_QEQ_NVTX_LIBRARY}" CACHE FILEPATH "NVTX for Torch" FORCE)
endif()
find_package(Torch REQUIRED)
# GMX_MIREANN BEGIN: build both backends in the same configured LibTorch library.
# ORIGINAL | add_library(gmx_qeq_reann SHARED qeq_reann_plugin.cpp)
add_library(gmx_qeq_reann SHARED qeq_reann_plugin.cpp mireann_backend.cpp)
# GMX_MIREANN_GPU_AUDIT: real-space charge potentials/reference kernels.
# No CUDA requirement is introduced for CPU builds. NVCC sees no Torch headers.
if(GMX_GPU STREQUAL "CUDA")
    enable_language(CUDA)
    target_sources(gmx_qeq_reann PRIVATE mireann_classical_cuda.cu)
    target_compile_definitions(gmx_qeq_reann PRIVATE GMX_MIREANN_CUDA_CLASSICAL=1)
    set_target_properties(gmx_qeq_reann PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()
# GMX_MIREANN END
target_compile_features(gmx_qeq_reann PRIVATE cxx_std_17)
target_include_directories(gmx_qeq_reann PRIVATE ../../src/gromacs/mdlib)
target_link_libraries(gmx_qeq_reann PRIVATE "${TORCH_LIBRARIES}")
# GMX_QEQ_REANN OPTIMIZATION: CPU-only LibTorch remains supported. CUDA graph
# APIs are compiled only when LibTorch exports its CUDA target.
if(TARGET torch_cuda)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(gmx_qeq_reann PRIVATE GMX_QEQ_TORCH_CUDA=1)
    target_link_libraries(gmx_qeq_reann PRIVATE CUDA::cudart)
endif()
set_property(TARGET gmx_qeq_reann PROPERTY CXX_STANDARD 17)
separate_arguments(TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(gmx_qeq_reann PRIVATE ${TORCH_FLAGS})
set_target_properties(gmx_qeq_reann PROPERTIES BUILD_RPATH_USE_ORIGIN ON)

# GMX_QEQ_REANN ABI FIX BEGIN (2026-10-22)
# The backend must resolve libtorch/libc10 from the selected Torch package.
# GNU ld defaults to DT_RUNPATH, which loses to LD_LIBRARY_PATH: a user's
# Conda/DeepMD path can then mix a CPU/new-ABI libc10 with CUDA/old-ABI Torch.
# DT_RPATH prioritizes this backend's configured Torch directory, including
# transitive dependencies. Keep the application's environment unchanged.
# This cannot replace a different Torch already loaded in the same process;
# the native GROMACS executable must remain free of another Torch backend.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(gmx_qeq_reann PRIVATE "LINKER:--disable-new-dtags")
    set_property(TARGET gmx_qeq_reann APPEND PROPERTY
        BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
endif()
# GMX_QEQ_REANN ABI FIX END

# GMX_QEQ_REANN: one top-level configure/build/install includes this backend.
# The runtime finds it beside libgromacs; the MDP never names this library.
if(GMX_QEQ_REANN)
    include(GNUInstallDirs)
    set_target_properties(gmx_qeq_reann PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib")
    install(TARGETS gmx_qeq_reann LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT libraries)
endif()

#!/bin/bash

module load ninja/1.9.0
module load cmake
module load gcc/10.2.1

cmake -G Ninja \
        -DCMAKE_BUILD_TYPE='Release' \
        -DLLVM_ENABLE_PROJECTS='clang;openmp' \
        -DCMAKE_INSTALL_PREFIX=./install \
        -DLLVM_OPTIMIZED_TABLEGEN='On' \
        -DBUILD_SHARED_LIBS='On' \
    	-DLLVM_USE_LINKER=gold \
        -DLLVM_ENABLE_ASSERTIONS='On' \
        ../llvm

#!/bin/bash

#module load clang/12.0.0
#export CC=$(which clang)
#export CXX=$(which clang++)

module load gcc/10.2.1
export CC=$(which gcc)
export CXX=$(which g++)

module load ninja/1.9.0

cmake -G Ninja \
        -DCMAKE_BUILD_TYPE='RelWithDebInfo' \
        -DLLVM_ENABLE_PROJECTS='clang;openmp' \
        -DCMAKE_INSTALL_PREFIX=./install \
        -DLLVM_OPTIMIZED_TABLEGEN='On' \
        -DBUILD_SHARED_LIBS='On' \
    	-DLLVM_USE_LINKER=gold \
        -DLLVM_ENABLE_ASSERTIONS='On' \
        ../llvm
        #-DLLVM_CCACHE_BUILD='On' \
	#-DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN='On'\
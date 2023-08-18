#!/bin/bash

set -eux

#-e     When this option is on, if a simple command fails for any of the reasons listed in Consequences of
#       Shell  Errors or returns an exit status value >0, and is not part of the compound list following a
#       while, until, or if keyword, and is not a part of an AND  or  OR  list,  and  is  not  a  pipeline
#       preceded by the ! reserved word, then the shell shall immediately exit.
#-u     The shell shall write a message to standard error when it tries to expand a variable that  is  not
#       set and immediately exit. An interactive shell shall not exit.
#-x     The  shell shall write to standard error a trace for each command after it expands the command and
#       before it executes it. It is unspecified whether the command that turns tracing off is traced.

LIBTORCH_VERSION=2.0.1

if [ -f "libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip" ]; then
    echo "Skipping libtorch download."
else
  wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip
fi
unzip -o libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}+cpu.zip

LIBTORCH=$(pwd)/libtorch/
LD_LIBRARY_PATH=$(pwd)/libtorch/lib/
ls $LD_LIBRARY_PATH

cargo fmt --all -- --check \
&& cargo clippy --all --all-features -- -D warnings \
&& cargo test --all --all-features && cargo build --release
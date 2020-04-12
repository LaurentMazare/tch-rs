#!/bin/sh
# The idea of using a wrapper to insert flags comes from the following thread.
# https://users.rust-lang.org/t/linking-with-local-c-library/39866/4
set -eu
FIRST_ARG=$1
shift
gcc "$FIRST_ARG" -Wl,--no-as-needed "$@"

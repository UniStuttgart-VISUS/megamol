#!/bin/bash
#
# Script preparing cmake settings
#

prefix=${1:-"/usr/"}

rm -rf build.release
mkdir build.release
cd build.release
cmake .. -DCMAKE_INSTALL_PREFIX=$prefix
cd ..

rm -rf build.debug
mkdir build.debug
cd build.debug
cmake .. -DCMAKE_INSTALL_PREFIX=$prefix -DCMAKE_BUILD_TYPE=Debug
cd ..


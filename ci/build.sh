#!/bin/bash

mkdir build
cd build
cmake ..
proc_count=`grep -c ^processor /proc/cpuinfo`
make -j $proc_count
make install
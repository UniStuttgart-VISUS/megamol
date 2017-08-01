#!/usr/bin/env bash

mkdir build
cd build
cmake ..
make -j
make install
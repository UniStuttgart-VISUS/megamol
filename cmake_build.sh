#!/bin/bash
#
# building all
#

prefix=${1:-"/usr/"}

./cmake_prep.sh $prefix
if [ $? -ne 0 ] ; then exit 1; fi
cd build.release
make
# make install
if [ $? -ne 0 ] ; then exit 1; fi
cd ..
cd build.debug
make
# make install
if [ $? -ne 0 ] ; then exit 1; fi
cd ..

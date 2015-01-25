#!/bin/bash
#
# building all
#

prefix=${1:-"/usr/"}

./prep_cmake.sh $prefix
if [ $? -ne 0 ] ; then exit 1; fi
cd build.release
make
if [ $? -ne 0 ] ; then exit 1; fi
cd ..
cd build.debug
make
if [ $? -ne 0 ] ; then exit 1; fi
cd ..

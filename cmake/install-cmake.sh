#!/bin/bash

if [ -z "$TRAVIS_BUILD_DIR" ]; then
  echo "Need to set TRAVIS_BUILD_DIR"
  exit 1
fi

CMAKE_FILE="cmake-3.12.4-Linux-x86_64"
CMAKE_URL="https://cmake.org/files/v3.12/$CMAKE_FILE.tar.gz"

# Download and install.
DEPS_DIR="$TRAVIS_BUILD_DIR/deps"
mkdir -p $DEPS_DIR/cmake-install && cd $DEPS_DIR
travis_retry wget --no-check-certificate $CMAKE_URL
tar xf $CMAKE_FILE.tar.gz 
echo $CMAKE_FILE installed to $DEPS_DIR/$CMAKE_FILE

# Add to path and go back
export PATH=$DEPS_DIR/$CMAKE_FILE:$DEPS_DIR/$CMAKE_FILE/bin:$PATH
cd $TRAVIS_BUILD_DIR

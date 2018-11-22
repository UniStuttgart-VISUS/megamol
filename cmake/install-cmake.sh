#!/bin/sh

if [ -z "$TRAVIS_BUILD_DIR" ]; then
    echo "Need to set TRAVIS_BUILD_DIR"
    exit 1
fi

CMAKE_FILE="cmake-3.12.4-Linux-x86_64"
CMAKE_URL="https://cmake.org/files/v3.12/$CMAKE_FILE.tar.gz"

# Download and install.
DEPS_DIR="${TRAVIS_BUILD_DIR}/deps"
mkdir ${DEPS_DIR} && cd ${DEPS_DIR}
travis_retry wget --no-check-certificate ${CMAKE_URL}
tar -xf ${CMAKE_FILE}.tar.gz -C cmake-install

# Add to path and go back
PATH=${DEPS_DIR}/cmake-install:${DEPS_DIR}/cmake-install/bin:$PATH
cd ${TRAVIS_BUILD_DIR}

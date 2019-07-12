#!/bin/bash
# vim: nospell
#
# Install the core CUDA toolkit for a ubuntu-xenial (16.04) system. Requires the
# CUDA environment variable to be set to the required version.
#
# Since this script updates environment variables, to execute correctly you must
# 'source' this script, rather than executing it in a sub-process.
#

travis_retry wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_${CUDA}_amd64.deb
travis_retry sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
travis_retry sudo dpkg -i cuda-repo-ubuntu1604_${CUDA}_amd64.deb
travis_retry sudo apt-get update -qq
export CUDA_VER=$(expr ${CUDA} : '\([0-9]*\.[0-9]*\)')
export CUDA_VER_MAJOR=${CUDA_VER%.*}
export CUDA_VER_MINOR=${CUDA_VER#*.}
export CUDA_APT=${CUDA_VER/./-}

export CUDA_HOME=/usr/local/cuda-${CUDA_VER}
export LD_LIBRARY_PATH=${CUDA_HOME}/nvvm/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:${PATH}

# The cuda-core package is deprecated in favour of cuda-compiler, available from
# version 9.1 onwards
#
CUDA_PKGS="cuda-drivers cuda-cudart-dev-${CUDA_APT} "
if [ ${CUDA_VER_MAJOR} -ge 10 ]; then
  CUDA_PKGS+="cuda-compiler-${CUDA_APT} "
else
  CUDA_PKGS+="cuda-core-${CUDA_APT} "
fi

# cuda-cublas-dev is not available for 10.1; use the version from 10.0 instead
#
if [ ${CUDA_INSTALL_EXTRA_LIBS:-1} -ne 0 ]; then
  CUDA_PKGS+="cuda-cufft-dev-${CUDA_APT} cuda-cusparse-dev-${CUDA_APT} "

  if [ ${CUDA_VER_MAJOR} -ge 7 ]; then
    CUDA_PKGS+=" cuda-cusolver-dev-${CUDA_APT} "
  fi
  if [ ${CUDA_VER_MAJOR} -eq 10 -a ${CUDA_VER_MINOR} -eq 1 ]; then
    CUDA_PKGS+="cuda-cublas-dev-10-0 "
    LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:${LD_LIBRARY_PATH}
  else
    CUDA_PKGS+="cuda-cublas-dev-${CUDA_APT} "
  fi
fi

travis_retry sudo apt-get install -y ${CUDA_PKGS}
travis_retry sudo apt-get clean

# sudo ldconfig ${CUDA_HOME}/lib64
# sudo ldconfig ${CUDA_HOME}/nvvm/lib64


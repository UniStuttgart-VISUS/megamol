# MegaMol Ubuntu CI Environment

FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  ninja-build \
  git \
  clang \
# omp with clang:
  libomp-dev \
# vcpkg:
  curl zip unzip tar \
# ci checks:
  clang-format \
  clang-tidy \
# glfw:
  xorg-dev \
# gmp:
  autoconf libtool \
# mpfr:
  autoconf-archive \
# GL/glu.h:
  libglu1-mesa-dev \
  && rm -rf /var/lib/apt/lists/*

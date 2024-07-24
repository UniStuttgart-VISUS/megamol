# MegaMol Ubuntu CI Environment

FROM ubuntu:24.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  cmake \
  ninja-build \
  git \
  clang \
# omp with clang:
  libomp-dev \
# vcpkg:
  curl zip unzip tar \
# vcpkg zeromq:
  pkg-config \
# vcpkg meson (for libsystemd):
  python3-jinja2 \
# ci checks:
  clang-format clang-tidy \
# glfw:
  xorg-dev \
# libsodium / gmp:
  autoconf libtool \
# thrift:
  flex bison \
# GL/glu.h:
  libglu1-mesa-dev \
  && rm -rf /var/lib/apt/lists/*

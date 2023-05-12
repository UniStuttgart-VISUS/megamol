# MegaMol Fedora CI Environment

FROM fedora:37

RUN dnf -y update && dnf -y install \
  gcc \
  gcc-c++ \
  make \
  automake \
  cmake \
  ninja-build \
  git \
  clang \
# vcpkg
  curl zip unzip tar \
# ci checks:
  clang-tools-extra \
# openssl
  perl-core \
# glfw
  libXcursor-devel libXi-devel libXinerama-devel libXrandr-devel \
# gmp
  libtool \
# mpfr
  autoconf-archive \
# OpenGL
  mesa-libGL-devel mesa-libGLU-devel \
  && dnf clean all

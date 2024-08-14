# MegaMol Oracle Linux CI Environment

FROM oraclelinux:9

RUN dnf -y update && dnf --enablerepo=ol9_codeready_builder -y install \
  gcc \
  gcc-c++ \
  make \
  automake \
  cmake \
  ninja-build \
  git \
  clang \
# vcpkg:
  curl zip unzip tar \
# vcpkg meson (for libsystemd): \
  diffutils python3-jinja2 \
# ci checks:
  clang-tools-extra \
# openssl:
  perl-core \
# glfw:
  libXcursor-devel libXi-devel libXinerama-devel libXrandr-devel \
# gmp:
  libtool \
# mpfr:
  autoconf-archive \
# thrift:
  flex bison \
# OpenGL:
  mesa-libGL-devel mesa-libGLU-devel \
  && dnf clean all

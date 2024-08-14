#!/bin/bash
set -e
set -o pipefail

wget -qO - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc
sudo add-apt-repository --yes "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-17 main"
sudo apt-get install -y clang-format-17

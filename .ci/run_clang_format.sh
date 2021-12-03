#!/bin/bash

clang-format-12 --version
find . -iname "*.cpp" -o -iname "*.h" -o -iname "*.hpp" -o -iname "*.inl" | grep -Ev '^\./(externals|utils)/' | grep -Ev '^\./plugins/.*(3rd|external)/' | xargs clang-format-12 -i

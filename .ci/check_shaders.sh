#!/bin/bash
set -e
set -o pipefail

EXIT_CODE=0

SHADER_DIRS=()
SHADER_DIRS+=('core_gl/shaders')

for pdir in plugins/*; do
  # Ignore files
  if [[ ! -d "$pdir" ]]; then
    continue
  fi

  # Check for shaders dir
  if [ -d "$pdir/shaders" ]; then
    SHADER_DIRS+=("${pdir}/shaders")
  fi
done

# Build include path string
INCLUDE_PATHS=()
for sdir in "${SHADER_DIRS[@]}"
do
  INCLUDE_PATHS+=("-I$(pwd)/${sdir}")
done

# Iterate over all shaders
for sdir in "${SHADER_DIRS[@]}"
do
  echo "=================================================="
  echo "=== $sdir"
  echo "=================================================="

  shader_files=$(find "$sdir" -type f -regex '.*\.\(vert\|tesc\|tese\|geom\|frag\|comp\|mesh\)\.glsl$' | sort)

  while read -r sfile; do
    # Skip empty or deleted filename
    if [[ ! -f "$sfile" ]]; then
      continue
    fi

    LOCAL_INCLUDE_PATHS=("-I$(pwd)/$(dirname "$sfile")")

    glslang_exit_code=0
    glslangValidator -l "${INCLUDE_PATHS[@]}" "${LOCAL_INCLUDE_PATHS[@]}" $sfile || glslang_exit_code=$?
    if [[ $glslang_exit_code -ne 0 ]]; then
      EXIT_CODE=1
    fi

  done <<< "$shader_files"
done

exit $EXIT_CODE

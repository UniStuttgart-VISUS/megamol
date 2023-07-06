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
for sdir in "${SHADER_DIRS[@]}"; do
  INCLUDE_PATHS+=("-I$(pwd)/${sdir}")
done

# Read ignore list
readarray -t ignore_list < .ci/check-shaders-ignore.txt

echo "::add-matcher::.ci/check-shaders-problem-matchers.json"

# Iterate over all shaders
for sdir in "${SHADER_DIRS[@]}"; do
  echo "=================================================="
  echo "=== $sdir"
  echo "=================================================="

  shader_files=$(find "$sdir" -type f -regex '.*\.\(vert\|tesc\|tese\|geom\|frag\|comp\|mesh\)\.glsl$' | sort)

  while read -r sfile; do
    # Skip empty or deleted filename
    if [[ ! -f "$sfile" ]]; then
      continue
    fi

    # Skip ignored files
    ignore=false
    for ignore_entry in "${ignore_list[@]}"; do
      if [[ "$sfile" == $ignore_entry ]]; then
        ignore=true
        break
      fi
    done
    if [[ "$ignore" == true ]]; then
      echo "::warning::Ignore file $sfile"
      continue
    fi

    LOCAL_INCLUDE_PATHS=("-I$(pwd)/$(dirname "$sfile")")

    glslang_exit_code=0
    output="$(glslangValidator -l "${INCLUDE_PATHS[@]}" --p "#extension GL_GOOGLE_include_directive : require" "${LOCAL_INCLUDE_PATHS[@]}" $sfile)" || glslang_exit_code=$?
    if [[ $glslang_exit_code -ne 0 ]]; then
      echo ""
      echo "::error::########## glslangValidator found issues in $sfile ##########"
      echo "$output"
      EXIT_CODE=1
    fi

  done <<< "$shader_files"
done

echo "::remove-matcher owner=glslang-check::"

exit $EXIT_CODE

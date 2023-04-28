#!/bin/bash
set -e
set -o pipefail

EXIT_CODE=0

cd plugins/

for pdir in *; do
  # Ignore files
  if [[ ! -d "$pdir" ]]; then
    continue
  fi

  # Styleguide exceptions
  # "doc_" prefix is ignored
  pname=${pdir#doc_}
  # doc_template is named megamolplugin
  if [[ "$pdir" = "doc_template" ]]; then
    pname="megamolplugin"
  fi

  # Check main dirs for being lower casing
  for dir in "include" "src" "shaders" "resources"; do
    found_dir=$(find "$pdir" -maxdepth 1 -type d -iname "$dir")
    if [[ -d "$found_dir" ]]; then
      if [[ $found_dir != "$pdir/$dir" ]]; then
        EXIT_CODE=1
        echo "::error::The directory \"$found_dir\" must be all lower case!"
      fi
    fi
  done

  # Check include dir has exactly one subdir <plugin-name>
  if [[ -d "$pdir/include" ]]; then
    count=$(find "$pdir/include" -maxdepth 1 -mindepth 1 | wc -l)
    if [[ ! -d "$pdir/include/$pname" ]] || [[ $count -ne 1 ]]; then
      EXIT_CODE=1
      echo "::error::The directory \"$pdir/include\" must have exactly one subdir named \"$pname\"!"
    fi
  fi

  # Check shaders dir has exactly one subdir <plugin-name>
  if [ -d "$pdir/shaders" ]; then
    count=$(find "$pdir/shaders" -maxdepth 1 -mindepth 1 | wc -l)
    if [[ ! -d "$pdir/shaders/$pname" ]] || [[ $count -ne 1 ]]; then
      # TODO legacy feature, as long as btf files are present, allow bad structure
      btf_num=$(find "$pdir/shaders" -name "*.btf" | wc -l)
      if [[ $btf_num -eq 0 ]]; then
        EXIT_CODE=1
        echo "::error::The directory \"$pdir/shaders\" must have exactly one subdir named \"$pname\"!"
      fi
    fi
    # Check if all shaders end with .glsl (ignore /3rd/, also allow .txt and .md files) (TODO and .btf for now)
    readarray -d '' bad_shader_files < <(find "$pdir/shaders" -type f -not -path "*/3rd/*" -not -name "*.glsl" -not -name "*.md" -not -name "*.txt" -not -name "*.btf" -print0)
    if [[ ${#bad_shader_files[@]} -ne 0 ]]; then
      EXIT_CODE=1
      echo "::error::Found shader files with invalid extension in \"$pdir/shaders\"! Use .glsl, .md or .txt."
      printf '  %s\n' "${bad_shader_files[@]}"
    fi
  fi

  # Check CMake target name
  target=$(< "$pdir/CMakeLists.txt" tr -d '\n' | grep -oP "megamol_plugin[[:space:]]*\([[:space:]]*\K[a-zA-Z0-9_-]+")
  if ! [[ $target == "$pname" ]]; then
    EXIT_CODE=1
    echo "::error::The CMake target in \"$pdir/CMakeLists.txt\" is not named \"$pname\", found \"$target\"!"
  fi

  # Check main cpp file is named <plugin-name>.cpp
  if [[ ! -f "$pdir/src/$pname.cpp" ]]; then
    EXIT_CODE=1
    echo "::error::The main plugin cpp file is missing or named wrong, expected \"$pdir/src/$pname.cpp\"!"
  fi
done

exit $EXIT_CODE

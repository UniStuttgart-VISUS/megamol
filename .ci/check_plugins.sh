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

  # Check include dir has exactly one subdir <plugin-name>
  if [[ -d "$pdir/include" ]]; then
    count=$(ls -1q "$pdir/include" | wc -l)
    if [[ ! -d "$pdir/include/$pname" ]] || [[ $count -ne 1 ]]; then
      EXIT_CODE=1
      echo "The directory \"$pdir/include\" must have exactly one subdir named \"$pname\"!"
    fi
  fi

  # Check shaders dir has exactly one subdir <plugin-name>
  if [ -d "$pdir/shaders" ]; then
    count=$(ls -1q "$pdir/shaders" | wc -l)
    if [[ ! -d "$pdir/shaders/$pname" ]] || [[ $count -ne 1 ]]; then
      # TODO legacy feature, as long as btf files are present, allow bad structure
      btf_num=$(find "$pdir/shaders" -name "*.btf" | wc -l)
      if [[ $btf_num -eq 0 ]]; then
        EXIT_CODE=1
        echo "The directory \"$pdir/shaders\" must have exactly one subdir named \"$pname\"!"
      fi
    fi
  fi

  # Check CMake target name
  target=$(cat "$pdir/CMakeLists.txt" | tr -d '\n' | grep -oP "megamol_plugin[[:space:]]*\([[:space:]]*\K[a-zA-Z0-9_-]+")
  if ! [[ $target == "$pname" ]]; then
    EXIT_CODE=1
    echo "The CMake target in \"$pdir/CMakeLists.txt\" is not named \"$pname\", found \"$target\"!"
  fi

  # Check main cpp file is named <plugin-name>.cpp
  if [[ ! -f "$pdir/src/$pname.cpp" ]]; then
    EXIT_CODE=1
    echo "The main plugin cpp file is missing or named wrong, expected \"$pdir/src/$pname.cpp\"!"
  fi
done

exit $EXIT_CODE

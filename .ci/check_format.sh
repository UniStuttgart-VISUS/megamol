#!/bin/bash
set -e
set -o pipefail

EXIT_CODE=0

# Find all files, ignore .git dirs.
file_list=$(find . -type d -name '.git' -prune -o -type f -print | sort)

while read -r file; do
  # ignore files ignored by git
  if git check-ignore -q "$file"; then
    continue
  fi

  # only process file if mime type is text
  mime=$(file -b --mime-type "$file")
  if ! [[ $mime == "text/"* ]]; then
    continue
  fi

  # ignore vcpkg ports, which are taken from upstream
  if [[ $file == "./cmake/vcpkg_ports/"* ]]; then
    if [[ $file == *"/implot/"* ]]; then
      continue
    fi
  fi

  # ignore 3rd dirs in plugins
  if [[ $file == "./plugins/"* ]]; then
    if [[ $file == *"/3rd/"* ]]; then
      continue
    fi
    if [[ $file == *"/protein/msms/"* ]]; then
      continue
    fi
  fi

  # ignore externals
  # TODO we probably want to distinguish more granular between 3rd-party and our files here
  if [[ $file == "./externals/"* ]]; then
    continue
  fi

  # is cpp file?
  is_cpp=false
  if [[ $file == *.cpp ]] || [[ $file == *.h ]] || [[ $file == *.hpp ]] || [[ $file == *.inl ]]; then
    is_cpp=true
  fi

  # === File tests ===

  # ClangFormat
  if [[ "$is_cpp" == true ]]; then
    if [[ $1 == "fix" ]]; then
      clang-format-14 -i "$file"
    else
      # Workaround "set -e" and store exit code
      format_exit_code=0
      output="$(clang-format-14 --dry-run --Werror "$file" 2>&1)" || format_exit_code=$?
      if [[ $format_exit_code -ne 0 ]]; then
        EXIT_CODE=1
        echo "::error::ClangFormat found issues in: $file"
        #echo "$output"
        # Show detailed diff. Requires ClangFormat to run again, but should mostly affect only a few files.
        clang-format-14 "$file" | diff --color=always -u "$file" - || true
      fi
    fi
  fi

  # Check if file is UTF-8 (or ASCII)
  encoding=$(file -b --mime-encoding "$file")
  if ! [[ $encoding == "us-ascii" || $encoding == "utf-8" ]]; then
    if [[ $1 == "fix" ]]; then
      tmp_file=$(mktemp)
      iconv -f "$encoding" -t utf-8 -o "$tmp_file" "$file"
      mv -f "$tmp_file" "$file"
    else
      EXIT_CODE=1
      echo "::error::File is not UTF-8 encoded: $file ($encoding)"
    fi
  fi

  # Check if file contains CRLF line endings
  fileinfo=$(file "$file")
  if [[ $fileinfo == *"CRLF"* ]]; then
    if [[ $1 == "fix" ]]; then
      sed -i 's/\r$//' "$file"
    else
      EXIT_CODE=1
      echo "::error::File contains CRLF line endings: $file"
    fi
  fi

  # Check if file starts with BOM
  if [[ $fileinfo == *"BOM"* ]]; then
    if [[ $1 == "fix" ]]; then
      sed -i '1s/^\xEF\xBB\xBF//' "$file"
    else
      EXIT_CODE=1
      echo "::error::File starts with BOM: $file"
    fi
  fi

  # Check if file ends with newline
  if [[ -n "$(tail -c 1 "$file")" ]]; then
    if [[ $1 == "fix" ]]; then
      sed -i -e '$a\' "$file"
    else
      EXIT_CODE=1
      echo "::error::File does not end with new line: $file"
    fi
  fi

  # Check if file contains tabs
  if grep -qP "\t" "$file"; then
    if [[ $1 == "fix" ]]; then
      tmp_file=$(mktemp)
      expand -t 4 "$file" > "$tmp_file"
      mv -f "$tmp_file" "$file"
    else
      EXIT_CODE=1
      echo "::error::File contains tabs: $file"
    fi
  fi

done <<< "$file_list"

exit $EXIT_CODE

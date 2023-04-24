#!/bin/bash
set -e
set -o pipefail

# Command line parameter
_fix=false
_uncommitted=false
_branch=false
while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--fix) _fix=true ;;
    -u|--uncommitted) _uncommitted=true ;;
    -b|--branch) _branch=true ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

EXIT_CODE=0

# Fast mode, only check changed files
if [[ "$_uncommitted" == true ]]; then
  # Git diff including staged + untracked files
  file_list=$(git diff --name-only HEAD ; git ls-files --exclude-standard --others .)
elif [[ "$_branch" == true ]]; then
  # Git diff work dir to master
  file_list=$(git diff --name-only master ; git ls-files --exclude-standard --others .)
else
  # Find all files, ignore .git dirs.
  file_list=$(find . -type d -name '.git' -prune -o -type f -print | sort)
fi

while read -r file; do
  # Skip empty or deleted filename
  if [[ ! -f "$file" ]]; then
    continue
  fi

  # ignore files ignored by git
  if git check-ignore -q "$file"; then
    continue
  fi

  # only process file if mime type is text
  mime=$(file -b --mime-type "$file")
  if [[ $mime != "text/"* ]]; then
    continue
  fi

  # ignore vcpkg ports, which are taken from upstream
  if [[ $file == "./cmake/vcpkg_ports/"* ]]; then
    if [[ $file == *"/implot/"* ]]; then
      continue
    fi
  fi

  # ignore 3rd party dirs
  if [[ $file == *"/3rd/"* ]]; then
    continue
  fi
  if [[ $file == "./plugins/protein_gl/msms/"* ]]; then
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
    if [[ "$_fix" == true ]]; then
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
  if [[ $encoding != "us-ascii" && $encoding != "utf-8" ]]; then
    if [[ "$_fix" == true ]]; then
      tmp_file=$(mktemp)
      iconv -f "$encoding" -t utf-8 -o "$tmp_file" "$file"
      mv -f "$tmp_file" "$file"
    else
      EXIT_CODE=1
      echo "::error::File is not UTF-8 encoded: $file ($encoding)"
    fi
  fi

  # Check if file contains CRLF line endings
  fileinfo=$(file -k "$file")
  if [[ $fileinfo == *"CRLF"* ]]; then
    if [[ "$_fix" == true ]]; then
      sed -i 's/\r$//' "$file"
    else
      EXIT_CODE=1
      echo "::error::File contains CRLF line endings: $file"
    fi
  fi

  # Check if file starts with BOM
  if [[ $fileinfo == *"BOM"* ]]; then
    if [[ "$_fix" == true ]]; then
      sed -i '1s/^\xEF\xBB\xBF//' "$file"
    else
      EXIT_CODE=1
      echo "::error::File starts with BOM: $file"
    fi
  fi

  # Check if file ends with newline
  if [[ -n "$(tail -c 1 "$file")" ]]; then
    if [[ "$_fix" == true ]]; then
      sed -i -e '$a\' "$file"
    else
      EXIT_CODE=1
      echo "::error::File does not end with new line: $file"
    fi
  fi

  # Check if file contains tabs
  if grep -qP "\t" "$file"; then
    if [[ "$_fix" == true ]]; then
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

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

# Constants
copyright_header_regex="^\/\*\*
 \* MegaMol
 \* Copyright \(c\) [0-9]{4}, MegaMol Dev Team
 \* All rights reserved\.
 \*\/"
copyright_header_template="/**
 * MegaMol
 * Copyright (c) <YYYY>, MegaMol Dev Team
 * All rights reserved.
 */"

EXIT_CODE=0

# Fast mode, only check changed files
if [[ "$_uncommitted" == true ]]; then
  # Git diff including staged + untracked files
  file_list=$(git diff --name-only HEAD ; git ls-files --exclude-standard --others .)
elif [[ "$_branch" == true ]]; then
  # Git diff work dir to master
  file_list=$(git diff --name-only master ; git ls-files --exclude-standard --others .)
else
  # Find all files, ignore .git dirs. Remove leading './' from results.
  file_list=$(find . -type d -name '.git' -prune -o -type f -print | sort | cut -c3-)
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
  if [[ $file == "cmake/vcpkg_ports/"* ]]; then
    if [[ $file == *"/boost-stacktrace/"* ]]; then
      continue
    fi
    if [[ $file == *"/glm/"* ]]; then
      continue
    fi
  fi

  # ignore 3rd party dirs
  if [[ $file == *"/3rd/"* ]]; then
    continue
  fi
  if [[ $file == "plugins/protein_gl/msms/"* ]]; then
    continue
  fi

  # is cpp file?
  is_cpp=false
  if [[ $file == *.cpp ]] || [[ $file == *.h ]] || [[ $file == *.hpp ]] || [[ $file == *.inl ]]; then
    is_cpp=true
  fi

  # === File tests ===

  # C++ checks
  if [[ "$is_cpp" == true ]]; then
    # ClangFormat
    if [[ "$_fix" == true ]]; then
      clang-format-17 -i "$file"
    else
      # Workaround "set -e" and store exit code
      format_exit_code=0
      output="$(clang-format-17 --dry-run --Werror "$file" 2>&1)" || format_exit_code=$?
      if [[ $format_exit_code -ne 0 ]]; then
        EXIT_CODE=1
        echo "::error::ClangFormat found issues in: $file"
        #echo "$output"
        # Show detailed diff. Requires ClangFormat to run again, but should mostly affect only a few files.
        clang-format-17 "$file" | diff --color=always -u "$file" - || true
      fi
    fi

    # Copyright header
    # TODO Disabled because simply replacing the first comment may is dangerous and should not happen in an automated script.
    #file_head=$(head -n 5 "$file")
    #if [[ ! "$file_head" =~ $copyright_header_regex ]]; then
    #  if [[ "$_fix" == true ]]; then
    #    year=$(git log --follow --format=%ad --date=format:'%Y' "$file" | tail -1)
    #    file_content=$(<"$file")
    #    # Remove first comment in file
    #    start_comment_regex="^\/\*"
    #    if [[ "$file_content" =~ $start_comment_regex ]]; then
    #      file_content=${file_content#*\*/}
    #    fi
    #    file_header="${copyright_header_template/<YYYY>/"$year"}"
    #    echo "$file_header" > "$file"
    #    echo "$file_content" >> "$file"
    #  else
    #    EXIT_CODE=1
    #    echo "::error::Missing or wrong copyright header in $file"
    #  fi
    #fi
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

#!/bin/bash

EXIT_CODE=0

# Store a list of files which are changed in the current branch compared to the common merge base with master.
changed_files=$(git diff --name-only origin/master...HEAD)
git_root=$(git rev-parse --show-toplevel)

file_list=$(find . -type f | sort)
while read -r file; do
  #ignore .git dir
  if [[ $file == "./.git/"* ]]; then
    continue
  fi

  # ignore files ignored by git
  if git check-ignore -q "$file"; then
    continue
  fi

  # only process file if mime type is text
  mime=$(file -b --mime-type "$file")
  if ! [[ $mime == "text/"* ]]; then
    continue
  fi

  # ignore 3rd dirs in plugins
  if [[ $file == "./plugins/"* ]]; then
    if [[ $file == *"/3rd/"* ]]; then
      continue
    fi
  fi

  # === File tests ===

  # Check if file is UTF-8 (or ASCII)
  encoding=$(file -b --mime-encoding "$file")
  if ! [[ $encoding == "us-ascii" || $encoding == "utf-8" ]]; then
    EXIT_CODE=1
    echo "ERROR: File is not UTF-8 encoded: $file ($encoding)"
  fi

  # Check if file contains CRLF line endings
  fileinfo=$(file "$file")
  if [[ $fileinfo == *"CRLF"* ]]; then
    EXIT_CODE=1
    echo "ERROR: File contains CRLF line endings: $file"
  fi

  # Check if file starts with BOM
  if [[ $fileinfo == *"BOM"* ]]; then
    EXIT_CODE=1
    echo "ERROR: File starts with BOM: $file"
  fi

  # Check if file ends with newline
  if [[ -n "$(tail -c 1 "$file")" ]]; then
    #EXIT_CODE=1 # TODO enable
    echo "ERROR: File does not end with new line: $file"
  fi

  # Check if file contains tabs
  if grep -qP "\t" "$file"; then
    #EXIT_CODE=1 # TODO enable
    echo "ERROR: File contains tabs: $file"
  fi

  # Get absolute file path and path relative to git root (for comparison with changed_files).
  # Assume we are always anywhere within the git work dir and just remove git_root based on string pattern.
  file_abs="$(cd "$(dirname "$file")"; pwd -P)/$(basename "$file")"
  file_git="${file_abs##"$git_root/"}"

  # If the current file is not changed within this git branch skip the following tests
  if [[ $changed_files != *"$file_git"* ]]; then
    continue
  fi

  # === File tests (only on changed files) ===

done <<< "$file_list"

exit $EXIT_CODE

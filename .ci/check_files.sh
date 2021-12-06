#!/bin/bash

EXIT_CODE=0

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
    if [[ $file == *"/protein/msms/"* ]]; then
      continue
    fi
  fi

  # === File tests ===

  # Check if file is UTF-8 (or ASCII)
  encoding=$(file -b --mime-encoding "$file")
  if ! [[ $encoding == "us-ascii" || $encoding == "utf-8" ]]; then
    if [[ $1 == "fix" ]]; then
      tmp_file=$(mktemp)
      iconv -f "$encoding" -t utf-8 -o "$tmp_file" "$file"
      mv -f "$tmp_file" "$file"
    else
      EXIT_CODE=1
      echo "ERROR: File is not UTF-8 encoded: $file ($encoding)"
    fi
  fi

  # Check if file contains CRLF line endings
  fileinfo=$(file "$file")
  if [[ $fileinfo == *"CRLF"* ]]; then
    if [[ $1 == "fix" ]]; then
      sed -i 's/\r$//' "$file"
    else
      EXIT_CODE=1
      echo "ERROR: File contains CRLF line endings: $file"
    fi
  fi

  # Check if file starts with BOM
  if [[ $fileinfo == *"BOM"* ]]; then
    if [[ $1 == "fix" ]]; then
      sed -i '1s/^\xEF\xBB\xBF//' "$file"
    else
      EXIT_CODE=1
      echo "ERROR: File starts with BOM: $file"
    fi
  fi

  # Check if file ends with newline
  if [[ -n "$(tail -c 1 "$file")" ]]; then
    if [[ $1 == "fix" ]]; then
      sed -i -e '$a\' "$file"
    else
      EXIT_CODE=1
      echo "ERROR: File does not end with new line: $file"
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
      echo "ERROR: File contains tabs: $file"
    fi
  fi

done <<< "$file_list"

exit $EXIT_CODE

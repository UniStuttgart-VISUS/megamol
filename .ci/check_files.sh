#!/bin/bash

EXIT_CODE=0

find . -type f -print0 | while read -d $'\0' file
do
  # only process file if mime type is text
  mime=$(file -b --mime-type "$file")
  if ! [[ $mime == "text/"* ]]; then
    continue
  fi

  # Check if file is UTF-8 (or ASCII)
  encoding=$(file -b --mime-encoding "$file")
  if ! [[ $encoding == "us-ascii" || $encoding == "utf-8" ]]; then
    EXIT_CODE=1
    echo "ERROR: File is not UTF-8 encoded: $file ($encoding)"
    continue
  fi

  # Check if file contains CRLF line endings
  fileinfo=$(file "$file")
  if [[ $fileinfo == *"CRLF"* ]]; then
    EXIT_CODE=1
    echo "ERROR: File contains CRLF line endings: $file"
    continue
  fi

  # Check if file starts with BOM
  if [[ $fileinfo == *"BOM"* ]]; then
    EXIT_CODE=1
    echo "ERROR: File starts with BOM: $file"
    continue
  fi
done

exit $EXIT_CODE

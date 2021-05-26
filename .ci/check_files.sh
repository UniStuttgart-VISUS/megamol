#!/bin/bash

EXIT_CODE=0

# Search CRLF line endings
BAD_LINE_ENDING_FILES=$(find . -type f -exec file {} \; | grep CRLF)
if [[ $BAD_LINE_ENDING_FILES ]]; then
    EXIT_CODE=1
    echo "############################################################"
    echo " ERROR: Files with CRLF line ending found!"
    echo "############################################################"
    echo "$BAD_LINE_ENDING_FILES"
else
    echo "Good: No CRLF line endings found."
fi

# Search non ASCII/UTF-8 Text files
BAD_ENCODING_FILES=$(find . -type f -exec file --mime {} \; | grep -Pi ": text/.*" | grep -vi "charset=us-ascii\|charset=utf-8")
if [[ $BAD_ENCODING_FILES ]]; then
    EXIT_CODE=1
    echo "############################################################"
    echo " ERROR: The following text files are not UTF-8 encoded!"
    echo "############################################################"
    echo "$BAD_ENCODING_FILES"
else
    echo "Good: All text files are UTF-8 encoded."
fi

# Search for BOM
BAD_FILES_WITH_BOM=$(grep -rl $'^\xEF\xBB\xBF' .)
if [[ $BAD_FILES_WITH_BOM ]]; then
    EXIT_CODE=1
    echo "############################################################"
    echo " ERROR: The following text files start with BOM!"
    echo "############################################################"
    echo "$BAD_FILES_WITH_BOM"
else
    echo "Good: All text files do not start with BOM."
fi

exit $EXIT_CODE

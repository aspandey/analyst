#!/bin/bash

# Script to remove all text before and including the string "Combined Text: " from every line in a file.
# Note: This script assumes the file is UTF-8 encoded and the space after the colon is a regular space.
# If the space is a non-breaking space (U+00A0), the pattern might need minor adjustment.

# Check if a file is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <filepath>"
    echo "Example: $0 log_file.txt"
    exit 1
fi

FILEPATH="$1"

# Check if the file exists
if [ ! -f "$FILEPATH" ]; then
    echo "Error: File not found at $FILEPATH"
    exit 1
fi

echo "Processing file: $FILEPATH"

# The 'sed' command performs the substitution:
# - 's/pattern/replacement/' is the substitution command.
# - '.*Combined Text: ' is the pattern:
#     - '.*' matches any character zero or more times (everything from the start of the line).
#     - 'Combined Text: ' is the exact string to match and include in the removal.
# - '' (empty string) is the replacement.
# - '-i' performs the change in place (modifies the file directly).
# sed -i 's/.*Combined Text: //g' "$FILEPATH"
awk '/Combined Text: / { sub(/.*Combined Text: /, ""); print } !/Combined Text: / { print }' "$FILEPATH" > temp && mv temp "$FILEPATH"

if [ $? -eq 0 ]; then
    echo "Successfully removed prefix from all lines in $FILEPATH."
else
    echo "An error occurred during sed execution."
fi

# Alternative using awk (removes everything before the field containing "Combined Text:")

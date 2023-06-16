#!/bin/bash

# Gnerate directory structure up to depth 3and save it to a file
echo '```' >> .directory_structure.txt
tree -d -L 3 --noreport >> .directory_structure.txt
echo '```' >> .directory_structure.txt

# Unique identifier for the section to be replaced
SOF="<!--structure_begin-->"
EOF="<!--structure_end-->"

# Create a temporary file
TEMP_FILE=$(mktemp)

# Print everything before the identifier to the temporary file
sed -n "1,/$SOF/p" README.md > $TEMP_FILE

# Append the directory structure to the temporary file
cat .directory_structure.txt >> $TEMP_FILE

# Print everything after the identifier to the temporary file

sed -n "/$EOF/,\$p" README.md >> $TEMP_FILE

# Replace the README with the temporary file
mv $TEMP_FILE README.md

# Clean up
rm .directory_structure.txt 

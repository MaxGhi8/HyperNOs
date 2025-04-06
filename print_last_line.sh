#!/bin/bash
# This script is to move to ~/ray_results and is used to plot all the runs inside
# result.json file that starts with a given accuracy.

# Set the accuracy
accuracy=0.01

# Iterate through all directories and subdirectories
find . -type f -name "result.json" | while read -r file; do
    last_line=$(tail -n 1 "$file")
    
    if [[ "$last_line" == "{\"relative_loss\": $accuracy"* ]]; then
	echo "File: $file"
	
	echo "$last_line"

	echo "---------------------------"
    fi
done


#!/bin/bash

wget https://zenodo.org/record/10406879/files/data.zip

if [ -d "data" ]; then
    echo "Data directory exists. Extracting and moving subfolders..."
    mkdir -p temp_extract
    unzip data.zip -d temp_extract
    if [ -d "temp_extract/data" ]; then
        find temp_extract/data -mindepth 1 -maxdepth 1 -type d -exec mv {} data/ \;
    fi
    rm -rf temp_extract
else
    echo "Data directory does not exist. Performing normal extraction..."
    unzip data.zip
fi

rm -rf data.zip
echo "Process completed."
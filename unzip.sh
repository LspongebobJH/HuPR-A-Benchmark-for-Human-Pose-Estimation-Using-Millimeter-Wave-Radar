#!/bin/bash

# Loop through all zip files in the folder
root_name=/home/ubuntu/hupr/raw_data/frames/single_
idx_list=(15 16 38 40 41 42 17 39  244 245 246 249 250 251 252 253 254 247 248 255 256)
for idx in ${idx_list[@]}; do
    zip_file=$root_name$idx.zip
    
    # Extract the filename without the extension
    filename=$(basename "$zip_file" ".zip")
    
    # Extract the contents of the zip file into the subdirectory
    unzip "$zip_file" -d "/home/ubuntu/hupr/raw_data/frames/$filename"
    # echo $zip_file

done

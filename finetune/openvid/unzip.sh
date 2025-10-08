#!/bin/bash 

# Define the destination directory
destination_dir="./videos/"

# Make sure the destination directory exists
mkdir -p "$destination_dir"

# Extract files 1 to 9 from each zip archive
for i in {40..49}
do
  # Define the current zip file
  zip_file="./OpenVid_part${i}.zip"
  
  # Extract each file individually from the current zip file
  unzip -j "$zip_file" -d "$destination_dir" 
  rm "$zip_file"
done

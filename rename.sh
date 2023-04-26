#!/bin/bash
# credit: chatgpt

# Loop through all subdirectories of raw_data/iwr1843/HuPR/
for dir in raw_data/iwr1843/HuPR/single_*; do

  # Rename the file in the "hori" subdirectory to "adc_data.bin"
  mv "${dir}/hori/"* "${dir}/hori/adc_data.bin"

  # Rename the file in the "vert" subdirectory to "adc_data.bin"
  mv "${dir}/vert/"* "${dir}/vert/adc_data.bin"

done
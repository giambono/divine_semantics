#!/bin/bash

# Path to metadata file
metadata_file="pages_to_extract.txt"

# Check if metadata file exists
if [[ ! -f "$metadata_file" ]]; then
    echo "Metadata file $metadata_file does not exist."
    exit 1
fi

# Loop through each line in the metadata file
while IFS= read -r line; do
  # Extract the PDF base name and page ranges from the metadata
  name=$(echo "$line" | awk '{print $1}')
  page_ranges=$(echo "$line" | awk '{print $2}')

  # Specify the corresponding input PDF file
  input_pdf="commedia_${name}.pdf"

  # Check if the PDF file exists
  if [[ ! -f "$input_pdf" ]]; then
    echo "Error: PDF file '$input_pdf' not found. Skipping..."
    continue
  fi

  # Output debug information to see which PDF and page ranges are being processed
  echo "Processing '$input_pdf' with page ranges: $page_ranges"

  # Loop through each page range and extract pages into a separate PDF
  IFS=',' read -ra ranges <<< "$page_ranges"
  for range in "${ranges[@]}"; do
    # Generate output PDF name based on the page range
    output_pdf="${name}_pages_${range}.pdf"

    # Check if the range extraction was successful
    echo "Attempting to extract pages $range from $input_pdf..."

    # Extract the pages using pdftk
    if pdftk "$input_pdf" cat "$range" output "$output_pdf"; then
      echo "Successfully extracted pages $range from $input_pdf to $output_pdf"
    else
      echo "Error extracting pages $range from $input_pdf"
    fi

    # If using qpdf instead, you can use the following command:
    # qpdf "$input_pdf" --pages "$input_pdf" "$range" -- "$output_pdf"
  done

done < "$metadata_file"

import re

def clean_text_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    cleaned_lines = []
    for line in lines:
        # Remove numbers and unwanted symbols
        line = re.sub(r'\d+', '', line)  # Remove digits
        line = re.sub(r'[^A-Za-z\s.,;!?\-]', '', line)  # Keep only letters and basic punctuation
        line = re.sub(r'\s+', ' ', line).strip()  # Normalize spaces

        if line:  # Ignore empty lines
            cleaned_lines.append(line)

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(cleaned_lines))

if __name__ == "__main__":
    input_filename = "texts_to_be_cleaned/commedia_singleton_inferno_text_en.txt"  # Change this to the actual filename
    output_filename = "texts_cleaned/commedia_singleton_inferno_text_en.txt"
    clean_text_file(input_filename, output_filename)
    print(f"Cleaned text saved to {output_filename}")

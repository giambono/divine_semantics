import re
import pandas as pd

def parse_inferno_text(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    canto_number = 0
    canto_text = []
    data = []

    for line in lines:
        line = line.strip()

        # Detect canto headers
        match = re.match(r'CANTO\s+(\w+)', line, re.IGNORECASE)
        if match:
            if canto_number > 0:  # Store previous canto data
                data.append({"canto": canto_number, "text": " ".join(canto_text)})
                canto_text = []

            # Convert Roman numeral to integer
            canto_number = roman_to_int(match.group(1).upper())
        else:
            if line:  # Ignore empty lines
                canto_text.append(line)

    # Store last canto
    if canto_number > 0 and canto_text:
        data.append({"canto": canto_number, "text": " ".join(canto_text)})

    # Create DataFrame
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding='utf-8')

    return df

def roman_to_int(roman):
    roman_numerals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    result = 0
    prev_value = 0
    for char in reversed(roman.upper()):
        value = roman_numerals.get(char, 0)
        if value < prev_value:
            result -= value
        else:
            result += value
        prev_value = value
    return result


if __name__ == "__main__":
    input_filename = "commedia_singleton_inferno_text_en.txt"  # Change this to the actual filename
    output_filename = "commedia_singleton_inferno_text_en_parsed.csv"
    df = parse_inferno_text(input_filename, output_filename)
    print(f"Parsed text saved to {output_filename}")
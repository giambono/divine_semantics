import re

def extract_outlines_and_translations(input_txt, outline_output, translation_output):
    outlines = {}
    translations = {}
    current_canto_outline = None
    current_canto_translation = None
    extracting_outline = False
    extracting_translation = False

    with open(input_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Detect start of an outline section
        if line.startswith("OUTLINE: INFERNO "):
            current_canto_outline = line.replace("OUTLINE: ", "").strip()
            outlines[current_canto_outline] = []
            extracting_outline = True
            extracting_translation = False  # Stop translation extraction

        # Detect start of a translation section
        elif "INFERNO" in line and re.match(r"^INFERNO\s+[IVXLCDM]+$", line):
            current_canto_translation = line.strip()
            translations[current_canto_translation] = []
            extracting_translation = True
            extracting_outline = False  # Stop outline extraction

        # Collect content under current outline
        elif extracting_outline and current_canto_outline:
            outlines[current_canto_outline].append(line)

        # Collect translation text
        elif extracting_translation and current_canto_translation:
            translations[current_canto_translation].append(line)

    # Save outlines
    with open(outline_output, "w", encoding="utf-8") as f:
        for canto, content in outlines.items():
            f.write(f"{canto}\n")
            f.write("=" * len(canto) + "\n")
            f.write("\n".join(content))
            f.write("\n\n")

    # Save translations
    with open(translation_output, "w", encoding="utf-8") as f:
        for canto, content in translations.items():
            f.write(f"{canto}\n")
            f.write("=" * len(canto) + "\n")
            f.write("\n".join(content))
            f.write("\n\n")

    print(f"Extraction complete.\nOutlines saved to: {outline_output}\nTranslations saved to: {translation_output}")



input_txt = "/home/rfflpllcn/IdeaProjects/divine_semantics/sandbox/commedia_inferno_hollander.txt"
outline_output = "outlines.txt"
translation_output = "translations.txt"

extract_outlines_and_translations(input_txt, outline_output, translation_output)



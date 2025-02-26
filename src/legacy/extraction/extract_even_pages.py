import fitz  # PyMuPDF

def extract_even_pages(input_pdf, output_pdf):
    doc = fitz.open(input_pdf)
    new_doc = fitz.open()

    for page_num in range(1, len(doc), 2):  # Even pages (0-based index)
        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

    new_doc.save(output_pdf)
    new_doc.close()
    print(f"Even pages saved to {output_pdf}")

# Usage
extract_even_pages(r"/home/rfflpllcn/IdeaProjects/divine_semantics/data/commedia_singleton_inferno_text.pdf", r"/home/rfflpllcn/IdeaProjects/divine_semantics/data/commedia_singleton_inferno_text_en.pdf")

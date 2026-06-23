import fitz  # PyMuPDF


def extract_pdf_text(filename):
    try:
        doc = fitz.open(filename)

        text = ""

        for page_number, page in enumerate(doc, start=1):
            page_text = page.get_text()
            text += f"\n\n--- Page {page_number} ---\n\n"
            text += page_text

        doc.close()

        return text

    except FileNotFoundError:
        print(f"Error: {filename} was not found.")
        return ""

    except Exception as e:
        print("Something went wrong while reading the PDF:", e)
        return ""


def save_text_file(filename, text):
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(text)

        print(f"Text saved successfully to {filename}")

    except Exception as e:
        print("Something went wrong while saving the file:", e)


pdf_text = extract_pdf_text("paper.pdf")

if pdf_text.strip() != "":
    print("PDF text extracted successfully.")
    print(pdf_text[:1000])  # show first 1000 characters only

    save_text_file("extracted_text.txt", pdf_text)

else:
    print("No readable text found in the PDF.")
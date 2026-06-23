import os
from pathlib import Path
import fitz
from google import genai


PAPERS_FOLDER = Path("papers")
REPORTS_FOLDER = Path("reports")
KNOWLEDGE_BASE_FILE = REPORTS_FOLDER / "paper_knowledge_base.txt"


def extract_pdf_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""

        for page_number, page in enumerate(doc, start=1):
            page_text = page.get_text()
            text += f"\n\n--- Page {page_number} ---\n\n"
            text += page_text

        doc.close()
        return text

    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""


def shorten_text(text, max_chars=18000):
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key is None:
        raise ValueError("Gemini API key not found. Set GEMINI_API_KEY first.")

    return genai.Client(api_key=api_key)


def generate_with_fallback(client, prompt):
    models_to_try = [
        "gemini-2.5-flash-lite",
        "gemini-flash-latest",
        "gemini-2.5-flash"
    ]

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text

        except Exception as e:
            print(f"Model failed: {model_name}")
            print("Error:", e)
            print("-" * 60)

    return "All Gemini models failed right now. Try again later."


def summarize_paper(client, paper_name, paper_text):
    paper_text = shorten_text(paper_text)

    prompt = f"""
You are NeuroScholar-AI Pro, a scientific paper analysis assistant.

Analyze the research paper text below.

Extract the following information:

1. Paper name
2. Main research problem
3. Field / topic area
4. Brain region / biological system / molecule studied
5. Method used
6. Dataset / sample / participants / model organism
7. Main findings
8. Limitations
9. Future work mentioned
10. Important variables
11. Possible research gaps inside this paper
12. Possible connections to other neuroscience topics
13. 3 possible research ideas inspired by this paper

Rules:
- Use simple English.
- Do not claim anything is proven unless the paper clearly says it.
- If information is missing, write "Not available in the provided text."
- Keep the output structured.

Paper file name:
{paper_name}

Paper text:
{paper_text}
"""

    return generate_with_fallback(client, prompt)


def analyze_all_papers():
    REPORTS_FOLDER.mkdir(exist_ok=True)

    pdf_files = sorted(PAPERS_FOLDER.glob("*.pdf"))

    if len(pdf_files) == 0:
        print("No PDF files found inside the papers folder.")
        return

    print(f"Found {len(pdf_files)} PDF papers.")

    client = get_gemini_client()

    full_knowledge_base = ""

    for index, pdf_path in enumerate(pdf_files, start=1):
        print(f"\nAnalyzing paper {index}: {pdf_path.name}")

        text = extract_pdf_text(pdf_path)

        print(f"Extracted characters: {len(text)}")

        if text.strip() == "":
            print(f"Skipping {pdf_path.name} because no readable text was found.")
            continue

        summary = summarize_paper(client, pdf_path.name, text)

        paper_report = f"""
============================================================
PAPER {index}: {pdf_path.name}
============================================================

{summary}

"""

        full_knowledge_base += paper_report

        individual_report_path = REPORTS_FOLDER / f"{pdf_path.stem}_summary.txt"

        with open(individual_report_path, "w", encoding="utf-8") as file:
            file.write(paper_report)

        print(f"Saved individual summary: {individual_report_path}")

    with open(KNOWLEDGE_BASE_FILE, "w", encoding="utf-8") as file:
        file.write(full_knowledge_base)

    print("\nAll papers analyzed.")
    print(f"Knowledge base saved to: {KNOWLEDGE_BASE_FILE}")


if __name__ == "__main__":
    analyze_all_papers()
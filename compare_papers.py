import os
import fitz
from google import genai


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
        print(f"Error reading {pdf_path}:", e)
        return ""


def shorten_text(text, max_chars=18000):
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def compare_papers(paper1_text, paper2_text):
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key is None:
        return "Error: Gemini API key not found."

    client = genai.Client(api_key=api_key)

    paper1_text = shorten_text(paper1_text)
    paper2_text = shorten_text(paper2_text)

    prompt = f"""
You are NeuroScholar-AI Pro, a scientific research comparison assistant.

Compare the two research papers using ONLY the text provided.

Your output should be structured like this:

1. Paper 1 Summary
- Research problem:
- Method used:
- Main findings:
- Limitations:
- Future work:

2. Paper 2 Summary
- Research problem:
- Method used:
- Main findings:
- Limitations:
- Future work:

3. Similarities Between Papers

4. Differences Between Papers

5. Possible Research Gaps

6. Possible New Hypotheses
For each hypothesis, include:
- Hypothesis:
- Why it is possible:
- Evidence from papers:
- What is still uncertain:

7. Suggested Experiments
For each experiment, include:
- Experiment idea:
- Control group:
- Experimental group:
- Measurement method:
- Expected result:
- Limitation:

Important rules:
- Use simple English.
- Do not claim a discovery is proven.
- Say "possible hypothesis" instead of "new discovery".
- If information is missing, say it is not available in the provided paper text.

Paper 1 text:
{paper1_text}

Paper 2 text:
{paper2_text}
"""

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


paper1_path = "papers/paper1.pdf"
paper2_path = "papers/paper2.pdf"

paper1_text = extract_pdf_text(paper1_path)
paper2_text = extract_pdf_text(paper2_path)

if paper1_text.strip() == "":
    print("Paper 1 text could not be extracted.")

elif paper2_text.strip() == "":
    print("Paper 2 text could not be extracted.")

else:
    print("Both papers extracted successfully.")
    print("Comparing papers...\n")

    comparison = compare_papers(paper1_text, paper2_text)

    print(comparison)

    with open("paper_comparison_report.txt", "w", encoding="utf-8") as file:
        file.write(comparison)

    print("\nComparison saved to paper_comparison_report.txt")
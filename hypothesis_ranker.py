import os
from google import genai


def read_report(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()

    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return ""


def rank_hypotheses(report_text):
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key is None:
        return "Error: Gemini API key not found."

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are NeuroScholar-AI Pro, a scientific hypothesis evaluator.

You are given a report generated from comparing two neuroscience research papers.

Your task:
Extract the possible hypotheses and suggested experiments from the report, then rank them.

For each hypothesis, provide:

1. Hypothesis
2. Scientific reasoning
3. Evidence from the papers
4. Novelty: High / Medium / Low
5. Feasibility: High / Medium / Low
6. Scientific impact: High / Medium / Low
7. Evidence strength: Strong / Moderate / Weak
8. Ethical risk: High / Medium / Low
9. Overall priority: High / Medium / Low
10. Why this priority was assigned

Rules:
- Do not say the hypothesis is proven.
- Use "possible hypothesis" or "research direction".
- Be honest if evidence is weak.
- Use simple English.
- Do not invent paper details not present in the report.

Comparison report:
{report_text}
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


report = read_report("paper_comparison_report.txt")

if report.strip() == "":
    print("No comparison report found to rank.")

else:
    print("Comparison report loaded.")
    print("Ranking hypotheses...\n")

    ranked_output = rank_hypotheses(report)

    print(ranked_output)

    with open("ranked_hypotheses_report.txt", "w", encoding="utf-8") as file:
        file.write(ranked_output)

    print("\nRanked hypotheses saved to ranked_hypotheses_report.txt")
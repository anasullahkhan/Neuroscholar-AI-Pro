import os
from google import genai


def read_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()

    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return ""


def design_experiments(ranked_hypotheses_text):
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key is None:
        return "Error: Gemini API key not found."

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are NeuroScholar-AI Pro, a neuroscience experiment design assistant.

You are given ranked hypotheses generated from comparing neuroscience research papers.

Your task:
Convert the strongest hypotheses into detailed experiment designs.

For each experiment, provide:

1. Experiment title
2. Research question
3. Hypothesis being tested
4. Independent variable
5. Dependent variable
6. Control group
7. Experimental group
8. Participants / sample type
9. Materials or tools needed
10. Procedure
11. Measurement method
12. Expected result
13. Possible confounding factors
14. Ethical concerns
15. Limitations
16. Why this experiment matters

Rules:
- Do not claim the hypothesis is proven.
- Use simple English.
- Be scientifically cautious.
- If human drug experiments are risky, suggest safer alternatives like observational study, animal model, or non-invasive measurement.
- Do not invent details that were not supported by the ranked hypothesis report.

Ranked hypotheses report:
{ranked_hypotheses_text}
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


ranked_text = read_file("ranked_hypotheses_report.txt")

if ranked_text.strip() == "":
    print("No ranked hypotheses report found.")

else:
    print("Ranked hypotheses loaded.")
    print("Designing experiments...\n")

    experiment_report = design_experiments(ranked_text)

    print(experiment_report)

    with open("experiment_design_report.txt", "w", encoding="utf-8") as file:
        file.write(experiment_report)

    print("\nExperiment design saved to experiment_design_report.txt")
import os
import time
from pathlib import Path
from google import genai


KNOWLEDGE_BASE_FILE = Path("reports") / "paper_knowledge_base.txt"
OUTPUT_FILE = Path("reports") / "cross_paper_research_ideas.txt"


def read_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()

    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return ""


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
        for attempt in range(1, 4):
            try:
                print(f"Trying {model_name}, attempt {attempt}...")

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                return response.text

            except Exception as e:
                print(f"Model failed: {model_name}, attempt {attempt}")
                print("Error:", e)
                print("-" * 60)

                time.sleep(10)

    return "All Gemini models failed right now. Try again later."
      


def detect_cross_paper_research_ideas(knowledge_base_text):
    client = get_gemini_client()

    prompt = f"""
You are NeuroScholar-AI Pro, an advanced neuroscience research assistant.

You are given a knowledge base created from multiple neuroscience research papers.

Your task:
Analyze all papers together and generate cross-paper research insights.

Output the following sections:

1. Big Picture Summary
- What broad scientific area do these papers cover?
- What major brain systems, molecules, behaviors, or cognitive processes appear?

2. Cross-Paper Connections
Find connections between papers.
Example:
- Paper A studies stress.
- Paper B studies reading.
- Possible connection: stress hormones may affect reading comprehension.

3. Research Gaps
For each gap, give:
- Gap title
- What is missing?
- Which papers suggest this gap?
- Why the gap matters

4. Possible Hypotheses
For each hypothesis, give:
- Hypothesis
- Why it is possible
- Evidence from the papers
- What is still uncertain

5. Possible Experiments
For each experiment, give:
- Experiment title
- Research question
- Control group
- Experimental group
- Measurement method
- Expected result
- Limitations
- Ethical concerns

6. Novel Research Ideas
For each idea, give:
- Idea title
- Scientific reasoning
- Novelty: High / Medium / Low
- Feasibility: High / Medium / Low
- Impact: High / Medium / Low
- Evidence strength: Strong / Moderate / Weak
- Overall priority: High / Medium / Low

7. Final Recommendation
List the top 3 best research ideas to pursue first.

Rules:
- Do not claim that the AI discovered proven facts.
- Use "possible hypothesis", "research direction", or "experimental idea".
- Be scientifically cautious.
- Use simple English.
- If evidence is weak, say it clearly.
- Do not invent details not supported by the knowledge base.

Knowledge base:
{knowledge_base_text}
"""

    return generate_with_fallback(client, prompt)


knowledge_base = read_file(KNOWLEDGE_BASE_FILE)

if knowledge_base.strip() == "":
    print("No knowledge base found. Run multi_paper_analyzer.py first.")

else:
    print("Knowledge base loaded.")
    print("Generating cross-paper research ideas...\n")

    output = detect_cross_paper_research_ideas(knowledge_base)

    print(output)

if output.startswith("All Gemini models failed"):
    print("\nNo useful report was saved because all models failed.")
else:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write(output)

    print(f"\nCross-paper research ideas saved to {OUTPUT_FILE}")    
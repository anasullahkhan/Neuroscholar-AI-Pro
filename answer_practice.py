from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
import glob
import os


def read_chunk_files():
    chunk_files = sorted(glob.glob("chunk_*.txt"))
    chunks = []

    for filename in chunk_files:
        with open(filename, "r", encoding="utf-8") as file:
            text = file.read()

        chunks.append({
            "filename": filename,
            "text": text
        })

    return chunks


def create_embeddings(model, chunks):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)
    return embeddings


def search_relevant_chunks(question, model, chunks, chunk_embeddings, top_k=3):
    question_embedding = model.encode([question])
    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]

    ranked_results = sorted(
        enumerate(similarities),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for index, score in ranked_results[:top_k]:
        results.append({
            "filename": chunks[index]["filename"],
            "text": chunks[index]["text"],
            "score": score
        })

    return results


def generate_answer(question, relevant_chunks):
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key is None:
        return "Error: Gemini API key not found."

    client = genai.Client(api_key=api_key)

    context = ""

    for chunk in relevant_chunks:
        context += f"\nSource: {chunk['filename']}\n"
        context += chunk["text"]
        context += "\n\n"

    prompt = f"""
You are NeuroScholar-AI, a scientific research assistant.

Answer the user's question using ONLY the paper context below.

Rules:
1. Use simple English.
2. Do not invent information.
3. If the answer is not available in the context, say:
"The paper context does not provide enough information."
4. Mention the chunk files used.

User question:
{question}

Paper context:
{context}

Final answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


chunks = read_chunk_files()

if len(chunks) == 0:
    print("No chunk files found. Run chunk_practice.py first.")

else:
    print(f"Loaded {len(chunks)} chunks.")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = create_embeddings(embedding_model, chunks)

    question = input("Ask a question about the paper: ")

    relevant_chunks = search_relevant_chunks(
        question,
        embedding_model,
        chunks,
        chunk_embeddings,
        top_k=3
    )

    print("\nRelevant chunks found:")
    for chunk in relevant_chunks:
        print(chunk["filename"], "score:", round(chunk["score"], 4))

    answer = generate_answer(question, relevant_chunks)

    print("\nAI Answer:\n")
    print(answer)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import glob


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

    top_results = ranked_results[:top_k]

    results = []

    for index, score in top_results:
        results.append({
            "filename": chunks[index]["filename"],
            "text": chunks[index]["text"],
            "score": score
        })

    return results


# Main program
chunks = read_chunk_files()

if len(chunks) == 0:
    print("No chunk files found. Run chunk_practice.py first.")

else:
    print(f"Loaded {len(chunks)} chunks.")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunk_embeddings = create_embeddings(model, chunks)

    question = input("Ask a question about the paper: ")

    results = search_relevant_chunks(
        question,
        model,
        chunks,
        chunk_embeddings,
        top_k=3
    )

    print("\nTop relevant chunks:\n")

    for result in results:
        print("File:", result["filename"])
        print("Similarity Score:", round(result["score"], 4))
        print("Text Preview:")
        print(result["text"][:800])
        print("\n" + "-" * 80 + "\n")
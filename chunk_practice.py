def read_text_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            text = file.read()
        return text

    except FileNotFoundError:
        print(f"Error: {filename} was not found.")
        return ""


def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("  ", " ")
    text = text.strip()
    return text


def split_into_chunks(text, chunk_size=500, overlap=100):
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append(chunk_text)

        start = end - overlap

    return chunks


def save_chunks(chunks):
    for i, chunk in enumerate(chunks, start=1):
        filename = f"chunk_{i}.txt"

        with open(filename, "w", encoding="utf-8") as file:
            file.write(chunk)

    print(f"{len(chunks)} chunks saved successfully.")


# Main program
text = read_text_file("extracted_text.txt")

if text != "":
    cleaned = clean_text(text)

    chunks = split_into_chunks(cleaned, chunk_size=500, overlap=100)

    print(f"Total chunks created: {len(chunks)}")
    print("\nFirst chunk preview:\n")
    print(chunks[0][:1000])

    save_chunks(chunks)

else:
    print("No text found to chunk.")
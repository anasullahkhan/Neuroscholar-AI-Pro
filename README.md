# 🧠 NeuroScholar-AI Pro

**NeuroScholar-AI Pro** is an AI-powered multi-paper research assistant that analyzes scientific PDF papers using **PDF extraction, semantic embeddings, retrieval-augmented generation, Gemini, and Streamlit**.

It helps users upload research papers, ask natural questions, summarize papers, identify research gaps, generate possible hypotheses, suggest experiment designs, and export a final research report.

---

## 🚀 Live Demo

🔗 **App:** `https://neuroscholar-ai-pro.streamlit.app/`

---

## ✨ What It Can Do

* 📄 Upload multiple scientific PDF papers
* 🧹 Extract and clean text using PyMuPDF
* 🧩 Split papers into searchable chunks
* 🧠 Generate semantic embeddings using SentenceTransformers
* 🔎 Retrieve relevant chunks using cosine similarity
* 🤖 Answer any prompt using Gemini + RAG
* 📌 Summarize papers in simple English
* 🔬 Find possible research gaps
* 💡 Generate possible scientific hypotheses
* 🧪 Suggest experiment designs
* 📥 Export a final research report

---

## 🧠 Why This Project Matters

Scientific literature is growing faster than any student or researcher can manually read. NeuroScholar-AI Pro helps reduce that overload by acting as a research companion that can analyze multiple papers and surface useful insights.

This project does **not** claim to make scientific discoveries by itself. Instead, it helps researchers and students explore papers faster, compare ideas, and generate possible research directions for further expert review.

---

## 🏗️ System Architecture

```text
PDF Upload
   ↓
Text Extraction with PyMuPDF
   ↓
Text Cleaning
   ↓
Chunking
   ↓
SentenceTransformer Embeddings
   ↓
Cosine Similarity Search
   ↓
Relevant Context Retrieval
   ↓
Gemini LLM
   ↓
Answer / Summary / Gaps / Hypotheses / Experiments / Report
```

---

## 🛠️ Tech Stack

| Layer             | Technology                     |
| ----------------- | ------------------------------ |
| Frontend          | Streamlit                      |
| PDF Processing    | PyMuPDF                        |
| Embeddings        | SentenceTransformers           |
| Similarity Search | Scikit-learn Cosine Similarity |
| LLM               | Gemini API                     |
| Language          | Python                         |
| Deployment        | Streamlit Cloud                |

---

## 📸 Screenshots

Add screenshots here after deployment:

```markdown
![Home Page](<img width="1920" height="1080" alt="Screenshot (111)" src="https://github.com/user-attachments/assets/6d853558-a6db-4155-9100-368d3c58793b" />
)
![Ask Anything](<img width="1920" height="1080" alt="Screenshot (112)" src="https://github.com/user-attachments/assets/7e02e540-269f-4c79-b620-c231678bc296" />
)
![Research Gaps](<img width="1920" height="1080" alt="Screenshot (113)" src="https://github.com/user-attachments/assets/dde6a6f1-86f8-4d23-ba74-897a11b49f7a" />
)
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/anasullahkhan/Neuroscholar-AI-Pro.git
cd Neuroscholar-AI-Pro
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔐 API Key Setup

Create this file locally:

```text
.streamlit/secrets.toml
```

Add your Gemini API key:

```toml
GEMINI_API_KEY = "your_real_gemini_api_key"
```

Never upload `secrets.toml` to GitHub.

The repository should only contain:

```text
.streamlit/secrets.example.toml
```

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

or:

```bash
python -m streamlit run app.py
```

---

## ☁️ Deployment

This app can be deployed on **Streamlit Cloud**.

Deployment steps:

1. Push the project to GitHub.
2. Open Streamlit Cloud.
3. Select this repository.
4. Set the main file as:

```text
app.py
```

5. Add this in Streamlit Secrets:

```toml
GEMINI_API_KEY = "your_real_gemini_api_key"
```

6. Deploy the app.

---

## 🧪 Example Prompts

You can ask:

```text
What did the paper say?
```

```text
Summarize all uploaded papers in simple English.
```

```text
Find research gaps between these papers.
```

```text
Generate possible hypotheses from these papers.
```

```text
Suggest experiment designs based on the research gaps.
```

```text
Make a comparison table between the uploaded papers.
```

---

## 📌 Project Highlights

This project demonstrates:

* Retrieval-Augmented Generation
* Multi-document AI analysis
* Semantic search
* Scientific paper summarization
* LLM-based hypothesis generation
* Experiment design assistance
* API key security
* Streamlit app deployment
* End-to-end AI product building

---

## ⚠️ Disclaimer

NeuroScholar-AI Pro is an educational and research-support tool. It can generate possible ideas and summaries, but it should not be used as a substitute for expert scientific review, clinical judgment, or peer-reviewed validation.

---

## 👨‍💻 Author

**Anasullah Khan**
B.Tech Artificial Intelligence & Machine Learning
GitHub: [@anasullahkhan](https://github.com/anasullahkhan)

---

## ⭐ Support

If you find this project useful, consider giving it a star on GitHub.

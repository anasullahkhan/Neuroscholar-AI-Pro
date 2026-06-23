import os
import time

import fitz  # PyMuPDF
import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# -----------------------------
# Default app settings
# -----------------------------

top_k = 5
chunk_size = 500
overlap = 100


# -----------------------------
# App setup
# -----------------------------

st.set_page_config(
    page_title="NeuroScholar-AI Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------
# Custom UI styling
# -----------------------------

st.markdown(
    """
<style>
    /* App background */
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, rgba(124, 58, 237, 0.25), transparent 35%),
            radial-gradient(circle at top right, rgba(14, 165, 233, 0.22), transparent 35%),
            linear-gradient(135deg, #07111f 0%, #0f172a 45%, #111827 100%);
        color: #e5e7eb;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 17, 31, 0.0);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(15, 23, 42, 0.98), rgba(17, 24, 39, 0.98));
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    /* Main block width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hero */
    .hero-card {
        padding: 2.2rem 2.4rem;
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(30, 41, 59, 0.92), rgba(15, 23, 42, 0.78)),
            linear-gradient(135deg, rgba(124, 58, 237, 0.35), rgba(14, 165, 233, 0.20));
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 30px 90px rgba(0, 0, 0, 0.38);
        margin-bottom: 1.25rem;
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -0.05em;
        margin: 0;
        line-height: 1.05;
        background: linear-gradient(90deg, #e0f2fe, #c4b5fd, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        color: #cbd5e1;
        margin-top: 0.8rem;
        max-width: 980px;
        line-height: 1.65;
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1.2rem;
    }

    .badge {
        padding: 0.42rem 0.78rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.25);
        color: #dbeafe;
        font-size: 0.88rem;
        font-weight: 700;
    }

    /* Cards */
    .metric-card {
        padding: 1.15rem 1.2rem;
        border-radius: 22px;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 18px 45px rgba(0,0,0,0.25);
        min-height: 118px;
    }

    .metric-title {
        color: #94a3b8;
        font-size: 0.86rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.45rem;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.55rem;
        font-weight: 900;
        margin-bottom: 0.25rem;
    }

    .metric-text {
        color: #cbd5e1;
        font-size: 0.93rem;
        line-height: 1.5;
    }

    .section-card {
        padding: 1.2rem 1.35rem;
        border-radius: 24px;
        background: rgba(15, 23, 42, 0.70);
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow: 0 16px 50px rgba(0,0,0,0.22);
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 900;
        color: #f8fafc;
        margin-bottom: 0.25rem;
    }

    .section-subtitle {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin-bottom: 0.7rem;
        line-height: 1.55;
    }

    .soft-warning {
        padding: 0.85rem 1rem;
        border-radius: 18px;
        border: 1px solid rgba(251, 191, 36, 0.35);
        background: rgba(120, 53, 15, 0.30);
        color: #fde68a;
        font-size: 0.95rem;
        margin: 0.8rem 0;
    }

    .success-box {
        padding: 0.9rem 1rem;
        border-radius: 18px;
        border: 1px solid rgba(34, 197, 94, 0.35);
        background: rgba(20, 83, 45, 0.28);
        color: #bbf7d0;
        font-size: 0.95rem;
        margin: 0.8rem 0;
    }

    .answer-box {
        padding: 1.15rem 1.25rem;
        border-radius: 22px;
        background: rgba(2, 6, 23, 0.64);
        border: 1px solid rgba(96, 165, 250, 0.26);
        box-shadow: 0 20px 50px rgba(0,0,0,0.28);
        margin-top: 1rem;
    }

    .chip {
        display: inline-block;
        padding: 0.45rem 0.7rem;
        margin: 0.2rem 0.2rem 0.2rem 0;
        border-radius: 999px;
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.20);
        color: #bfdbfe;
        font-size: 0.86rem;
        font-weight: 700;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.55rem;
        background: rgba(15, 23, 42, 0.55);
        padding: 0.55rem;
        border-radius: 20px;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 16px;
        padding: 0.7rem 1rem;
        color: #cbd5e1;
        font-weight: 800;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.75), rgba(14, 165, 233, 0.60));
        color: #ffffff !important;
    }

    /* Inputs */
    textarea, input {
        border-radius: 16px !important;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 16px;
        padding: 0.72rem 1.05rem;
        font-weight: 900;
        border: 1px solid rgba(147, 197, 253, 0.28);
        background: linear-gradient(135deg, #7c3aed, #0ea5e9);
        color: white;
        box-shadow: 0 14px 30px rgba(14, 165, 233, 0.18);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 38px rgba(124, 58, 237, 0.28);
        border: 1px solid rgba(255,255,255,0.32);
        color: white;
    }

    /* Download button */
    div.stDownloadButton > button {
        border-radius: 16px;
        padding: 0.72rem 1.05rem;
        font-weight: 900;
        border: 1px solid rgba(34, 197, 94, 0.28);
        background: linear-gradient(135deg, #16a34a, #0ea5e9);
        color: white;
    }

    /* Expander */
    .streamlit-expanderHeader {
        border-radius: 14px;
        font-weight: 800;
    }

    /* Hide Streamlit footer/menu a bit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True
)


# -----------------------------
# Hero UI
# -----------------------------

st.markdown(
    """
<div class="hero-card">
    <div class="hero-title">NeuroScholar-AI Pro</div>
    <div class="hero-subtitle">
        Upload scientific papers and ask anything. The app reads PDFs, builds a knowledge base,
        retrieves relevant source chunks, and helps generate summaries, comparisons, research gaps,
        hypotheses, experiment ideas, and final reports.
    </div>
    <div class="badge-row">
        <span class="badge">📄 Multi-PDF Analysis</span>
        <span class="badge">🔎 Semantic Search</span>
        <span class="badge">🧠 Gemini RAG</span>
        <span class="badge">🔬 Research Gap Detection</span>
        <span class="badge">🧪 Experiment Planning</span>
    </div>
</div>
""",
    unsafe_allow_html=True
)

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.markdown(
        """
<div class="metric-card">
    <div class="metric-title">Pipeline</div>
    <div class="metric-value">PDF → RAG → Report</div>
    <div class="metric-text">Extract, chunk, embed, retrieve, reason, and export.</div>
</div>
""",
        unsafe_allow_html=True
    )

with metric_col2:
    st.markdown(
        """
<div class="metric-card">
    <div class="metric-title">Ask Mode</div>
    <div class="metric-value">Any Prompt</div>
    <div class="metric-text">Summary, tables, gaps, methods, limitations, or experiments.</div>
</div>
""",
        unsafe_allow_html=True
    )

with metric_col3:
    st.markdown(
        """
<div class="metric-card">
    <div class="metric-title">Safety</div>
    <div class="metric-value">Cautious Science</div>
    <div class="metric-text">Suggests possible ideas, not fake proven discoveries.</div>
</div>
""",
        unsafe_allow_html=True
    )


# -----------------------------
# Helper functions
# -----------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_api_key():
    """Use Streamlit secrets in deployment, or an environment variable locally."""
    try:
        key = st.secrets["GEMINI_API_KEY"]
        if key:
            return key
    except Exception:
        pass

    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key

    return None


def get_gemini_client():
    api_key = get_api_key()

    if api_key is None:
        return None

    return genai.Client(api_key=api_key)


def is_model_failure(text):
    if not isinstance(text, str):
        return False

    failure_markers = [
        "All Gemini models failed",
        "503 UNAVAILABLE",
        "Gemini API key is not configured",
        "Error: Gemini API key"
    ]

    return any(marker in text for marker in failure_markers)


def generate_with_fallback(prompt, max_attempts=1):
    client = get_gemini_client()

    if client is None:
        return (
            "Error: Gemini API key is not configured. "
            "Add GEMINI_API_KEY in Streamlit secrets."
        )

    model_name = "gemini-2.5-flash-lite"

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except Exception as e:
        error_text = str(e)

        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
            return (
                "Gemini quota is exhausted right now. "
                "The API key is valid, but the free request limit has been reached. "
                "Try again later or contact the app owner for a full report."
            )

        if "503" in error_text or "UNAVAILABLE" in error_text:
            return (
                "Gemini is temporarily overloaded. "
                "Please try again after some time."
            )

        return f"Gemini API error: {error_text}"


def extract_pdf_text(uploaded_file):
    try:
        file_bytes = uploaded_file.getvalue()
        doc = fitz.open(stream=file_bytes, filetype="pdf")

        text = ""

        for page_number, page in enumerate(doc, start=1):
            page_text = page.get_text()
            text += f"\n\n--- Page {page_number} ---\n\n"
            text += page_text

        doc.close()
        return text

    except Exception as e:
        return f"PDF_EXTRACTION_ERROR: {e}"


def clean_text(text):
    text = text.replace("\n", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def shorten_text(text, max_chars=18000):
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def split_into_chunks(text, paper_name, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 1

    if overlap >= chunk_size:
        overlap = 100

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        if chunk_text.strip():
            chunks.append({
                "paper_name": paper_name,
                "chunk_id": chunk_id,
                "text": chunk_text
            })

        chunk_id += 1
        start = end - overlap

    return chunks


def summarize_paper(paper_name, paper_text):
    paper_text = shorten_text(paper_text)

    prompt = f"""
You are NeuroScholar-AI Pro, a scientific paper analysis assistant.

Analyze the research paper text below.

Extract:

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
12. Possible connections to other neuroscience or biomedical topics
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

    return generate_with_fallback(prompt)


def build_embeddings(chunks):
    model = load_embedding_model()
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(chunk_texts)
    return embeddings


def retrieve_relevant_chunks(question, chunks, embeddings, top_k=5):
    if embeddings is None or len(chunks) == 0:
        return []

    model = load_embedding_model()
    question_embedding = model.encode([question])

    similarities = cosine_similarity(question_embedding, embeddings)[0]

    ranked = sorted(
        enumerate(similarities),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for index, score in ranked[:top_k]:
        results.append({
            "paper_name": chunks[index]["paper_name"],
            "chunk_id": chunks[index]["chunk_id"],
            "text": chunks[index]["text"],
            "score": float(score)
        })

    return results


def format_relevant_chunks_for_prompt(relevant_chunks):
    context = ""

    for chunk in relevant_chunks:
        context += (
            f"\nSource: {chunk['paper_name']} | "
            f"Chunk {chunk['chunk_id']} | "
            f"Score {chunk['score']:.4f}\n"
        )
        context += chunk["text"]
        context += "\n\n"

    return context


def answer_any_prompt(question, knowledge_base, chunks, embeddings, top_k=5):
    relevant_chunks = retrieve_relevant_chunks(
        question=question,
        chunks=chunks,
        embeddings=embeddings,
        top_k=top_k
    )

    relevant_context = format_relevant_chunks_for_prompt(relevant_chunks)

    prompt = f"""
You are NeuroScholar-AI Pro, a scientific research assistant.

The user can give ANY prompt about the uploaded papers.

Use BOTH:
1. The full paper knowledge base
2. The most relevant source chunks

Your job:
- Answer broad questions using the paper knowledge base.
- Answer specific questions using the relevant source chunks.
- If the user asks for a summary, summarize clearly.
- If the user asks "what did the paper say", explain the paper in simple English.
- If the user asks for a table, create a table.
- If the user asks for research gaps, identify possible gaps.
- If the user asks for hypotheses, generate possible hypotheses.
- If the user asks for experiments, suggest possible experiments.
- If the user asks for limitations, list limitations.
- If the user asks for methods, explain methods.
- If the user uses angry/casual language, ignore the tone and answer professionally.
- Do not say "not enough information" unless the knowledge base and chunks are both useless.
- Do not invent unsupported scientific claims.
- Use simple English.
- Mention source paper names/chunk IDs when using source chunks.

User prompt:
{question}

Full paper knowledge base:
{knowledge_base}

Most relevant source chunks:
{relevant_context}

Final answer:
"""

    answer = generate_with_fallback(prompt)

    if is_model_failure(answer):
        fallback_text = (
            "Gemini is temporarily overloaded, so I cannot generate a fresh answer right now.\n\n"
            "But your paper knowledge base is already built. Here is the available context:\n\n"
            f"{knowledge_base[:5000]}"
        )
        if len(knowledge_base) > 5000:
            fallback_text += "\n\n[Preview shortened. Full report is available in Export.]"

        return fallback_text, relevant_chunks

    return answer, relevant_chunks


def generate_cross_paper_ideas(knowledge_base):
    prompt = f"""
You are NeuroScholar-AI Pro, an advanced neuroscience research assistant.

You are given a knowledge base created from multiple scientific papers.

Generate cross-paper research insights.

Output:

1. Big Picture Summary

2. Cross-Paper Connections

3. Research Gaps
For each gap:
- Gap title
- What is missing?
- Which papers suggest this gap?
- Why the gap matters

4. Possible Hypotheses
For each hypothesis:
- Hypothesis
- Why it is possible
- Evidence from papers
- What is still uncertain

5. Possible Experiments
For each experiment:
- Experiment title
- Research question
- Control group
- Experimental group
- Measurement method
- Expected result
- Limitations
- Ethical concerns

6. Novel Research Ideas
For each idea:
- Idea title
- Scientific reasoning
- Novelty: High / Medium / Low
- Feasibility: High / Medium / Low
- Impact: High / Medium / Low
- Evidence strength: Strong / Moderate / Weak
- Overall priority: High / Medium / Low

7. Final Recommendation
List the top 3 research directions to pursue first.

Rules:
- Do not claim proven discoveries.
- Use "possible hypothesis", "research direction", or "experimental idea".
- Be scientifically cautious.
- Use simple English.
- If evidence is weak, say it clearly.
- Do not invent details not supported by the knowledge base.

Knowledge base:
{knowledge_base}
"""

    return generate_with_fallback(prompt)


def generate_experiment_plan(research_ideas):
    prompt = f"""
You are NeuroScholar-AI Pro, a neuroscience experiment design assistant.

You are given research gaps, hypotheses, and ideas generated from multiple scientific papers.

Convert the strongest ideas into detailed experiment designs.

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
- If human drug experiments are risky, suggest safer alternatives such as observational studies, animal models, or non-invasive measurement.
- Do not invent unsupported paper details.

Research ideas:
{research_ideas}
"""

    return generate_with_fallback(prompt)


# -----------------------------
# Session state
# -----------------------------

if "paper_summaries" not in st.session_state:
    st.session_state.paper_summaries = {}

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "cross_paper_ideas" not in st.session_state:
    st.session_state.cross_paper_ideas = ""

if "experiment_plan" not in st.session_state:
    st.session_state.experiment_plan = ""


# -----------------------------
# Tabs
# -----------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Upload Papers",
    "✨ Ask Anything",
    "🔬 Research Gaps & Ideas",
    "🧪 Experiments",
    "📥 Export"
])


# -----------------------------
# Tab 1: Upload papers
# -----------------------------

with tab1:
    st.markdown(
        """
<div class="section-card">
    <div class="section-title">📄 Upload Scientific Papers</div>
    <div class="section-subtitle">
        Upload one or more PDF papers. NeuroScholar will extract text, summarize each paper,
        create source chunks, and build semantic embeddings.
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Upload multiple PDF papers",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and len(uploaded_files) > 2:
        st.error("Demo limit: upload maximum 2 PDFs. Contact the owner for full research reports.")
        uploaded_files = uploaded_files[:2]

    col_a, col_b = st.columns([1, 2])

    with col_a:
        build_clicked = st.button("🚀 Build Knowledge Base")

    with col_b:
        if uploaded_files:
            st.markdown(
                f"<div class='success-box'>✅ {len(uploaded_files)} PDF file(s) selected.</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='soft-warning'>Upload PDFs first, then build the knowledge base.</div>",
                unsafe_allow_html=True
            )

    if build_clicked:
        if get_api_key() is None:
            st.error("Gemini API key is not configured. Add GEMINI_API_KEY in Streamlit secrets.")

        elif not uploaded_files:
            st.error("Upload at least one PDF.")

        else:
            all_chunks = []
            knowledge_base = ""
            paper_summaries = {}

            progress = st.progress(0)

            for i, uploaded_file in enumerate(uploaded_files):
                paper_name = uploaded_file.name

                with st.spinner(f"Extracting and analyzing {paper_name}..."):
                    raw_text = extract_pdf_text(uploaded_file)

                    if raw_text.startswith("PDF_EXTRACTION_ERROR"):
                        st.error(f"{paper_name}: {raw_text}")
                        continue

                    cleaned = clean_text(raw_text)

                    if cleaned.strip() == "":
                        st.warning(f"No readable text found in {paper_name}. Skipping.")
                        continue

                    summary = summarize_paper(paper_name, cleaned)

                    paper_summaries[paper_name] = summary

                    knowledge_base += f"""
============================================================
PAPER: {paper_name}
============================================================

{summary}

"""

                    paper_chunks = split_into_chunks(
                        cleaned,
                        paper_name=paper_name,
                        chunk_size=chunk_size,
                        overlap=overlap
                    )

                    all_chunks.extend(paper_chunks)

                progress.progress((i + 1) / len(uploaded_files))

            if len(all_chunks) == 0:
                st.error("No chunks were created. Check your PDFs.")

            else:
                with st.spinner("Creating embeddings for all paper chunks..."):
                    embeddings = build_embeddings(all_chunks)

                st.session_state.paper_summaries = paper_summaries
                st.session_state.knowledge_base = knowledge_base
                st.session_state.chunks = all_chunks
                st.session_state.embeddings = embeddings

                st.success("Knowledge base created successfully.")

    if st.session_state.knowledge_base:
        stat1, stat2, stat3 = st.columns(3)

        with stat1:
            st.markdown(
                f"""
<div class="metric-card">
    <div class="metric-title">Papers</div>
    <div class="metric-value">{len(st.session_state.paper_summaries)}</div>
    <div class="metric-text">Analyzed documents</div>
</div>
""",
                unsafe_allow_html=True
            )

        with stat2:
            st.markdown(
                f"""
<div class="metric-card">
    <div class="metric-title">Chunks</div>
    <div class="metric-value">{len(st.session_state.chunks)}</div>
    <div class="metric-text">Searchable source units</div>
</div>
""",
                unsafe_allow_html=True
            )

        with stat3:
            kb_words = len(st.session_state.knowledge_base.split())
            st.markdown(
                f"""
<div class="metric-card">
    <div class="metric-title">Knowledge Base</div>
    <div class="metric-value">{kb_words}</div>
    <div class="metric-text">Approximate words generated</div>
</div>
""",
                unsafe_allow_html=True
            )

        st.subheader("Paper Knowledge Base Preview")
        st.text_area(
            "Knowledge Base",
            st.session_state.knowledge_base,
            height=400
        )


# -----------------------------
# Tab 2: Ask anything
# -----------------------------

with tab2:
    st.markdown(
        """
<div class="section-card">
    <div class="section-title">✨ Ask Anything About Uploaded Papers</div>
    <div class="section-subtitle">
        Ask naturally. Summary, table, gaps, hypotheses, methods, limitations, experiments, or simple explanation.
        The app uses the full paper knowledge base plus the most relevant source chunks.
    </div>
    <span class="chip">what did the paper say?</span>
    <span class="chip">explain like I am 10</span>
    <span class="chip">make a comparison table</span>
    <span class="chip">find research gaps</span>
    <span class="chip">suggest experiments</span>
</div>
""",
        unsafe_allow_html=True
    )

    question = st.text_area(
        "Enter any prompt",
        placeholder="Example: Explain the paper in simple English and give 5 key takeaways.",
        height=135
    )

    answer_clicked = st.button("🧠 Generate Answer")

    if answer_clicked:
        if st.session_state.knowledge_base.strip() == "":
            st.error("Build the knowledge base first in the Upload Papers tab.")

        elif question.strip() == "":
            st.error("Type a prompt first.")

        else:
            with st.spinner("Thinking across the full knowledge base and source chunks..."):
                answer, relevant_chunks = answer_any_prompt(
                    question=question,
                    knowledge_base=st.session_state.knowledge_base,
                    chunks=st.session_state.chunks,
                    embeddings=st.session_state.embeddings,
                    top_k=top_k
                )

            st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
            st.subheader("AI Answer")
            st.write(answer)
            st.markdown("</div>", unsafe_allow_html=True)

            if relevant_chunks:
                st.subheader("Most Relevant Source Chunks")

                for chunk in relevant_chunks:
                    with st.expander(
                        f"{chunk['paper_name']} | Chunk {chunk['chunk_id']} | Score {chunk['score']:.4f}"
                    ):
                        st.write(chunk["text"])


# -----------------------------
# Tab 3: Research gaps and ideas
# -----------------------------

with tab3:
    st.markdown(
        """
<div class="section-card">
    <div class="section-title">🔬 Cross-Paper Research Gaps and Ideas</div>
    <div class="section-subtitle">
        Generate cautious research gaps, possible hypotheses, experiment directions, novelty/feasibility/impact ratings,
        and final research recommendations.
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    if st.button("🔬 Generate Research Gaps and Ideas"):
        if st.session_state.knowledge_base.strip() == "":
            st.error("Build the knowledge base first.")

        else:
            with st.spinner("Generating cross-paper research insights..."):
                ideas = generate_cross_paper_ideas(st.session_state.knowledge_base)
                st.session_state.cross_paper_ideas = ideas

            st.success("Research ideas generated.")

    if st.session_state.cross_paper_ideas:
        st.text_area(
            "Cross-Paper Research Ideas",
            st.session_state.cross_paper_ideas,
            height=600
        )


# -----------------------------
# Tab 4: Experiments
# -----------------------------

with tab4:
    st.markdown(
        """
<div class="section-card">
    <div class="section-title">🧪 Experiment Design Assistant</div>
    <div class="section-subtitle">
        Convert research ideas into structured experiment plans with variables, groups, measurement methods,
        expected results, limitations, ethics, and significance.
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    if st.button("🧪 Generate Experiment Designs"):
        if st.session_state.cross_paper_ideas.strip() == "":
            st.error("Generate research gaps and ideas first.")

        else:
            with st.spinner("Designing experiments..."):
                experiment_plan = generate_experiment_plan(
                    st.session_state.cross_paper_ideas
                )
                st.session_state.experiment_plan = experiment_plan

            st.success("Experiment designs generated.")

    if st.session_state.experiment_plan:
        st.text_area(
            "Experiment Design Report",
            st.session_state.experiment_plan,
            height=600
        )


# -----------------------------
# Tab 5: Export
# -----------------------------

with tab5:
    st.markdown(
        """
<div class="section-card">
    <div class="section-title">📥 Export Final Research Report</div>
    <div class="section-subtitle">
        Download a complete text report containing the paper knowledge base, cross-paper ideas,
        and experiment design report.
    </div>
</div>
""",
        unsafe_allow_html=True
    )

    final_report = f"""
NEUROSCHOLAR-AI PRO FINAL REPORT

============================================================
1. PAPER KNOWLEDGE BASE
============================================================

{st.session_state.knowledge_base}

============================================================
2. CROSS-PAPER RESEARCH GAPS AND IDEAS
============================================================

{st.session_state.cross_paper_ideas}

============================================================
3. EXPERIMENT DESIGN REPORT
============================================================

{st.session_state.experiment_plan}
"""

    st.download_button(
        label="📥 Download Final Report",
        data=final_report,
        file_name="neuroscholar_final_report.txt",
        mime="text/plain"
    )

    st.text_area("Final Report Preview", final_report, height=600)

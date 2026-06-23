
import os
import time

import fitz
import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="NeuroScholar-AI Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100
DEMO_MAX_PDFS = 2
MAX_TEXT_FOR_SUMMARY = 12000
FALLBACK_PREVIEW_CHARS = 3000


st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(124,58,237,.25), transparent 35%),
        radial-gradient(circle at top right, rgba(14,165,233,.22), transparent 35%),
        linear-gradient(135deg, #07111f 0%, #0f172a 45%, #111827 100%);
    color: #e5e7eb;
}
[data-testid="stHeader"] { background: rgba(7,17,31,0); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15,23,42,.98), rgba(17,24,39,.98));
    border-right: 1px solid rgba(148,163,184,.18);
}
[data-testid="stSidebar"] * { color: #e5e7eb; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
.hero-card {
    padding: 2.2rem 2.4rem;
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(30,41,59,.92), rgba(15,23,42,.78)),
        linear-gradient(135deg, rgba(124,58,237,.35), rgba(14,165,233,.20));
    border: 1px solid rgba(148,163,184,.25);
    box-shadow: 0 30px 90px rgba(0,0,0,.38);
    margin-bottom: 1.25rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -.05em;
    margin: 0;
    line-height: 1.05;
    background: linear-gradient(90deg, #e0f2fe, #c4b5fd, #93c5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #cbd5e1;
    margin-top: .8rem;
    max-width: 980px;
    line-height: 1.65;
}
.badge-row { display:flex; flex-wrap:wrap; gap:.65rem; margin-top:1.2rem; }
.badge {
    padding: .42rem .78rem;
    border-radius: 999px;
    background: rgba(15,23,42,.78);
    border: 1px solid rgba(148,163,184,.25);
    color: #dbeafe;
    font-size: .88rem;
    font-weight: 700;
}
.metric-card, .section-card, .answer-box {
    border-radius: 22px;
    background: rgba(15,23,42,.72);
    border: 1px solid rgba(148,163,184,.18);
    box-shadow: 0 18px 45px rgba(0,0,0,.25);
}
.metric-card { padding: 1.15rem 1.2rem; min-height: 118px; }
.section-card { padding: 1.2rem 1.35rem; margin-bottom: 1rem; }
.answer-box { padding: 1.15rem 1.25rem; margin-top: 1rem; background: rgba(2,6,23,.64); border-color: rgba(96,165,250,.26); }
.metric-title { color:#94a3b8; font-size:.86rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em; margin-bottom:.45rem; }
.metric-value { color:#f8fafc; font-size:1.45rem; font-weight:900; margin-bottom:.25rem; }
.metric-text, .section-subtitle { color:#cbd5e1; font-size:.95rem; line-height:1.55; }
.section-title { font-size:1.35rem; font-weight:900; color:#f8fafc; margin-bottom:.25rem; }
.chip {
    display:inline-block;
    padding:.45rem .7rem;
    margin:.2rem;
    border-radius:999px;
    background:rgba(30,41,59,.85);
    border:1px solid rgba(148,163,184,.20);
    color:#bfdbfe;
    font-size:.86rem;
    font-weight:700;
}
.stTabs [data-baseweb="tab-list"] {
    gap:.55rem;
    background:rgba(15,23,42,.55);
    padding:.55rem;
    border-radius:20px;
    border:1px solid rgba(148,163,184,.15);
}
.stTabs [data-baseweb="tab"] { border-radius:16px; padding:.7rem 1rem; color:#cbd5e1; font-weight:800; }
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg, rgba(124,58,237,.75), rgba(14,165,233,.60));
    color:#fff !important;
}
textarea, input { border-radius: 16px !important; }
div.stButton > button, div.stDownloadButton > button {
    border-radius:16px;
    padding:.72rem 1.05rem;
    font-weight:900;
    border:1px solid rgba(147,197,253,.28);
    background:linear-gradient(135deg, #7c3aed, #0ea5e9);
    color:white;
    box-shadow:0 14px 30px rgba(14,165,233,.18);
}
#MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Source chunks for Q&A", 3, 10, DEFAULT_TOP_K)
    chunk_size = st.slider("Chunk size", 300, 1000, DEFAULT_CHUNK_SIZE, step=100)
    overlap = st.slider("Chunk overlap", 50, 250, DEFAULT_OVERLAP, step=50)
    st.markdown("---")
    st.caption("Demo mode: max 2 PDFs. Add paid quota for real production usage.")


st.markdown(
    """
<div class="hero-card">
    <div class="hero-title">NeuroScholar-AI Pro</div>
    <div class="hero-subtitle">
        Upload scientific papers and ask anything. The app extracts PDF text, builds a searchable
        knowledge base, retrieves source chunks, and helps generate summaries, research gaps,
        possible hypotheses, experiment ideas, and final reports.
    </div>
    <div class="badge-row">
        <span class="badge">📄 Multi-PDF Analysis</span>
        <span class="badge">🔎 Semantic Search</span>
        <span class="badge">🧠 Gemini RAG</span>
        <span class="badge">🔬 Research Gaps</span>
        <span class="badge">🧪 Experiment Planning</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-title">Pipeline</div><div class="metric-value">PDF → RAG → Report</div><div class="metric-text">Extract, chunk, embed, retrieve, reason, export.</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-title">Ask Mode</div><div class="metric-value">Any Prompt</div><div class="metric-text">Summary, tables, gaps, methods, limitations, experiments.</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-title">Safety</div><div class="metric-value">Cautious Science</div><div class="metric-text">Suggests possible ideas, not fake proven discoveries.</div></div>', unsafe_allow_html=True)

st.info("Demo version: upload up to 2 papers. For full literature review reports with 5–10 papers, contact the owner.")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY", None)
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


def get_gemini_client():
    api_key = get_api_key()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def is_model_failure(text):
    if not isinstance(text, str):
        return False

    t = text.lower()
    markers = [
        "all gemini models failed",
        "503 unavailable",
        "429 resource_exhausted",
        "resource_exhausted",
        "quota is exhausted",
        "quota exceeded",
        "free request limit",
        "temporarily overloaded",
        "try again after some time",
        "gemini api error",
        "api key is not configured",
        "error: gemini api key",
    ]
    return any(m in t for m in markers)


def generate_with_fallback(prompt):
    client = get_gemini_client()
    if client is None:
        return "Error: Gemini API key is not configured. Add GEMINI_API_KEY in Streamlit Cloud Secrets."

    model_name = "gemini-2.5-flash-lite"

    try:
        response = client.models.generate_content(model=model_name, contents=prompt)
        return response.text or "Gemini returned an empty response."
    except Exception as e:
        error_text = str(e)
        error_lower = error_text.lower()

        if "429" in error_text or "resource_exhausted" in error_lower or "quota" in error_lower:
            return "Gemini quota is exhausted right now. The API key is valid, but the free request limit has been reached. Try again later or contact the app owner for a full report."

        if "503" in error_text or "unavailable" in error_lower or "overloaded" in error_lower:
            return "Gemini is temporarily overloaded. Please try again after some time."

        return f"Gemini API error: {error_text}"


def extract_pdf_text(uploaded_file):
    try:
        file_bytes = uploaded_file.getvalue()
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""

        for page_number, page in enumerate(doc, start=1):
            page_text = page.get_text()
            text += f"\n\n--- Page {page_number} ---\n\n{page_text}"

        doc.close()
        return text
    except Exception as e:
        return f"PDF_EXTRACTION_ERROR: {e}"


def clean_text(text):
    text = text.replace("\n", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def shorten_text(text, max_chars=MAX_TEXT_FOR_SUMMARY):
    return text[:max_chars] if len(text) > max_chars else text


def split_into_chunks(text, paper_name, chunk_size_value=DEFAULT_CHUNK_SIZE, overlap_value=DEFAULT_OVERLAP):
    words = text.split()
    chunks = []

    if not words:
        return chunks

    if overlap_value >= chunk_size_value:
        overlap_value = DEFAULT_OVERLAP

    start = 0
    chunk_id = 1

    while start < len(words):
        end = start + chunk_size_value
        chunk_text = " ".join(words[start:end]).strip()

        if chunk_text:
            chunks.append({"paper_name": paper_name, "chunk_id": chunk_id, "text": chunk_text})

        chunk_id += 1
        next_start = end - overlap_value

        if next_start <= start:
            break

        start = next_start

    return chunks


def fallback_summary(cleaned_text, reason):
    preview = cleaned_text[:FALLBACK_PREVIEW_CHARS]
    return f"""
Gemini summary unavailable.

Reason:
{reason}

Fallback paper preview from extracted PDF text:
{preview}

Note:
The PDF text was extracted successfully. Searchable chunks were created, so the paper can still be used for retrieval after Gemini becomes available again.
"""


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
    if not chunks:
        return None
    model = load_embedding_model()
    return model.encode([chunk["text"] for chunk in chunks])


def retrieve_relevant_chunks(question, chunks, embeddings, top_k_value=DEFAULT_TOP_K):
    if embeddings is None or not chunks:
        return []

    model = load_embedding_model()
    question_embedding = model.encode([question])
    similarities = cosine_similarity(question_embedding, embeddings)[0]

    ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    results = []
    for index, score in ranked[:top_k_value]:
        results.append(
            {
                "paper_name": chunks[index]["paper_name"],
                "chunk_id": chunks[index]["chunk_id"],
                "text": chunks[index]["text"],
                "score": float(score),
            }
        )
    return results


def format_relevant_chunks(relevant_chunks):
    context = ""
    for chunk in relevant_chunks:
        context += f"\nSource: {chunk['paper_name']} | Chunk {chunk['chunk_id']} | Score {chunk['score']:.4f}\n"
        context += chunk["text"] + "\n\n"
    return context


def answer_any_prompt(question, knowledge_base, chunks, embeddings, top_k_value=DEFAULT_TOP_K):
    relevant_chunks = retrieve_relevant_chunks(question, chunks, embeddings, top_k_value)
    relevant_context = format_relevant_chunks(relevant_chunks)

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
- If the user asks for a table, create a table.
- If the user asks for research gaps, identify possible gaps.
- If the user asks for hypotheses, generate possible hypotheses.
- If the user asks for experiments, suggest possible experiments.
- If the user asks for limitations, list limitations.
- If the user asks for methods, explain methods.
- Ignore angry/casual tone and answer professionally.
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
            "Gemini cannot generate a fresh answer right now because quota is exhausted or the service is overloaded.\n\n"
            "Here is the available knowledge base preview:\n\n"
            f"{knowledge_base[:5000]}"
        )
        if len(knowledge_base) > 5000:
            fallback_text += "\n\n[Preview shortened. Full report is available in Export.]"
        return fallback_text, relevant_chunks

    return answer, relevant_chunks


def generate_cross_paper_ideas(knowledge_base):
    prompt = f"""
You are NeuroScholar-AI Pro, an advanced neuroscience research assistant.

Generate cross-paper research insights from this knowledge base.

Output:
1. Big Picture Summary
2. Cross-Paper Connections
3. Research Gaps
4. Possible Hypotheses
5. Possible Experiments
6. Novel Research Ideas with Novelty/Feasibility/Impact/Evidence/Priority
7. Final Recommendation: top 3 research directions

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
- Suggest safer alternatives for risky human drug experiments.
- Do not invent unsupported paper details.

Research ideas:
{research_ideas}
"""
    return generate_with_fallback(prompt)


def reset_state():
    st.session_state.paper_summaries = {}
    st.session_state.knowledge_base = ""
    st.session_state.chunks = []
    st.session_state.embeddings = None
    st.session_state.cross_paper_ideas = ""
    st.session_state.experiment_plan = ""


for key, default_value in {
    "paper_summaries": {},
    "knowledge_base": "",
    "chunks": [],
    "embeddings": None,
    "cross_paper_ideas": "",
    "experiment_plan": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📄 Upload Papers", "✨ Ask Anything", "🔬 Research Gaps & Ideas", "🧪 Experiments", "📥 Export"]
)


with tab1:
    st.markdown(
        """
<div class="section-card">
    <div class="section-title">📄 Upload Scientific Papers</div>
    <div class="section-subtitle">
        Upload PDF papers. If Gemini fails, the app still extracts text, creates chunks, builds embeddings,
        and keeps a fallback preview instead of saving ugly error text as the summary.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if st.button("🧹 Clear Current Session"):
        reset_state()
        st.success("Session cleared. Upload PDFs again.")

    uploaded_files = st.file_uploader("Upload PDF papers", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) > DEMO_MAX_PDFS:
        st.warning(f"Demo limit: only the first {DEMO_MAX_PDFS} PDFs will be processed.")
        uploaded_files = uploaded_files[:DEMO_MAX_PDFS]

    left, right = st.columns([1, 2])
    with left:
        build_clicked = st.button("🚀 Build Knowledge Base")
    with right:
        if uploaded_files:
            st.success(f"{len(uploaded_files)} PDF file(s) selected.")
        else:
            st.warning("Upload PDFs first, then build the knowledge base.")

    if build_clicked:
        if not uploaded_files:
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
                        progress.progress((i + 1) / len(uploaded_files))
                        continue

                    cleaned = clean_text(raw_text)

                    if not cleaned:
                        st.warning(f"No readable text found in {paper_name}. Skipping.")
                        progress.progress((i + 1) / len(uploaded_files))
                        continue

                    paper_chunks = split_into_chunks(
                        text=cleaned,
                        paper_name=paper_name,
                        chunk_size_value=chunk_size,
                        overlap_value=overlap,
                    )

                    if not paper_chunks:
                        st.warning(f"No chunks created for {paper_name}.")
                        progress.progress((i + 1) / len(uploaded_files))
                        continue

                    all_chunks.extend(paper_chunks)

                    summary = summarize_paper(paper_name, cleaned)

                    if is_model_failure(summary):
                        st.warning(f"{paper_name}: Gemini summary failed, but chunks were created successfully.")
                        summary = fallback_summary(cleaned, summary)

                    paper_summaries[paper_name] = summary

                    knowledge_base += f"""
============================================================
PAPER: {paper_name}
============================================================

{summary}

"""

                progress.progress((i + 1) / len(uploaded_files))

            if not all_chunks:
                st.error("No chunks were created. Check whether your PDFs contain selectable text.")
            else:
                with st.spinner("Creating embeddings for all paper chunks..."):
                    embeddings = build_embeddings(all_chunks)

                st.session_state.paper_summaries = paper_summaries
                st.session_state.knowledge_base = knowledge_base
                st.session_state.chunks = all_chunks
                st.session_state.embeddings = embeddings

                st.success("Knowledge base created successfully.")

    if st.session_state.knowledge_base:
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Papers</div><div class="metric-value">{len(st.session_state.paper_summaries)}</div><div class="metric-text">Analyzed documents</div></div>', unsafe_allow_html=True)
        with s2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Chunks</div><div class="metric-value">{len(st.session_state.chunks)}</div><div class="metric-text">Searchable source units</div></div>', unsafe_allow_html=True)
        with s3:
            kb_words = len(st.session_state.knowledge_base.split())
            st.markdown(f'<div class="metric-card"><div class="metric-title">Knowledge Base</div><div class="metric-value">{kb_words}</div><div class="metric-text">Approximate words generated</div></div>', unsafe_allow_html=True)

        st.subheader("Paper Knowledge Base Preview")
        st.text_area("Knowledge Base", st.session_state.knowledge_base, height=400)


with tab2:
    st.markdown(
        """
<div class="section-card">
    <div class="section-title">✨ Ask Anything About Uploaded Papers</div>
    <div class="section-subtitle">
        Ask naturally. Summary, table, gaps, hypotheses, methods, limitations, experiments,
        or simple explanation. The app uses the full paper knowledge base plus relevant chunks.
    </div>
    <span class="chip">what did the paper say?</span>
    <span class="chip">explain like I am 10</span>
    <span class="chip">make a comparison table</span>
    <span class="chip">find research gaps</span>
    <span class="chip">suggest experiments</span>
</div>
""",
        unsafe_allow_html=True,
    )

    question = st.text_area(
        "Enter any prompt",
        placeholder="Example: Explain the uploaded papers in simple English and give 5 key takeaways.",
        height=135,
    )

    if st.button("🧠 Generate Answer"):
        if not st.session_state.knowledge_base.strip():
            st.error("Build the knowledge base first in the Upload Papers tab.")
        elif not question.strip():
            st.error("Type a prompt first.")
        else:
            with st.spinner("Thinking across the full knowledge base and source chunks..."):
                answer, relevant_chunks = answer_any_prompt(
                    question=question,
                    knowledge_base=st.session_state.knowledge_base,
                    chunks=st.session_state.chunks,
                    embeddings=st.session_state.embeddings,
                    top_k_value=top_k,
                )

            st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
            st.subheader("AI Answer")
            st.write(answer)
            st.markdown("</div>", unsafe_allow_html=True)

            if relevant_chunks:
                st.subheader("Most Relevant Source Chunks")
                for chunk in relevant_chunks:
                    with st.expander(f"{chunk['paper_name']} | Chunk {chunk['chunk_id']} | Score {chunk['score']:.4f}"):
                        st.write(chunk["text"])


with tab3:
    st.markdown(
        '<div class="section-card"><div class="section-title">🔬 Cross-Paper Research Gaps and Ideas</div><div class="section-subtitle">Generate cautious research gaps, possible hypotheses, experiment directions, and final research recommendations.</div></div>',
        unsafe_allow_html=True,
    )

    if st.button("🔬 Generate Research Gaps and Ideas"):
        if not st.session_state.knowledge_base.strip():
            st.error("Build the knowledge base first.")
        else:
            with st.spinner("Generating cross-paper research insights..."):
                ideas = generate_cross_paper_ideas(st.session_state.knowledge_base)
                st.session_state.cross_paper_ideas = ideas

            if is_model_failure(ideas):
                st.warning("Gemini failed or quota is exhausted. Try later or use increased quota.")
            else:
                st.success("Research ideas generated.")

    if st.session_state.cross_paper_ideas:
        st.text_area("Cross-Paper Research Ideas", st.session_state.cross_paper_ideas, height=600)


with tab4:
    st.markdown(
        '<div class="section-card"><div class="section-title">🧪 Experiment Design Assistant</div><div class="section-subtitle">Convert research ideas into structured experiment plans with variables, groups, methods, limitations, ethics, and significance.</div></div>',
        unsafe_allow_html=True,
    )

    if st.button("🧪 Generate Experiment Designs"):
        if not st.session_state.cross_paper_ideas.strip():
            st.error("Generate research gaps and ideas first.")
        else:
            with st.spinner("Designing experiments..."):
                experiment_plan = generate_experiment_plan(st.session_state.cross_paper_ideas)
                st.session_state.experiment_plan = experiment_plan

            if is_model_failure(experiment_plan):
                st.warning("Gemini failed or quota is exhausted. Try later or use increased quota.")
            else:
                st.success("Experiment designs generated.")

    if st.session_state.experiment_plan:
        st.text_area("Experiment Design Report", st.session_state.experiment_plan, height=600)


with tab5:
    st.markdown(
        '<div class="section-card"><div class="section-title">📥 Export Final Research Report</div><div class="section-subtitle">Download the complete report containing the knowledge base, research ideas, and experiment design report.</div></div>',
        unsafe_allow_html=True,
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
        mime="text/plain",
    )

    st.text_area("Final Report Preview", final_report, height=600)

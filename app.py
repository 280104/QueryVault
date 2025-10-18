# app.py
import os
import uuid
import shutil
import re
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import html

#  CONFIG 
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"
INDEX_DIR = "index_store"
PDF_DIR = "uploaded_pdfs"
CHUNK_SIZE = 5
CHUNK_OVERLAP = 1

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

#       LOAD MODELS 
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource
def load_gen_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

emb_model = load_embedding_model()
gen_pipe = load_gen_pipeline()
EMBED_DIM = emb_model.get_sentence_embedding_dimension()
index_path = os.path.join(INDEX_DIR, "faiss.index")
meta_path = os.path.join(INDEX_DIR, "metadatas.pkl")

#         FAISS helpers 
def create_index():
    return faiss.IndexFlatIP(EMBED_DIM)

def save_index(index, metadatas):
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadatas, f)

def load_index():
    if os.path.exists(index_path) and os.path.exists(meta_path):
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            metadatas = pickle.load(f)
        return index, metadatas
    return None, []

index, metadatas = load_index()
if index is None:
    index = create_index()
    metadatas = []

#        UTILS 
def pdf_to_chunks(file_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    reader = PdfReader(file_path)
    chunks = []
    sentence_splitter = re.compile(r'(?<=[.!?])\s+')
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        sentences = sentence_splitter.split(text)
        if not sentences:
            continue
        start = 0
        while start < len(sentences):
            chunk_sentences = sentences[start:start + chunk_size]
            chunk_text = " ".join(chunk_sentences).strip()
            if chunk_text:
                chunks.append({"page": page_num + 1, "text": chunk_text})
            start += chunk_size - chunk_overlap
    # deduplicate exact text
    seen = set()
    unique_chunks = []
    for c in chunks:
        t = c["text"]
        if t not in seen:
            unique_chunks.append(c)
            seen.add(t)
    return unique_chunks

def add_pdf_to_index(file_bytes, filename):
    idname = f"{uuid.uuid4().hex}_{filename}"
    path = os.path.join(PDF_DIR, idname)
    with open(path, "wb") as f:
        f.write(file_bytes)

    chunks = pdf_to_chunks(path)
    if not chunks:
        st.warning(f"No text could be extracted from {filename}. Skipping.")
        return 0

    texts = [c["text"] for c in chunks]
    embeddings = emb_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    # ensure 2D array
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, axis=0)

    # check embedding dimension matches index
    if embeddings.shape[1] != index.d:
        st.error(f"Embedding dimension mismatch: index={index.d}, embeddings={embeddings.shape[1]}. "
                 f"Delete old index and re-upload PDFs.")
        return 0

    # normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    embeddings = embeddings / norms

    index.add(embeddings)

    start_idx = len(metadatas)
    for i, c in enumerate(chunks):
        metadatas.append({
            "id": start_idx + i,
            "filename": filename,
            "page": c["page"],
            "text": c["text"]
        })
    save_index(index, metadatas)
    return len(chunks)

def query_index(query, topk=5, fetch_multiplier=3):
    q_emb = emb_model.encode([query], convert_to_numpy=True)
    if q_emb.ndim == 1:
        q_emb = np.expand_dims(q_emb, axis=0)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    fetch_k = topk * fetch_multiplier
    D, I = index.search(q_emb, fetch_k)
    results = []
    seen_texts = set()
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadatas):
            continue
        meta = metadatas[idx]
        text = meta["text"].strip()
        if text in seen_texts:
            continue
        seen_texts.add(text)
        results.append({"score": float(score), "meta": meta})
        if len(results) >= topk:
            break
    return results

def synthesize_answer(question, contexts):
    prompt = (
        "You are a knowledgeable assistant. "
        "Provide a complete definition first, then examples if relevant. "
        "Summarize concisely (2-3 sentences). Include source filename and page numbers. "
        "Do not repeat sentences.\n\n"
    )
    for i, c in enumerate(contexts):
        prompt += f"Passage {i+1} (source: {c['meta']['filename']} page {c['meta']['page']}):\n{c['meta']['text']}\n\n"
    prompt += f"Question: {question}\nAnswer:"
    out = gen_pipe(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    safe_out = html.escape(out).replace("\n", "<br>")
    return safe_out

#              STREAMLIT UI 
st.set_page_config(page_title="QueryVault", layout="wide")

bg_color = "#f8f9fa"
card_color = "#ffffff"
accent_color = "#3b82f6"

st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    color: #111827;
    font-family: 'Inter', sans-serif;
}}
div.stButton > button {{
    background-color: {accent_color};
    color: white;
    border-radius: 0.5rem;
    padding: 0.5rem 1rem;
}}
.card {{
    background-color: {card_color};
    border-radius: 0.75rem;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    color: #111827;
    
}}
.footer {{
    text-align: center;
    font-size: 0.85rem;
    color: #6b7280;
    margin-top: 2rem;
}}
</style>
""", unsafe_allow_html=True)

st.title("QueryVault — Multi-Document PDF Search")

with st.expander("Upload PDFs"):
    uploaded = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        # safely clear old PDFs and index
        for filename in os.listdir(PDF_DIR):
            file_path = os.path.join(PDF_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

        # reset index and metadata
        index = create_index()
        metadatas = []

        for f in uploaded:
            add_pdf_to_index(f.read(), f.name)
        st.success(f"Indexed {len(uploaded)} files. Current index size: {len(metadatas)} chunks.")

st.markdown("---")
q = st.text_input("Ask a question about your uploaded PDFs", placeholder="Enter your question here")
top_k = st.slider("Number of retrieved passages to show", 1, 10, 5)

if st.button("Search") and q.strip():
    if len(metadatas) == 0:
        st.warning("No documents indexed yet. Upload PDFs first.")
    else:
        results = query_index(q, topk=top_k)
        answer = synthesize_answer(q, results)
        st.markdown(f"<div class='card'><h3>Answer</h3><p>{answer}</p></div>", unsafe_allow_html=True)
        st.markdown("<h4>Top passages:</h4>", unsafe_allow_html=True)
        for i, r in enumerate(results):
            passage_html = f"""
            <div class='card'>
                <strong>{i+1}. (score: {r['score']:.3f}) — {r['meta']['filename']} (page {r['meta']['page']})</strong>
                <p>{r['meta']['text'][:400]}{"..." if len(r['meta']['text'])>400 else ""}</p>
            </div>
            """
            st.markdown(passage_html, unsafe_allow_html=True)

st.markdown("<div class='footer'>Powered by Astrovex 🚀</div>", unsafe_allow_html=True)

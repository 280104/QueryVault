

# QueryVault

**Multi-Document Intelligent Search and Q&A Application**

## Overview

QueryVault is a local, privacy-first PDF search application powered by retrieval-augmented generation (RAG). It allows users to upload multiple PDF files, automatically index their contents, and query them using natural language. The system retrieves the most relevant passages and generates concise answers citing their sources.

## Key Features

* **Multi-PDF Uploads:** Handle multiple documents in a single workspace.
* **Vector-Based Retrieval:** Uses FAISS for efficient semantic similarity search.
* **Local Embedding & Generation:** Runs entirely on the user’s machine without external API calls.
* **Concise Answers:** Generates summarized responses with contextual grounding.
* **Simple Interface:** Built with Streamlit, designed for accessibility and ease of use.
* **Document Management:** Clear and re-index PDFs dynamically without manual cleanup.

## Technical Stack

| Component             | Technology                                 |
| --------------------- | ------------------------------------------ |
| Programming Language  | Python 3.10+                               |
| Frontend / UI         | Streamlit                                  |
| Vector Database       | FAISS                                      |
| Embedding Model       | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Text Generation Model | Google FLAN-T5 (Small)                     |
| Document Parsing      | PyPDF (`pypdf`)                            |

## How It Works

1. **Upload PDFs:** Each document is processed page by page.
2. **Chunking:** Text is divided into token-based chunks to maintain contextual integrity.
3. **Embedding:** Each chunk is converted into a numerical vector using a sentence transformer.
4. **Indexing:** The embeddings are stored in a FAISS index for efficient similarity search.
5. **Querying:** When a question is asked, the query is embedded, compared against the index, and the most relevant chunks are retrieved.
6. **Answer Synthesis:** Retrieved text passages are passed to a lightweight generative model to produce a coherent, cited answer.

## Installation

```bash
git clone https://github.com/<your-username>/QueryVault.git
cd QueryVault
python -m venv venv
venv\Scripts\activate   # on Windows
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
QueryVault/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Dependencies
├── index_store/          # FAISS index and metadata
├── uploaded_pdfs/        # Temporary uploaded PDFs
├── .gitignore
└── README.md
```

## Future Improvements

* Replace small FLAN-T5 with instruction-tuned open-source LLMs (e.g., Mistral-7B) for richer answers.
* Introduce hybrid retrieval (BM25 + embeddings).
* Add support for DOCX, TXT, and web-scraped documents.
* Build user authentication for persistent document workspaces.
* Implement a responsive design layer using custom Streamlit components.

## License

This project is released under the MIT License.

## Developed By
*ASTROVEX*

## For More Information
*Contact* -- saisricharankolli@gmail.com
*Linkedin* -- linkedin.com/in/charan-kolli/


# app.py
import os
import tempfile
import textwrap
from typing import List

import streamlit as st
from dotenv import load_dotenv

# LangChain "modern" imports
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory
# from langchain_classic.chains import ConversationChain

# ---------- Configuration ----------
load_dotenv()
PERSIST_DIR = "chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)

# UI: inject a modern light theme
def inject_custom_css():
    st.markdown(
        """
    <style>
      /* Body & fonts */
      .main { background-color: #f7f8fa; padding: 18px; }
      section[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #eef0f3; }
      h1 { font-size: 34px; color: #1f2937; font-weight:700; }
      /* Buttons */
      .stButton>button { background: linear-gradient(90deg,#4b7bec,#5a8ff2); color:#fff; border-radius: 10px; padding: 8px 14px; border: none; font-weight:600; }
      .stButton>button:hover { transform: translateY(-2px); }
      /* Cards / panels */
      .stAlert, .css-1d391kg { border-radius: 12px; }
      /* Chat bubbles */
      .stChatMessage { border-radius: 12px !important; padding: 12px !important; background: #fff !important; border: 1px solid #eee !important; box-shadow: 0 1px 3px rgba(16,24,40,0.03); }
      .stChatMessage[data-testid="chat-message-user"] { background: #eaf2ff !important; border-color: #d6e8ff !important; }
      /* Code blocks inside expanders */
      details { background: #fcfdff !important; border-radius: 10px; border: 1px solid #f0f2f6; padding: 10px; }
      /* Input */
      .stTextInput>div>input, .stTextArea>div>textarea { border-radius: 10px !important; border: 1px solid #e6e9ee !important; padding: 10px !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )


# Call early
st.set_page_config(page_title="Modern Enterprise RAG", layout="wide")
inject_custom_css()
st.markdown("<h1>📚 Modern Multi-PDF RAG — Portfolio Edition</h1>", unsafe_allow_html=True)
st.write("Upload PDFs (including PPT-exported PDFs). Works with text PDFs and image PDFs (OCR fallback).")

# ---------- Session state initialization ----------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

# ---------- Helpers: embeddings & vector store ----------
def get_embeddings():
    # Local Sentence-Transformers (fast & free)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_or_create_vector_store():
    embeddings = get_embeddings()
    # If any files exist in PERSIST_DIR assume existing DB
    if os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0:
        try:
            vs = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
            st.success("Loaded existing Chroma knowledge base.")
            return vs
        except Exception as e:
            st.warning(f"Failed loading existing DB ({e}). Creating a fresh DB.")
    # Create an "empty" Chroma (it will initialize when adding docs)
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

# ---------- Document loading with OCR fallback ----------
def load_pdf_with_fallback(path: str):
    """
    Try PyPDFLoader (text extraction). If text is empty or nearly empty,
    fall back to UnstructuredPDFLoader with OCR strategy.
    Returns list of Document objects.
    """
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        # Determine if docs contain meaningful text
        total_len = sum(len(d.page_content.strip()) for d in docs)
        if total_len > 50:  # heuristic: enough text
            return docs
        # else fall back to OCR loader
    except Exception:
        docs = []

    # OCR fallback - UnstructuredPDFLoader supports OCR strategy in many setups
    try:
        ocr_loader = UnstructuredPDFLoader(path, strategy="ocr")
        ocr_docs = ocr_loader.load()
        return ocr_docs
    except Exception as e:
        # Final fallback: return what we have (could be empty)
        st.warning(f"OCR fallback failed for {os.path.basename(path)}: {e}")
        return docs

# ---------- Text splitting ----------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# ---------- Indexing ----------
def process_and_index(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]):
    if not uploaded_files:
        st.error("No files provided.")
        return

    vs = st.session_state.vector_store or load_or_create_vector_store()
    all_chunks = []
    for uploaded in uploaded_files:
        # Save to a temp file for loaders
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            docs = load_pdf_with_fallback(tmp_path)
            # Attach metadata
            for d_idx, d in enumerate(docs):
                # Ensure metadata exists
                if not isinstance(d.metadata, dict):
                    d.metadata = {}
                d.metadata["source"] = uploaded.name
                # PyPDFLoader sometimes sets "page" or "page_number"
                if "page" not in d.metadata and "page_number" not in d.metadata:
                    # best-effort: attach chunk index as page fallback
                    d.metadata["page"] = d_idx + 1
            chunks = split_documents(docs)
            all_chunks.extend(chunks)

            # Track files indexed
            st.session_state.indexed_files.add(uploaded.name)
        finally:
            os.remove(tmp_path)

    if not all_chunks:
        st.warning("No text could be extracted from uploaded files (try different files or ensure OCR).")
        return

    vs.add_documents(all_chunks)
    vs.persist()
    st.session_state.vector_store = vs
    st.success(f"Indexed {len(all_chunks)} text chunks from {len(uploaded_files)} file(s).")

# ---------- Build LLM (Gemini) ----------
def get_or_create_llm():
    if st.session_state.llm:
        return st.session_state.llm
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    st.session_state.llm = llm
    return llm

# ---------- Retrieval + Prompting ----------
PROMPT_TMPL = PromptTemplate.from_template(textwrap.dedent("""
You are a helpful assistant specialized in answering questions about the provided document context.
Use ONLY the context provided. If the context doesn't contain the answer, respond:
"I don't have enough information in the documents to answer that."

Context:
{context}

Conversation history:
{history}

User question:
{question}

Answer succinctly and cite sources by file name and page number when possible.
"""))

def format_retrieved_context(docs, max_chars=2500):
    """
    Build a single context string from retrieved docs.
    We include source & page metadata and a short excerpt.
    """
    parts = []
    total = 0
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", d.metadata.get("page_number", "N/A"))
        excerpt = d.page_content.strip()
        if len(excerpt) > 800:
            excerpt = excerpt[:800].rsplit("\n", 1)[0] + " ..."

        chunk = f"Source: {src} | Page: {page}\n{excerpt}\n---"
        parts.append(chunk)
        total += len(chunk)
        if total > max_chars:
            break
    return "\n\n".join(parts)

def get_history_text():
    """Return conversation history as simple text (safe if no memory)."""
    try:
        mem = st.session_state.memory
        if not mem:
            return ""
        mem_vars = mem.load_memory_variables({})
        # memory key might be 'history' or similar; try several fields
        history = mem_vars.get("history") or mem_vars.get("chat_history") or ""
        if isinstance(history, list):
            # list of messages
            lines = []
            for m in history:
                # message object could be dict-like or Message object
                content = getattr(m, "content", None) or m.get("content", "") if isinstance(m, dict) else str(m)
                role = getattr(m, "type", None) or (m.get("role") if isinstance(m, dict) else "msg")
                lines.append(f"{role}: {content}")
            return "\n".join(lines)
        return str(history)
    except Exception:
        return ""

def answer_query(question: str):
    """
    1) retrieve top-k docs
    2) build context string
    3) format prompt with history
    4) call LLM and return answer + used docs
    """
    vs = st.session_state.vector_store
    if not vs:
        return "No documents indexed. Please upload and index PDFs.", []

    retriever = vs.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(question)
    context = format_retrieved_context(docs)
    history = get_history_text()

    prompt = PROMPT_TMPL.format(context=context, history=history, question=question)
    llm = get_or_create_llm()

    # Call LLM. Many wrappers support __call__ returning a string.
    try:
        # prefer plain call
        answer = llm(prompt)
        # Some wrappers return object; try to extract text if needed
        if hasattr(answer, "content"):
            # wrapper returns an object with content attribute
            answer_text = answer.content
        elif isinstance(answer, list) and len(answer) > 0:
            answer_text = str(answer[0])
        else:
            answer_text = str(answer)
    except Exception as e:
        # Fallback: try .generate or .create if available
        try:
            gen = llm.generate([prompt])
            # extract
            if hasattr(gen, "generations"):
                answer_text = gen.generations[0][0].text
            else:
                answer_text = str(gen)
        except Exception as e2:
            answer_text = f"LLM call failed: {e}; fallback error: {e2}"

    # Save to memory
    try:
        st.session_state.memory.chat_memory.add_user_message(question)
        st.session_state.memory.chat_memory.add_ai_message(answer_text)
    except Exception:
        # If memory API differs, silently continue
        pass

    return answer_text, docs

# ---------- Initialize vector store on start ----------
if st.session_state.vector_store is None:
    st.session_state.vector_store = load_or_create_vector_store()

# ---------- UI layout ----------
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.header("1 • Document Management")
    uploaded = st.file_uploader("Upload PDF(s) — supports text PDFs and image/PPT PDFs (OCR)", type=["pdf"], accept_multiple_files=True)
    idx_btn = st.button("Index uploaded PDFs")

    if idx_btn:
        if uploaded:
            with st.spinner("Indexing files..."):
                process_and_index(uploaded)
                st.success("Done indexing. You can now ask questions.")
        else:
            st.error("Please select at least one PDF to index.")

    st.markdown("---")
    st.subheader("Indexed files")
    if st.session_state.indexed_files:
        for f in sorted(st.session_state.indexed_files):
            st.markdown(f"- {f}")
    else:
        st.info("No files indexed yet.")

    st.markdown("---")
    st.subheader("Settings")
    st.write("Model & retrieval settings")
    temp = st.slider("LLM temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    # update llm temperature live (best-effort)
    if st.session_state.llm:
        try:
            st.session_state.llm.temperature = float(temp)
        except Exception:
            pass

with col_right:
    st.header("2 • Chat")
    if not st.session_state.vector_store or len(os.listdir(PERSIST_DIR)) == 0:
        st.info("No index available. Upload PDFs in the left panel and click Index.")
    else:
        # Show chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input
        user_q = st.chat_input("Ask anything about the indexed documents...")
        if user_q:
            # display user message
            st.session_state.messages.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            with st.chat_message("assistant"):
                with st.spinner("Searching documents & generating answer..."):
                    answer, used_docs = answer_query(user_q)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    # show sources
                    if used_docs:
                        st.markdown("---")
                        st.subheader("📚 Sources used (expand to view excerpts)")
                        # Deduplicate by (source, page)
                        seen = set()
                        for d in used_docs:
                            src = d.metadata.get("source", "unknown")
                            page = d.metadata.get("page", d.metadata.get("page_number", "N/A"))
                            key = (src, str(page))
                            if key in seen:
                                continue
                            seen.add(key)
                            excerpt = d.page_content.strip()
                            # show small excerpt with expander
                            with st.expander(f"{src} — page {page}"):
                                # Cap the display length
                                if len(excerpt) > 2000:
                                    st.write(excerpt[:2000] + "... (truncated)")
                                else:
                                    st.write(excerpt)

# ---------- Footer / tips ----------
st.markdown("---")
st.markdown(
    """
**Tips**
- If a PDF is exported from PowerPoint as images (text not selectable), indexing will automatically attempt OCR.
- For best results upload original PDF/PPT or text-based export.
- To persist across sessions, keep the `chroma_db` folder.
"""
)

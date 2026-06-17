import os
import streamlit as st
import cohere
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ------------------ PAGE ------------------
st.write("GROQ_API_KEY present:", bool(os.getenv("GROQ_API_KEY")))

st.set_page_config(page_title="Mini RAG", layout="centered")
st.title("Mini RAG Application")

# ------------------ KEYS ------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not set")
    st.stop()

# ------------------ SESSION INIT ------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "has_data" not in st.session_state:
    st.session_state.has_data = False

if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )
    
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

def rerank_docs(query, docs, top_n=3):
    texts = [doc.page_content for doc in docs]

    rerank_results = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=texts,
        top_n=top_n,
    )

    reranked_docs = [docs[r.index] for r in rerank_results.results]
    return reranked_docs

# ------------------ INGEST ------------------

st.subheader("Ingest Document")

text = st.text_area("Paste text to ingest")

if st.button("Ingest"):
    if not text.strip():
        st.warning("Please paste some text.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = splitter.create_documents([text])

    st.session_state.vectorstore = Chroma.from_documents(
        docs,
        st.session_state.embeddings,
    )

    st.session_state.has_data = True
    st.success(f"Ingested {len(docs)} chunks")

# ------------------ LLM ------------------

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,   
    model="llama-3.1-8b-instant",
    temperature=0,
)

# ------------------ QUERY ------------------

st.subheader("Ask a Question")

question = st.text_input("Your question")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    if not st.session_state.has_data:
        st.warning("Please ingest a document first.")
        st.stop()

    retrieved_docs = st.session_state.vectorstore.similarity_search(question, k=8)
    docs = rerank_docs(question, retrieved_docs, top_n=3)


    if not docs:
        st.warning("No relevant context found.")
        st.stop()

    context = "\n\n".join(
        [f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)]
    )

    if not context.strip():
        st.warning("Empty context. Cannot answer.")
        st.stop()

    prompt = f"""
Use ONLY the context below to answer.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""

    try:
        response = llm.invoke(prompt)

        st.markdown("### Answer")
        st.write(response.content)

        st.markdown("### Sources")
        for i, doc in enumerate(docs):
            st.markdown(f"[{i+1}] {doc.page_content[:200]}...")

    except Exception as e:
        st.error("LLM failed.")
        st.exception(e)

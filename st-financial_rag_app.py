import streamlit as st
import faiss
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# Load and Extract Text from PDFs
# -------------------------------

pdf_path = "/content/sample_data/BMW_Finance_NV_Annual_Report_2023.pdf"
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# -------------------------------
# Chunk the Text
# -------------------------------
def chunk_text(text, chunk_size=600, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

# -------------------------------
# Embed Chunks
# -------------------------------
def embed_text(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings, model

# -------------------------------
# Store Embeddings in FAISS
# -------------------------------
def store_embeddings(embeddings, index_path="financial_statements.index"):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    faiss.write_index(index, index_path)
    return index

# -------------------------------
# Implement BM25
# -------------------------------
def bm25_index(chunks):
    tokenized_chunks = [chunk.split() for chunk in chunks]
    return BM25Okapi(tokenized_chunks), tokenized_chunks

# -------------------------------
# Multi-Stage Retrieval
# -------------------------------

def multi_stage_retrieval(query, chunks, bm25, tokenized_chunks, index_path="financial_statements.index", model_name="sentence-transformers/all-MiniLM-L6-v2", k=5):
    # 1. Initial Retrieval (BM25)
    bm25_scores = bm25.get_scores(query.split())  
    bm25_top_indices = np.argsort(bm25_scores)[-k:][::-1]  # Top-k BM25 results

    # 2. Re-ranking (Embeddings)
    index = faiss.read_index(index_path)
    model = SentenceTransformer(model_name)

    # Embed query
    query_embedding = model.encode([query])

    # Select embeddings of top-k BM25 chunks
    top_k_embeddings = np.array([model.encode(chunks[i]) for i in bm25_top_indices])

    # Calculate similarity scores between query and top-k chunks
    distances, indices = index.search(np.array(query_embedding).reshape(1, -1), k)
    similarity_scores = -distances.flatten()  # Higher score = more similar

    # Re-rank BM25 results based on similarity scores
    ranked_indices = [bm25_top_indices[i] for i in np.argsort(similarity_scores)[::-1]]

    # Combine scores for confidence (optional)
    combined_scores = similarity_scores  # Use embedding similarity for confidence

    return ranked_indices, np.max(combined_scores)  # Return ranked indices and confidence


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Financial RAG Assistant", layout="wide")
st.title("📊 Financial Question Answering System")

# Upload PDF
pdf_file = st.file_uploader("Upload Financial Report (PDF)", type=["pdf"])

if pdf_file:
    st.success("📄 File uploaded successfully! Processing...")

    # Save uploaded file
    with open("uploaded_financial_report.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())

    # Extract and Process
    raw_text = extract_text_from_pdf("uploaded_financial_report.pdf")
    chunks = chunk_text(raw_text)
    embeddings, model = embed_text(chunks)
    store_embeddings(embeddings)
    bm25, tokenized_chunks = bm25_index(chunks)

    st.success("✅ Document indexed! Ask your question below.")

    # Query Input
    query = st.text_input("🔍 Ask a financial question:")

    if query:
           # Call multi_stage_retrieval
        ranked_indices, confidence_score = multi_stage_retrieval(query, chunks, bm25, tokenized_chunks)  
        top_chunk = chunks[ranked_indices[0]]

        # Display Answer
        st.subheader("📢 Answer:")
        st.write(top_chunk)

        # Confidence Score
        st.progress(float(confidence_score))
        st.caption(f"Confidence Score: {round(confidence_score * 100, 2)}%")

import re
import streamlit as st

# -------------------------------
# Query Validation Function
# -------------------------------
def validate_query(query):
    # Financial Keywords List (expand as needed)
    financial_keywords = [
        "revenue", "profit", "net income", "cash flow", "earnings", "assets",
        "liabilities", "equity", "debt", "dividends", "financial report", "expenses"
    ]

    # Check if query contains at least one financial keyword
    if not any(keyword in query.lower() for keyword in financial_keywords):
        return "❌ Invalid: Your question does not seem to be financial-related. Please ask about company finances."

    # Block harmful queries
    forbidden_patterns = [
        r"hack", r"password", r"exploit", r"illegal", r"fraud", r"scam"
    ]
    if any(re.search(pattern, query.lower()) for pattern in forbidden_patterns):
        return "🚨 Security Alert: This type of question is not allowed."

    # Check if the query is too vague
    if len(query.split()) < 3:
        return "⚠️ Too vague: Please provide more details in your question."

    return "✅ Valid"

# -------------------------------
# Streamlit UI with Guardrail
# -------------------------------
st.title("📊 Financial Question Answering System")

query = st.text_input("🔍 Ask a financial question:")

if query:
    validation_result = validate_query(query)

    if validation_result == "✅ Valid":
        st.success("✅ Your question is valid! Retrieving answer...")
        # Call retrieval function here
    else:
        st.error(validation_result)

# Predefined test cases
test_questions = [
    "What was the net income of the company last year?",
    "How will the company's revenue change next year?",
    "What is the capital of France?"
]

st.subheader("🛠 Testing & Validation")

def multi_stage_retrieval(query, chunks, bm25, tokenized_chunks, index_path="financial_statements.index", 
                         model_name="sentence-transformers/all-mpnet-base-v2", k=5): # Use the same model as in embed_text
    # 1. Initial Retrieval (BM25)
    bm25_scores = bm25.get_scores(query.split())  
    bm25_top_indices = np.argsort(bm25_scores)[-k:][::-1]  # Top-k BM25 results

    # 2. Re-ranking (Embeddings)
    index = faiss.read_index(index_path)
    model = SentenceTransformer(model_name) # Load the model here

    # Embed query
    query_embedding = model.encode([query])

    # Select embeddings of top-k BM25 chunks
    top_k_embeddings = np.array([model.encode(chunks[i]) for i in bm25_top_indices])

    # Calculate similarity scores between query and top-k chunks
    # Ensure query_embedding has the correct dimensionality
    query_embedding = query_embedding.reshape(1, -1)  
    distances, indices = index.search(query_embedding, k)
    
    similarity_scores = -distances.flatten()  # Higher score = more similar

    # Re-rank BM25 results based on similarity scores
    ranked_indices = [bm25_top_indices[i] for i in np.argsort(similarity_scores)[::-1]]

    # Combine scores for confidence (optional)
    combined_scores = similarity_scores  # Use embedding similarity for confidence

    return ranked_indices, np.max(combined_scores)  # Return ranked indices and confidence

for query in test_questions:
    st.write(f"**📝 Test Query:** {query}")

    validation_result = validate_query(query)

    if validation_result == "✅ Valid":
        ranked_indices, confidence_score = multi_stage_retrieval(query, chunks, bm25, tokenized_chunks)
        top_chunk = chunks[ranked_indices[0]]

        st.success(f"✅ Retrieved Answer: {top_chunk}")
        scaler = MinMaxScaler()
        confidence_score_scaled = scaler.fit_transform(confidence_score.reshape(-1, 1)).flatten()[0]  # Scale to 0-1
        st.progress(float(confidence_score_scaled))  # Use scaled score for progress bar
        st.caption(f"Confidence Score: {round(confidence_score_scaled * 100, 2)}%")
    else:
        st.error(validation_result)

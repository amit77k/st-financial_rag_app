import streamlit as st
import numpy as np
import pdfplumber

# -------------------------------
# Load and Extract Text from PDFs
# -------------------------------
pdf_path = "/BMW_Finance_NV_Annual_Report_2023.pdf"
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def validate_query(query):
    # Financial Keywords List (expand as needed)
    financial_keywords = [
        "revenue", "profit", "net income", "cash flow", "earnings", "assets",
        "liabilities", "equity", "debt", "dividends", "financial report", "expenses"
    ]

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
# Hybrid Retrieval (BM25 + Dense)
# -------------------------------
def hybrid_retrieval(query, chunks, bm25, tokenized_chunks, index_path="financial_statements.index", model_name="sentence-transformers/all-MiniLM-L6-v2", k=5):
    index = faiss.read_index(index_path)
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])
    
    distances, faiss_indices = index.search(np.array(query_embedding), k)
    
    bm25_scores = bm25.get_scores(query.split())
    bm25_top_indices = np.argsort(bm25_scores)[-k:][::-1]

    scaler = MinMaxScaler()
    dense_scores = -distances.flatten()
    bm25_scores = np.array([bm25_scores[i] for i in bm25_top_indices])

    combined_scores = np.concatenate([dense_scores, bm25_scores])
    combined_scores = scaler.fit_transform(combined_scores.reshape(-1, 1)).flatten()

    merged_indices = list(faiss_indices.flatten()) + list(bm25_top_indices)
    ranked_indices = [idx for _, idx in sorted(zip(combined_scores, merged_indices), reverse=True)][:k]

    return ranked_indices, np.max(combined_scores)

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
    raw_text = extract_text_from_pdf("BMW_Finance_NV_Annual_Report_2023.pdf")
    chunks = chunk_text(raw_text)
    embeddings, model = embed_text(chunks)
    store_embeddings(embeddings)
    bm25, tokenized_chunks = bm25_index(chunks)

    st.success("✅ Document indexed! Ask your question below.")

    # Query Input
    query = st.text_input("🔍 Ask a financial question:")

    if query:
        ranked_indices, confidence_score = hybrid_retrieval(query, chunks, bm25, tokenized_chunks)
        top_chunk = chunks[ranked_indices[0]]

        # Display Answer
        st.subheader("📢 Answer:")
        st.write(top_chunk)
        
        # Confidence Score
        st.progress(float(confidence_score))
        st.caption(f"Confidence Score: {round(confidence_score * 100, 2)}%")

# Predefined test cases
test_questions = [
    "What was the net income of the company last year?",
    "How will the company's revenue change next year?",
    "What is the capital of France?"
]

st.subheader("🛠 Testing & Validation")

for query in test_questions:
    st.write(f"**📝 Test Query:** {query}")
    
    validation_result = validate_query(query)

    if validation_result == "✅ Valid":
        ranked_indices, confidence_score = hybrid_retrieval(query, chunks, bm25, tokenized_chunks)
        top_chunk = chunks[ranked_indices[0]]

        st.success(f"✅ Retrieved Answer: {top_chunk}")
        st.progress(float(confidence_score))
        st.caption(f"Confidence Score: {round(confidence_score * 100, 2)}%")
    else:
        st.error(validation_result)


import fitz
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =========================
# 1️⃣ PDF LOADING
# =========================
def load_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# =========================
# 2️⃣ SMART CHUNKING
# =========================
def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# =========================
# 3️⃣ LOAD MODELS
# =========================
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


# =========================
# 4️⃣ BUILD VECTOR INDEX
# =========================
pdf_path = r"C:\Users\UJWALA\Downloads\Abhiram_resume.pdf"

raw_text = load_pdf(pdf_path)
documents = chunk_text(raw_text)

doc_embeddings = embedder.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype("float32")

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)


# =========================
# 5️⃣ QUERY LOOP
# =========================
while True:
    query = input("User: ")
    if query.lower() == "exit":
        break

    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    k = 3
    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = [documents[i] for i in indices[0]]
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
Answer strictly using the context below.
If answer not found, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm.generate(**inputs, max_new_tokens=200)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nAnswer:\n", answer, "\n")

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from transformers import pipeline

DATA_PATH = r"C:\Users\UJWALA\Downloads\Abhiram_resume.pdf"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3
#loading the document 
def load_document(path):
    return PyPDFLoader(path).load()
#splitting the texts
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)
#creating embeddings
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(chunks, embeddings)
#prompt template
prompt = PromptTemplate.from_template(
    """
Answer the question strictly using the context below.
If the answer is not present, say:
"I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
)
#loading the llm
def load_llm():
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=200
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)

def main():
    print("🔹 Loading document...")
    docs = load_document(DATA_PATH)

    print("🔹 Splitting document...")
    chunks = split_documents(docs)

    print("🔹 Creating vector store...")
    vectorstore = create_vectorstore(chunks)

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = load_llm()

    rag_chain = (
        {
            "context": lambda q: "\n\n".join(
                d.page_content for d in retriever.invoke(q)
            ),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | RunnableLambda(lambda x: x.strip())
    )

    print("\n✅ RAG system ready. ...Ask questions (type 'exit' to quit)\n")

    while True:
        query = input("User: ")
        if query.lower() == "exit":
            break

        print("🤖 Answering...")
        answer = rag_chain.invoke(query)
        print("\nAnswer:\n", answer, "\n")

if __name__ == "__main__":

    main()

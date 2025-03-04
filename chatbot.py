import os
import xml.etree.ElementTree as ET
import numpy as np
import faiss  # Ensure faiss-cpu is installed

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS  # Updated Import
from langchain_community.llms import HuggingFacePipeline  # Updated Import
from langchain_community.embeddings import HuggingFaceEmbeddings  # Fixed Embeddings
from langchain_community.docstore.in_memory import InMemoryDocstore  # Updated Import
from langchain.schema import Document as LangChainDocument
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# ----------------- LOCAL MODEL SETUP -----------------
MODEL_NAME = "google/flan-t5-small"
print("🚀 Loading the language model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
local_llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Hugging Face Embeddings (Optimized)
print("🧠 Loading the embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ----------------- XML DATA LOADER -----------------
def load_xml_data(folder_path="xml_files/"):
    """Load and parse XML files into structured format."""
    print(f"📂 Loading XML data from: {folder_path}")
    documents = []

    if not os.path.exists(folder_path):
        print(f"⚠️ Warning: Folder '{folder_path}' does not exist!")
        return []

    for filename in os.listdir(folder_path):
        if filename.endswith(".xml"):
            file_path = os.path.join(folder_path, filename)
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()

                # Extract structured content
                questions = []
                answers = []
                for elem in root.findall("question"):
                    questions.append(elem.text.strip() if elem.text else "")
                for elem in root.findall("answer"):
                    answers.append(elem.text.strip() if elem.text else "")

                # Combine into structured documents
                for q, a in zip(questions, answers):
                    text_content = f"Q: {q}\nA: {a}"
                    documents.append(LangChainDocument(page_content=text_content, metadata={"source": filename}))

                print(f"✅ Loaded {len(questions)} questions and {len(answers)} answers from {filename}")

            except Exception as e:
                print(f"❌ Error parsing {filename}: {e}")

    print(f"✅ Total Documents Loaded: {len(documents)}")
    return documents

# ----------------- FAISS VECTOR STORE -----------------
def build_vector_db():
    """Build FAISS vector store from structured XML QA data."""
    print("🧠 Creating FAISS Vector Store...")
    xml_docs = load_xml_data()

    if not xml_docs:
        print("⚠️ No documents found! Cannot build FAISS database.")
        return None

    # Optimized Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = text_splitter.split_documents(xml_docs)

    # Convert text content from documents and encode
    texts = [doc.page_content for doc in split_docs]
    embeddings = embedding_model.embed_documents(texts)

    # Create FAISS index
    dimension = len(embeddings[0])
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings, dtype=np.float32))

    # Store in FAISS vector database
    index_to_docstore_id = {i: str(i) for i in range(len(split_docs))}
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(split_docs)})

    vector_db = FAISS(
        embedding_function=embedding_model,
        index=faiss_index,
        docstore=docstore,  # ✅ Fix docstore linkage
        index_to_docstore_id=index_to_docstore_id  # ✅ Ensure correct mapping
    )

    print("✅ FAISS Vector Store Created Successfully!")
    return vector_db

# ----------------- MAIN EXECUTION -----------------
if __name__ == "__main__":
    vector_db = build_vector_db()
    
    if vector_db:
        retriever = vector_db.as_retriever(search_kwargs={"k": 5})  # Fetch top 5 results
        print("🤖 Chatbot Ready! You can now ask questions. Type 'exit' to quit.")

        while True:
            query = input("\n📝 Ask a question: ")
            if query.lower() == "exit":
                print("👋 Exiting chatbot.")
                break

            results = retriever.invoke(query)

            # Remove duplicate answers
            unique_answers = set()
            formatted_answers = []

            print("\n🔍 Retrieved Unique Answers:")
            for doc in results[:3]:  # Show top 3 unique answers
                if doc.page_content not in unique_answers:
                    unique_answers.add(doc.page_content)
                    formatted_answers.append(f"💡 {doc.page_content}\n")

            if formatted_answers:
                print("\n".join(formatted_answers))
            else:
                print("❌ No relevant answer found.")

    else:
        print("❌ Error: Chatbot initialization failed due to missing data.")

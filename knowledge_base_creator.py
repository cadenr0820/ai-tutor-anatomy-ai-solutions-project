import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
# The name of the file with your academic content
KB_FILE = "my_kb_data.txt" 
# The folder where the searchable database will be saved
CHROMA_PATH = "chroma_db" 
# A free, small, and fast model to create the embeddings (vectors)
MODEL_NAME = "all-MiniLM-L6-v2" 
# The size of each text chunk (in characters)
CHUNK_SIZE = 400 
# How many characters overlap between chunks to ensure context isn't lost
CHUNK_OVERLAP = 80 

def create_knowledge_base():
    """Reads text, chunks it, embeds it, and saves the vector store."""
    print(f"--- Starting Knowledge Base Creation ---")
    
    # 1. Load the Text Data
    with open(KB_FILE, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    print(f"Loaded {len(raw_text)} characters from {KB_FILE}")

    # 2. Chunk the Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    print(f"Split text into {len(chunks)} chunks.")

    # 3. Create Embeddings and the Vector Store (ChromaDB)
    print("Loading Sentence Transformer...")
    # This downloads the embedding model (all-MiniLM-L6-v2) for the first time
    model = SentenceTransformer(MODEL_NAME)
    
    # Custom function to generate embeddings using the loaded model
    def get_embeddings(texts):
        # We convert the NumPy array output to a list for ChromaDB
        return model.encode(texts, convert_to_numpy=True).tolist()

    print("Creating ChromaDB vector store...")
    # ChromaDB stores the text chunks and their corresponding vectors
    Chroma.from_texts(
        texts=chunks,
        embedding=lambda x: get_embeddings(x), # Pass the custom embedding function
        persist_directory=CHROMA_PATH # Tell it to save the database to the folder
    )
    
    print("✅ Knowledge Base successfully created and saved!")

if __name__ == "__main__":
    create_knowledge_base()
import streamlit as st
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import time # Used for C5 testing (response time)

# --- CONFIG ---
CHROMA_PATH = "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"
# Using a local model pipeline that performs text generation/summarization for the answer
GENERATION_MODEL_NAME = "facebook/bart-large-cnn" 

# This defines the required response structure for C2 (Out-of-Scope Redirection)
OUT_OF_SCOPE_MESSAGE = "I can only assist with topics found in my current study material. Please try a different question related to my subject scope."


@st.cache_resource
def load_resources():
    """Load the models and vector store only once for performance (C5)."""
    
    # 1. Load the Embedding Model (same one used to create the KB)
    embedding_model = SentenceTransformer(MODEL_NAME)
    
    # Function to use the loaded model
    def get_embeddings(texts):
        return embedding_model.encode(texts, convert_to_numpy=True).tolist()
    
    # 2. Load the Vector Store (Your Knowledge Base)
    try:
        vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=lambda x: get_embeddings(x) 
        )
    except Exception as e:
        # Handle case where chroma_db wasn't created or loaded correctly
        st.error(f"Error loading knowledge base. Did you run knowledge_base_creator.py? Error: {e}")
        return None, None 
    
    # 3. Load the Response Generator (a fast text summarization pipeline)
    generator = pipeline("summarization", model=GENERATION_MODEL_NAME)
    
    return vector_store, generator

# --- MAIN APP LOGIC ---

# Load resources (only runs once thanks to the cache decorator)
vector_store, generator = load_resources()

# Check if resources loaded successfully
if vector_store is None:
    st.stop() # Stop the Streamlit app if there was a loading error

st.title("🧠 AI Tutor Prototype")
st.caption("Ask me about the specific topic I was trained on!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def generate_rag_response(question, context, generator):
    """Generates the response using RAG and applies C2 logic."""
    
    context_text = "\n\n".join([doc.page_content for doc in context])
    
    # --- C2 Logic: Check for Sufficient Context ---
    # We assume if the retrieved context is short or not relevant enough, it's out of scope.
    # We will use a simple length check, but for C2, a relevance score check would be better.
    # For this prototype, if the retrieved context is very small, we use the redirection.
    if len(context_text) < 100:
         return OUT_OF_SCOPE_MESSAGE, [] # Return redirection message and no context

    # --- C1 Logic: Generate Answer from Context ---
    # The prompt explicitly tells the generator to use ONLY the context.
    prompt = f"""
    You are an expert high school tutor. Answer the user's question ONLY using the following context. 
    Do not use any external knowledge. Provide a factual and comprehensive answer.
    
    CONTEXT:
    {context_text}

    QUESTION:
    {question}

    TUTOR RESPONSE:
    """
    
    # Generate the response using the local model
    # We generate a summary that answers the question based on the strict prompt
    response = generator(prompt, max_length=512, min_length=128, do_sample=False)[0]['summary_text']
        
    return response, context


# Handle user input
if prompt := st.chat_input("Ask a question..."):
    # Store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- C5 Logic: Start Timer ---
    start_time = time.time()
    
    with st.spinner("Thinking..."):
        # 1. Retrieve Context (Search Phase)
        retrieved_context = vector_store.similarity_search(prompt, k=3) 
        
        # 2. Generate Final Response
        full_response, context_used = generate_rag_response(prompt, retrieved_context, generator)

    # --- C5 Logic: Calculate Response Time and Store ---
    response_time = time.time() - start_time
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Display the final output
    with st.chat_message("assistant"):
        st.markdown(full_response)
        st.caption(f"Response Time: {response_time:.2f} seconds (for C5 testing)")
        
        # Display sources only if an actual answer was given (for C1 testing)
        if context_used:
            st.caption("--- Sources Used (For C1 Testing) ---")
            for i, doc in enumerate(context_used):
                 st.code(f"Chunk {i+1}: {doc.page_content[:100]}...")
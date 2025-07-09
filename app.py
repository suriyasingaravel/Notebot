import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI
import chromadb
from chromadb.config import Settings
import time

# Initialize OpenAI client
api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=api_key)

# Function to load and chunk the knowledge base
@st.cache_data
def load_knowledge_base(uploaded_file):
    knowledge_base = ""
    with fitz.open(stream=uploaded_file.read()) as doc:
        for page in doc:
            knowledge_base += page.get_text()
    return knowledge_base

def fixed_word_chunking(text, chunk_size):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i + chunk_size]
        chunk = " ".join(chunk)
        chunks.append(chunk)
    return chunks

def get_embedding(word):
    response = client.embeddings.create(
        input=word,
        model="text-embedding-3-small"
    )
    result = response.data[0].embedding
    return result

# ChromaDB setup
@st.cache_resource
def setup_chroma_db(knowledge_chunks):
    chroma_client = chromadb.Client(Settings(persist_directory='./chroma_db'))
    collection = chroma_client.get_or_create_collection(name="my_collection")
    for i, chunk in enumerate(knowledge_chunks):
        collection.add(
            ids=[f"chunk-{i+1}"],
            documents=[chunk],
            embeddings=[get_embedding(chunk)]
        )
    return collection

# Function to get response from ChromaDB
def get_top_chunks_from_query(query, collection):
    query_embedding = get_embedding(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=2)
    return results['documents'][0]

def get_chatbot_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Streamlit UI
def main():
    st.title("AI Knowledge Retrieval Chatbot")
    st.markdown("Welcome to the AI-powered chatbot that answers your questions based on the provided knowledge base.")
    
    col1, col2 = st.columns([1, 2])  # Creating two columns for the layout

    with col1:
        # File upload
        uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])
        
        if uploaded_file is not None:
            # Show the file details
            st.write(f"**File Name**: {uploaded_file.name}")
            
            # Check if the knowledge is already cached
            if "knowledge_base" not in st.session_state:
                st.session_state['knowledge_base'] = load_knowledge_base(uploaded_file)
                st.session_state['knowledge_chunks'] = fixed_word_chunking(st.session_state['knowledge_base'], 30)
                st.session_state['collection'] = setup_chroma_db(st.session_state['knowledge_chunks'])
                st.write("Processing and indexing the document...")

    with col2:
        # Chatbot interface after file upload
        if uploaded_file is not None:
            # Initialize session state for conversation history
            if "conversation_history" not in st.session_state:
                st.session_state.conversation_history = []

            # Create a scrollable area for conversation history
            chat_history = ""
            for message in st.session_state.conversation_history:
                if message.startswith("User:"):
                    chat_history += f'<div style="background-color:#DCF8C6; color:black; padding:10px; border-radius:10px; max-width: 70%; margin: 10px 0;">{message}</div>'
                else:
                    chat_history += f'<div style="background-color:#ECECEC; color:black; padding:10px; border-radius:10px; max-width: 70%; margin: 10px 0; text-align: right;">{message}</div>'

            # Show the chat history in a scrollable area
            st.markdown(f'<div style="height: 300px; overflow-y: scroll; padding: 10px; border: 1px solid #ddd; border-radius: 10px;">{chat_history}</div>', unsafe_allow_html=True)

            # Input for user query fixed at the bottom
            user_query = st.text_input("Ask me a question:", key="user_query")

            if user_query:
                with st.spinner('Finding the answer...'):
                    # Get relevant context from ChromaDB based on the query
                    collection = st.session_state['collection']
                    top_chunks = get_top_chunks_from_query(user_query, collection)

                    # Prepare the prompt for the model
                    prompt = f"""
                    You are a knowledgeable and friendly AI tutor.

                    Your task is to answer the question strictly based on the context provided below. Do not use any prior knowledge or make assumptions.

                    Guidelines:
                    - Only use facts and insights present in the context.
                    - If the context does **not** contain enough information to answer the question, respond with:
                      **"I don't know based on the provided context."**
                    - Do not fabricate or hallucinate any information.
                    - Be clear and concise in your explanation.

                    ---

                    📚 **Context**:
                    {top_chunks}

                    ❓ **Question**:
                    {user_query.strip()}

                    ✍️ **Answer**:
                    """

                    # Get the AI response
                    ai_response = get_chatbot_response(prompt)

                    # Store the conversation in session state
                    st.session_state.conversation_history.append(f"User: {user_query}")
                    st.session_state.conversation_history.append(f"AI: {ai_response}")

                    # Display the conversation history with updated chat bubbles
                    st.markdown(f'<div style="background-color:#DCF8C6; color:black; padding:10px; border-radius:10px; max-width: 70%; margin: 10px 0;">User: {user_query}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="background-color:#ECECEC; color:black; padding:10px; border-radius:10px; max-width: 70%; margin: 10px 0; text-align: right;">AI: {ai_response}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

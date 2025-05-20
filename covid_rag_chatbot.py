import os
import streamlit as st
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# === Set Google API Key ===
os.environ["GOOGLE_API_KEY"] = "AIzaSyAF-Pq4AZdOicHKsoJfuP9ClFmUGLnCVE4"  # Replace securely

# === Load Data ===
@st.cache_resource
def load_data():
    loader = JSONLoader(
        file_path="data/rag_dataset.json",
        jq_schema=".examples[] | {page_content: .reference_answer, metadata: {query: .query}}",
        text_content=False
    )
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

@st.cache_resource
def create_chain():
    vectorstore = load_data()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        verbose=False
    )
    return qa_chain, memory

# === Apply Custom Styling ===
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');

html, body, [class*="css"]  {
    background-color: #050a1f;
    color: #00f9ff;
    font-family: 'Orbitron', sans-serif;
    scroll-behavior: smooth;
}

.block-container {
    background: linear-gradient(135deg, rgba(0,255,255,0.1), rgba(255,0,255,0.05));
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 0 25px rgba(0,255,255,0.4);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(0,255,255,0.2);
}

h1 {
    text-align: center;
    color: #00ffe7;
    text-shadow: 0 0 10px #00f0ff, 0 0 20px #00eaff;
}

.stChatInputContainer {
    background: rgba(0, 255, 255, 0.05);
    border: 1px solid #00ffff;
    border-radius: 10px;
    padding: 12px;
    box-shadow: 0 0 12px #00ffe7;
}

.stButton > button {
    background: linear-gradient(135deg, #00ffff, #0088ff);
    color: #0b0f1a;
    border-radius: 12px;
    font-weight: bold;
    box-shadow: 0 0 8px #00ffff;
    transition: 0.3s ease-in-out;
}
.stButton > button:hover {
    box-shadow: 0 0 18px #00ffff;
    transform: scale(1.05);
}

.stChatMessage {
    background-color: rgba(0, 255, 255, 0.05);
    border-left: 3px solid #00ffff;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 12px;
    box-shadow: 0 0 12px rgba(0,255,255,0.2);
    transition: all 0.3s ease;
    font-size: 1.05rem;
}

.stChatMessage.user {
    color: #ff00ff;
    border-left: 3px solid #ff00ff;
    box-shadow: 0 0 15px rgba(255,0,255,0.4);
}

.stChatMessage.assistant {
    color: #00ffff;
    border-left: 3px solid #00ffff;
    box-shadow: 0 0 15px rgba(0,255,255,0.4);
}
</style>
""", unsafe_allow_html=True)

# === UI ===
st.title("🤖         RAG Chatbot")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain, st.session_state.memory = create_chain()
    st.session_state.chat_history = []

user_input = st.chat_input("Type your futuristic query...")

if user_input:
    with st.spinner("Processing with Gemini AI..."):
        response = st.session_state.qa_chain.run(user_input)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", response))

# === Chat Display ===
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

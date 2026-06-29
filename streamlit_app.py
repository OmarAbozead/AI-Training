import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Set page config for mobile responsiveness
st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🤖", layout="centered")

st.title("🤖 AI Knowledge Assistant")
st.write("Ask me anything about the uploaded document.")

# 1. Securely fetch the API key (Streamlit handles secrets via environment variables)
if "GOOGLE_API_KEY" in os.environ:
    api_key = os.environ["GOOGLE_API_KEY"]
elif "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please set your GOOGLE_API_KEY in the environment or Streamlit secrets.")
    st.stop()

# 2. Cache the RAG setup so it doesn't reload on every single user click
@st.cache_resource
def initialize_rag():
    # Load your small dataset (single document chunk)
    loader = TextLoader("training.txt") 
    docs = loader.load()
    
    # Store in a temporary local vector database
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    system_prompt = (
        "You are a helpful assistant. Use the following piece of retrieved context "
        "to answer the user's question. If you don't know the answer, say you don't know.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# Initialize the RAG chain
try:
    rag_chain = initialize_rag()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# 3. Handle Chat History (so users see their previous questions and answers)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 4. Mobile Chat Input
if user_query := st.chat_input("Ask a question..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
        
    # Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

import streamlit as st
import sys
from main import run_agent, PDFProcessor, Config
import os

# Page configuration
st.set_page_config(
    page_title="LangGraph Agent - Weather & PDF RAG",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# Header
st.markdown('<div class="main-header">🤖 LangGraph Agentic Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Weather API + PDF RAG System</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check API keys
    st.subheader("API Keys Status")
    openai_key = os.getenv("OPENAI_API_KEY")
    weather_key = os.getenv("OPENWEATHER_API_KEY")
    langchain_key = os.getenv("LANGCHAIN_API_KEY")
    
    if openai_key:
        st.success("✅ OpenAI API Key Configured")
    else:
        st.error("❌ OpenAI API Key Missing")
    
    if weather_key:
        st.success("✅ OpenWeather API Key Configured")
    else:
        st.error("❌ OpenWeather API Key Missing")
    
    if langchain_key:
        st.success("✅ LangChain API Key Configured")
    else:
        st.warning("⚠️ LangChain API Key Missing (Optional for tracing)")
    
    st.divider()
    
    # PDF Upload
    st.subheader("📄 PDF Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if uploaded_file is not None:
        # Save the uploaded file
        os.makedirs("./documents", exist_ok=True)
        pdf_path = "./documents/sample.pdf"
        
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Process PDF"):
            with st.spinner("Processing PDF and creating embeddings..."):
                try:
                    processor = PDFProcessor(pdf_path)
                    processor.load_and_process_pdf()
                    st.session_state.pdf_uploaded = True
                    st.success("✅ PDF processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
    
    if st.session_state.pdf_uploaded:
        st.success("✅ PDF Ready for Queries")
    
    st.divider()
    
    # Example queries
    st.subheader("💡 Example Queries")
    st.markdown("""
    **Weather Queries:**
    - What's the weather in Paris?
    - Tell me the temperature in Tokyo
    - Is it raining in London?
    
    **PDF Queries:**
    - What is mentioned in the document?
    - Summarize the main points
    - What does the document say about [topic]?
    """)
    
    st.divider()
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
st.subheader("💬 Chat Interface")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show routing info if available
        if "route" in message:
            st.caption(f"🔀 Route: {message['route']}")

# Chat input
if prompt := st.chat_input("Ask about weather or query the PDF..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = run_agent(prompt)
                response = result["final_response"]
                route = result["route"]
                
                st.markdown(response)
                st.caption(f"🔀 Route: {route}")
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "route": route
                })
                
                # Show debug info in expander
                with st.expander("🔍 Debug Information"):
                    st.json({
                        "route": route,
                        "messages": result["messages"],
                        "weather_data": result.get("weather_data", {}),
                        "pdf_context_length": len(result.get("pdf_context", ""))
                    })
            
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Built with LangGraph, LangChain, Qdrant, and Streamlit
</div>
""", unsafe_allow_html=True)

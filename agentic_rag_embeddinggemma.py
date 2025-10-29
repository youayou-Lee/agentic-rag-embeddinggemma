import streamlit as st
import os
from agno.agent import Agent
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.knowledge.embedder.huggingface import HuggingfaceCustomEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.ollama import Ollama
from agno.models.openai import OpenAIChat
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.vectordb.pgvector import PgVector
from agno.vectordb.milvus import Milvus

# Page configuration
st.set_page_config(
    page_title="Agentic RAG with Multiple Models",
    page_icon="🔥",
    layout="wide"
)

def get_embedder(embedder_type, model_name=None):
    """根据选择的类型创建embedder"""
    if embedder_type == "Ollama":
        return OllamaEmbedder(id=model_name or "embeddinggemma:latest", dimensions=768)
    elif embedder_type == "OpenAI":
        return OpenAIEmbedder(id=model_name or "text-embedding-3-small", dimensions=1536)
    elif embedder_type == "SentenceTransformers":
        return SentenceTransformerEmbedder(id=model_name or "all-MiniLM-L6-v2")
    elif embedder_type == "HuggingFace":
        return HuggingfaceCustomEmbedder(id=model_name or "sentence-transformers/all-MiniLM-L6-v2")
    else:
        return OllamaEmbedder(id="embeddinggemma:latest", dimensions=768)

def get_vector_db(db_type, embedder, table_name="recipes"):
    """根据选择的类型创建向量数据库"""
    if db_type == "LanceDB":
        return LanceDb(
            table_name=table_name,
            uri="tmp/lancedb",
            search_type=SearchType.vector,
            embedder=embedder,
        )
    elif db_type == "PgVector":
        # 需要设置PostgreSQL连接
        db_url = os.getenv("POSTGRES_URL", "postgresql+psycopg://ai:ai@localhost:5432/ai")
        return PgVector(
            table_name=table_name,
            db_url=db_url,
            search_type=SearchType.vector,
            embedder=embedder,
        )
    elif db_type == "Milvus":
        # 需要设置Milvus连接
        milvus_url = os.getenv("MILVUS_URL", "http://localhost:19530")
        return Milvus(
            collection=table_name,
            url=milvus_url,
            search_type=SearchType.vector,
            embedder=embedder,
        )
    else:
        return LanceDb(
            table_name=table_name,
            uri="tmp/lancedb",
            search_type=SearchType.vector,
            embedder=embedder,
        )

def get_llm_model(model_type, model_name=None):
    """根据选择的类型创建LLM模型"""
    if model_type == "Ollama":
        return Ollama(id=model_name or "llama3.2:latest")
    elif model_type == "OpenAI":
        return OpenAIChat(id=model_name or "gpt-4o-mini")
    else:
        return Ollama(id="llama3.2:latest")

@st.cache_resource
def load_knowledge_base(urls, embedder_type, embedder_model, db_type):
    if not urls:
        return None
    
    # 创建embedder和向量数据库
    embedder = get_embedder(embedder_type, embedder_model)
    vector_db = get_vector_db(db_type, embedder)
    
    knowledge_base = Knowledge(vector_db=vector_db)
    
    # Add URLs to knowledge base
    for url in urls:
        knowledge_base.add_content(url=url)
    
    return knowledge_base

# Initialize session state
if 'urls' not in st.session_state:
    st.session_state.urls = []
if 'embedder_type' not in st.session_state:
    st.session_state.embedder_type = "Ollama"
if 'embedder_model' not in st.session_state:
    st.session_state.embedder_model = "embeddinggemma:latest"
if 'db_type' not in st.session_state:
    st.session_state.db_type = "LanceDB"
if 'llm_type' not in st.session_state:
    st.session_state.llm_type = "Ollama"
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = "llama3.2:latest"

# Sidebar for configuration and knowledge sources
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    # LLM Model Configuration
    st.subheader("🤖 Language Model")
    llm_type = st.selectbox(
        "LLM Type:",
        ["Ollama", "OpenAI"],
        index=0 if st.session_state.llm_type == "Ollama" else 1,
        key="llm_type_select"
    )
    
    if llm_type == "Ollama":
        llm_model = st.text_input(
            "Ollama Model:",
            value=st.session_state.llm_model if st.session_state.llm_type == "Ollama" else "llama3.2:latest",
            key="llm_model_input"
        )
    else:  # OpenAI
        llm_model = st.selectbox(
            "OpenAI Model:",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            key="openai_model_select"
        )
    
    # Embedder Configuration
    st.subheader("🔤 Embedder")
    embedder_type = st.selectbox(
        "Embedder Type:",
        ["Ollama", "OpenAI", "SentenceTransformers", "HuggingFace"],
        index=["Ollama", "OpenAI", "SentenceTransformers", "HuggingFace"].index(st.session_state.embedder_type),
        key="embedder_type_select"
    )
    
    if embedder_type == "Ollama":
        embedder_model = st.text_input(
            "Ollama Embedder Model:",
            value=st.session_state.embedder_model if st.session_state.embedder_type == "Ollama" else "embeddinggemma:latest",
            key="embedder_model_input"
        )
    elif embedder_type == "OpenAI":
        embedder_model = st.selectbox(
            "OpenAI Embedder Model:",
            ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
            index=0,
            key="openai_embedder_select"
        )
    elif embedder_type == "SentenceTransformers":
        embedder_model = st.selectbox(
            "SentenceTransformers Model:",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"],
            index=0,
            key="st_embedder_select"
        )
    else:  # HuggingFace
        embedder_model = st.text_input(
            "HuggingFace Model:",
            value="sentence-transformers/all-MiniLM-L6-v2",
            key="hf_embedder_input"
        )
    
    # Vector Database Configuration
    st.subheader("🗄️ Vector Database")
    db_type = st.selectbox(
        "Database Type:",
        ["LanceDB", "PgVector", "Milvus"],
        index=["LanceDB", "PgVector", "Milvus"].index(st.session_state.db_type) if st.session_state.db_type in ["LanceDB", "PgVector", "Milvus"] else 0,
        key="db_type_select"
    )
    
    if db_type == "PgVector":
        st.info("💡 Make sure PostgreSQL is running and POSTGRES_URL environment variable is set")
    elif db_type == "Milvus":
        st.info("💡 Make sure Milvus is running and MILVUS_URL environment variable is set (default: http://localhost:19530)")
    
    # Update session state
    if (llm_type != st.session_state.llm_type or 
        llm_model != st.session_state.llm_model or
        embedder_type != st.session_state.embedder_type or 
        embedder_model != st.session_state.embedder_model or
        db_type != st.session_state.db_type):
        
        st.session_state.llm_type = llm_type
        st.session_state.llm_model = llm_model
        st.session_state.embedder_type = embedder_type
        st.session_state.embedder_model = embedder_model
        st.session_state.db_type = db_type
        st.rerun()
    
    st.divider()
    
    st.header("📚 Knowledge Sources")
    
    # Add URL input
    new_url = st.text_input("Add PDF URL:", placeholder="https://example.com/document.pdf")
    
    if st.button("Add URL"):
        if new_url and new_url not in st.session_state.urls:
            st.session_state.urls.append(new_url)
            st.success(f"Added: {new_url}")
            st.rerun()
        elif new_url in st.session_state.urls:
            st.warning("URL already exists!")
        else:
            st.error("Please enter a valid URL")
    
    # Display current URLs
    if st.session_state.urls:
        st.subheader("Current URLs:")
        for i, url in enumerate(st.session_state.urls):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"{i+1}. {url[:50]}...")
            with col2:
                if st.button("🗑️", key=f"delete_{i}"):
                    st.session_state.urls.pop(i)
                    st.rerun()

# Create knowledge base and agent with current configuration
kb = load_knowledge_base(
    st.session_state.urls, 
    st.session_state.embedder_type, 
    st.session_state.embedder_model, 
    st.session_state.db_type
)

llm_model = get_llm_model(st.session_state.llm_type, st.session_state.llm_model)

agent = Agent(
    model=llm_model,
    knowledge=kb,
    instructions=[
        "Search the knowledge base for relevant information and base your answers on it.",
        "Be clear, and generate well-structured answers.",
        "Use clear headings, bullet points, or numbered lists where appropriate.",
    ],
    search_knowledge=True,
    markdown=True,
)

# Main title and description
st.title("🔥 Agentic RAG with Multiple Models")
st.markdown("---")

# Display current configuration
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"🤖 **LLM**: {st.session_state.llm_type} - {st.session_state.llm_model}")
with col2:
    st.info(f"🔤 **Embedder**: {st.session_state.embedder_type} - {st.session_state.embedder_model}")
with col3:
    st.info(f"🗄️ **Vector DB**: {st.session_state.db_type}")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about the knowledge base..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = agent.run(prompt)
                st.markdown(response.content)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response.content})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

with st.expander("📖 How This Works"):
    st.markdown(
        """
**This app uses the Agno framework to create an intelligent Q&A system:**

1. **Knowledge Loading**: PDF URLs are processed and stored in LanceDB vector database
2. **EmbeddingGemma as Embedder**: EmbeddingGemma generates local embeddings for semantic search
3. **Llama 3.2**: The Llama 3.2 model generates answers based on retrieved context

**Key Components:**
- `EmbeddingGemma` as the embedder
- `LanceDB` as the vector database
- `PDFUrlKnowledgeBase`: Manages document loading from PDF URLs
- `OllamaEmbedder`: Uses EmbeddingGemma for embeddings
- `Agno Agent`: Orchestrates everything to answer questions
        """
    )

import streamlit as st
import os
from agno.agent import Agent
from agno.knowledge.embedder.sentence_transformer import SentenceTransformerEmbedder
from agno.knowledge.embedder.huggingface import HuggingfaceCustomEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.openai import OpenAIChat
from agno.vectordb.milvus import Milvus, SearchType
from dotenv import load_dotenv
load_dotenv()  # 这会尝试加载 .env 文件

# Page configuration
st.set_page_config(
    page_title="Agentic RAG with Multiple Models",
    page_icon="🔥",
    layout="wide"
)

def get_embedder(embedder_type, model_name=None):
    """根据选择的类型创建embedder"""
    if embedder_type == "SentenceTransformers":
        return SentenceTransformerEmbedder(id=model_name or "all-MiniLM-L6-v2")
    elif embedder_type == "HuggingFace":
        return HuggingfaceCustomEmbedder(id=model_name or "sentence-transformers/all-MiniLM-L6-v2")
    else:
        return SentenceTransformerEmbedder(id="all-MiniLM-L6-v2")

def get_vector_db(db_type, embedder, table_name="recipes"):
    """根据选择的类型创建向量数据库"""
    # 只支持Milvus向量数据库
    milvus_url = os.getenv("MILVUS_URL")
    return Milvus(
        collection=table_name,
        uri=milvus_url,
        search_type=SearchType.vector,
        embedder=embedder,
    )

def get_llm_model(model_type, model_name=None):
    """根据选择的类型创建LLM模型"""
    # 只支持OpenAI模型
    return OpenAIChat(id=model_name or "deepseek-chat", base_url="https://api.deepseek.com")

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
    st.session_state.embedder_type = "SentenceTransformers"
if 'embedder_model' not in st.session_state:
    st.session_state.embedder_model = "all-MiniLM-L6-v2"
if 'db_type' not in st.session_state:
    st.session_state.db_type = "Milvus"
if 'llm_type' not in st.session_state:
    st.session_state.llm_type = "OpenAI"
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = "deepseek-chat"

# Sidebar for configuration and knowledge sources
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    # LLM Model Configuration
    st.subheader("🤖 Language Model")
    llm_type = "OpenAI"  # 只支持OpenAI
    st.info("🤖 **LLM Type**: OpenAI (固定)")
    
    llm_model = st.selectbox(
        "OpenAI Model:",
        ["deepseek-chat"],
        index=0,
        key="openai_model_select"
    )
    
    # Embedder Configuration
    st.subheader("🔤 Embedder")
    embedder_type = st.selectbox(
        "Embedder Type:",
        ["SentenceTransformers", "HuggingFace"],
        index=0 if st.session_state.embedder_type == "SentenceTransformers" else 1,
        key="embedder_type_select"
    )
    
    if embedder_type == "SentenceTransformers":
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
    db_type = "Milvus"  # 只支持Milvus
    st.info("🗄️ **Database Type**: Milvus (固定)")
    st.info("💡 Make sure Milvus is running and MILVUS_URL environment variable is set (default: http://192.168.102.100:19530)")
    
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
本应用采用 Agno 框架构建智能问答系统，主要流程如下：

1. 知识库加载：系统可处理 PDF 文档链接，并将其内容存储至 LanceDB 向量数据库
2. EmbeddingGemma 嵌入模型：使用 EmbeddingGemma 生成本地语义嵌入向量，支持语义检索
3. Llama 3.2 生成答案：基于检索到的上下文信息，由 Llama 3.2 模型生成最终答案

核心组件说明：
• EmbeddingGemma 作为嵌入模型

• LanceDB 作为向量数据库

• PDFUrlKnowledgeBase：管理从 PDF 链接加载文档的过程

• OllamaEmbedder：调用 EmbeddingGemma 生成嵌入向量

• Agno Agent：协调各组件运作，实现问答功能

        """
    )

# 🔥 Agentic RAG with EmbeddingGemma (100% Local)

一个基于本地模型的智能问答系统，使用 Google 的 EmbeddingGemma 模型和 Ollama 框架构建的完全本地化的 RAG（检索增强生成）应用。

## 📋 项目概述

本项目展示了如何使用 Agno 框架构建一个智能的文档问答系统，该系统能够：

- 📄 从 PDF URL 加载文档内容
- 🔍 使用 EmbeddingGemma 生成语义向量
- 💾 在本地 LanceDB 向量数据库中存储文档
- 🤖 基于检索到的相关内容生成智能回答
- 🌐 提供友好的 Streamlit Web 界面

## ✨ 主要特性

- **100% 本地运行**：所有模型和数据库都在本地运行，保护数据隐私
- **智能文档理解**：支持从 PDF URL 自动提取和理解文档内容
- **语义搜索**：使用 EmbeddingGemma 进行高质量的语义向量搜索
- **流式回答**：实时显示 AI 生成的回答过程
- **易于使用**：简洁的 Web 界面，支持动态添加文档源

## 🛠️ 技术栈

- **前端界面**：Streamlit
- **AI 框架**：Agno
- **语言模型**：Llama 3.2 (通过 Ollama)
- **嵌入模型**：EmbeddingGemma (通过 Ollama)
- **向量数据库**：LanceDB
- **包管理**：uv

## 📦 安装要求

### 系统要求
- Python 3.8+
- Windows/macOS/Linux
- 至少 8GB RAM（推荐 16GB+）

### 依赖安装

1. **安装 Ollama**
   ```bash
   # 访问 https://ollama.com/ 下载并安装 Ollama
   ```

2. **拉取所需模型**
   ```bash
   # 安装 Llama 3.2 模型
   ollama pull llama3.2:latest
   
   # 安装 EmbeddingGemma 模型
   ollama pull embeddinggemma:latest
   ```

3. **安装项目依赖**
   ```bash
   # 使用 uv 安装依赖
   uv sync
   ```

## 🚀 快速开始

1. **克隆项目**
   ```bash
   git clone https://github.com/Shubhamsaboo/awesome-llm-apps.git
   cd awesome-llm-apps/rag_tutorials/agentic_rag_embedding_gemma
   ```

2. **启动应用**
   ```bash
   uv run streamlit run agentic_rag_embeddinggemma.py
   ```

3. **访问应用**
   - 打开浏览器访问：http://localhost:8502

4. **添加文档源**
   - 在侧边栏的"Add URL"输入框中输入 PDF 文档的 URL
   - 点击"➕ Add URL"按钮添加文档到知识库

5. **开始问答**
   - 在主界面的输入框中输入您的问题
   - 点击"🚀 Get Answer"获取基于文档内容的智能回答

## 📖 使用说明

### 添加文档源
1. 在左侧边栏找到"🌐 Add Knowledge Sources"部分
2. 在"Add URL"输入框中输入 PDF 文档的完整 URL
3. 点击"➕ Add URL"按钮
4. 系统会自动下载并处理文档，将其添加到向量数据库中
5. 添加成功后，文档会显示在"📚 Current Knowledge Sources"列表中

### 进行问答
1. 确保已添加至少一个文档源
2. 在主界面的"Enter your question:"输入框中输入您的问题
3. 点击"🚀 Get Answer"按钮
4. 系统会搜索相关文档内容并生成回答
5. 回答会以流式方式实时显示

## 🔧 核心处理逻辑

### 1. 文档加载与处理流程

```
PDF URL → 下载文档 → 文本提取 → 分块处理 → 向量化 → 存储到 LanceDB
```

**详细步骤：**
1. **URL 验证**：验证输入的 PDF URL 格式和可访问性
2. **文档下载**：从 URL 下载 PDF 文件
3. **内容提取**：使用 PDF 阅读器提取文本内容
4. **文本分块**：将长文档分割成适合处理的文本块
5. **向量生成**：使用 EmbeddingGemma 为每个文本块生成 768 维向量
6. **数据存储**：将向量和原始文本存储到 LanceDB 数据库

### 2. 问答处理流程

```
用户问题 → 问题向量化 → 相似度搜索 → 上下文检索 → LLM 生成回答
```

**详细步骤：**
1. **问题预处理**：清理和标准化用户输入的问题
2. **问题向量化**：使用 EmbeddingGemma 将问题转换为向量
3. **相似度搜索**：在 LanceDB 中搜索最相关的文档片段
4. **上下文构建**：将检索到的相关内容组织成上下文
5. **回答生成**：Llama 3.2 基于上下文生成回答
6. **流式输出**：实时显示生成的回答内容

### 3. 核心组件架构

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   Agno Agent     │    │   Ollama Models │
│                 │◄──►│                  │◄──►│                 │
│ - 文档管理      │    │ - 知识检索       │    │ - Llama 3.2     │
│ - 问答界面      │    │ - 回答生成       │    │ - EmbeddingGemma│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Session State  │    │   Knowledge Base │    │    LanceDB      │
│                 │    │                  │    │                 │
│ - URL 列表      │    │ - 文档管理       │    │ - 向量存储      │
│ - 状态管理      │    │ - 内容检索       │    │ - 相似度搜索    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 项目结构

```
agentic_rag_embedding_gemma/
├── agentic_rag_embeddinggemma.py  # 主应用文件
├── pyproject.toml                 # 项目配置和依赖
├── requirements.txt               # Python 依赖列表
├── README.md                      # 项目文档
├── google.png                     # Google 图标
├── ollama.png                     # Ollama 图标
├── agno.png                       # Agno 图标
└── tmp/                          # 临时文件目录
    └── lancedb/                  # LanceDB 数据库文件
```

## ⚙️ 配置说明

### 模型配置
- **语言模型**：`llama3.2:latest`
- **嵌入模型**：`embeddinggemma:latest`
- **向量维度**：768
- **数据库表名**：`recipes`

### 可自定义参数
```python
# 在 agentic_rag_embeddinggemma.py 中可以修改以下参数：

# 向量数据库配置
LanceDb(
    table_name="recipes",           # 数据库表名
    uri="tmp/lancedb",             # 数据库存储路径
    search_type=SearchType.vector,  # 搜索类型
    embedder=OllamaEmbedder(
        id="embeddinggemma:latest", # 嵌入模型
        dimensions=768              # 向量维度
    )
)

# 语言模型配置
Ollama(id="llama3.2:latest")       # 可更换为其他 Ollama 模型
```

## 🔍 故障排除

### 常见问题

1. **模型未找到错误**
   ```bash
   # 确保已安装所需模型
   ollama list
   ollama pull llama3.2:latest
   ollama pull embeddinggemma:latest
   ```

2. **依赖安装问题**
   ```bash
   # 重新安装依赖
   uv sync --reinstall
   ```

3. **端口占用问题**
   ```bash
   # 指定其他端口启动
   uv run streamlit run agentic_rag_embeddinggemma.py --server.port 8503
   ```

4. **PDF 加载失败**
   - 确保 PDF URL 可以公开访问
   - 检查网络连接
   - 验证 PDF 文件格式

### 性能优化建议

1. **硬件要求**
   - 推荐使用 GPU 加速（如果可用）
   - 至少 16GB RAM 以获得更好性能

2. **模型选择**
   - 可以根据需要选择更小或更大的模型
   - 调整向量维度以平衡性能和准确性

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- [Agno](https://github.com/agno-ai/agno) - AI 应用开发框架
- [Ollama](https://ollama.com/) - 本地 LLM 运行平台
- [Streamlit](https://streamlit.io/) - Web 应用框架
- [LanceDB](https://lancedb.com/) - 向量数据库
- Google - EmbeddingGemma 模型

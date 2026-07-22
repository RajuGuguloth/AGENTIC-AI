# Research Assistant Agent 🔬

An **agentic RAG (Retrieval-Augmented Generation) system** that helps you research academic papers by breaking down high-level questions into sub-questions, searching through PDF documents, and synthesizing comprehensive answers.

## Features

**Document Loading**: Load PDFs, text files, or use built-in datasets
**Cranfield Dataset**: Built-in support for the classic IR benchmark (1,400 abstracts)
**arXiv AI Papers**: Download recent AI/ML research papers directly from arXiv (10-100 papers)
**Semantic Search**: Uses Hugging Face embeddings (local `sentence-transformers`) for efficient search
**Agentic Planning**: Automatically breaks down research goals into sub-questions
**LLM Reasoning**: Uses Perplexity API for planning and synthesis
**Simple UI**: Streamlit web interface for easy interaction

## Architecture

```
User Question (High-level goal)
    ↓
Agent Planning (Perplexity) → Sub-questions
    ↓
For each sub-question:
    RAG Retrieval (FAISS + Embeddings)
    ↓
    Answer Generation (Perplexity + Context)
    ↓
Synthesis (Perplexity) → Final Answer
```

## Setup

### Prerequisites

- Python 3.10+
- Perplexity API key (set as environment variable)

### Installation

1. **Navigate to the research_assistant directory:**
   ```bash
   cd research_assistant
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Mac/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables:**
   ```bash
   export PERPLEXITY_API_KEY="your-api-key-here"
   ```
   
   Or create a `.env` file:
   ```
   PERPLEXITY_API_KEY=your-api-key-here
   ```

5. **Add Documents (Optional):**
   - **Option A**: Use Cranfield dataset (recommended for testing) - available in the UI
   - **Option B**: Place your PDF/text files in the `pdfs/` folder (or specify a different path)

## Usage

### Run the Streamlit UI:

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`.

### Steps:

**Option 1: Use Cranfield Dataset (Recommended for testing)**
1. Select "Cranfield Dataset" in the sidebar
2. Click "Download & Load Cranfield" - automatically downloads 1,400 research abstracts
3. Once indexed, ask your research questions!

**Option 2: Use arXiv AI Papers (Recommended for AI research)**
1. Select "arXiv AI Papers" in the sidebar
2. Choose number of papers (10-100, default: 50)
3. Click "Download & Load arXiv Papers" - downloads recent AI/ML papers (may take a few minutes)
4. Once indexed, ask your research questions!

**Option 3: Use Your Own Documents**
1. Select "My Own Documents" in the sidebar
2. Enter the path to your document folder (default: `./pdfs`) in the sidebar
3. Click "Build Index" to process the documents (this may take a minute)
4. Once indexed, ask your research questions!

## Example Questions

- "Summarize the main contributions of these papers"
- "What are the key limitations mentioned?"
- "Compare the different methods proposed"
- "What datasets were used in these studies?"
- "What are the future research directions suggested?"

## Project Structure

```
research_assistant/
├── main.py              # Streamlit UI entry point
├── config.py            # Configuration and env vars
├── perplexity_client.py # Perplexity API client
├── pdf_ingestion.py     # Document loading and chunking (PDFs, text files)
├── cranfield_loader.py  # Cranfield dataset download and processing
├── arxiv_loader.py      # arXiv paper downloader
├── embeddings.py        # Embedding generation
├── vector_store.py      # FAISS vector index
├── rag.py               # RAG retrieval and answering
├── agent.py             # Agent planning and synthesis
├── requirements.txt     # Dependencies
├── README.md           # This file
└── pdfs/               # Place your PDFs/text files here
```

## How It Works

1. **Document Ingestion**: PDFs/text files are loaded and split into overlapping chunks
   - Supports PDFs, text files, and the Cranfield dataset (XML format)
2. **Embedding**: Each chunk is converted to a vector using `sentence-transformers`
3. **Indexing**: Vectors are stored in a FAISS index for fast similarity search
4. **Agent Loop**:
   - **Planning**: High-level goal → 2-4 sub-questions (via Perplexity)
   - **Research**: Each sub-question → RAG retrieval → Answer (via Perplexity + context)
   - **Synthesis**: All sub-answers → Final coherent answer (via Perplexity)

## Built-in Datasets

### Cranfield Dataset

The [Cranfield dataset](https://github.com/oussbenk/cranfield-trec-dataset) is a classic information retrieval benchmark containing:
- **1,400 research abstracts** from aeronautical engineering papers
- Well-structured format ideal for testing RAG systems
- Perfect for experimentation and learning

### arXiv AI Papers

Download recent AI/ML research papers directly from arXiv:
- **10-100 papers** (configurable)
- Covers AI, Machine Learning, NLP, Computer Vision, Neural Networks
- Automatically downloads PDFs and indexes them
- Great for staying up-to-date with latest research

To use either dataset, select it in the UI sidebar and click the download button. The system will automatically download, process, and index the documents.

## Configuration

Edit `config.py` to adjust:
- `CHUNK_SIZE`: Size of text chunks (default: 500 characters)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50 characters)
- `TOP_K`: Number of chunks to retrieve per question (default: 5)
- `PERPLEXITY_MODEL`: Model for reasoning (default: "sonar-reasoning")
- `EMBEDDING_MODEL`: Embedding model (default: "all-MiniLM-L6-v2")

## Troubleshooting

- **"PERPLEXITY_API_KEY not set"**: Make sure you've set the environment variable
- **"No PDF/text files found"**: Check that your document folder path is correct, or try using the Cranfield dataset
- **Slow embedding**: First run downloads the model (~80MB). Subsequent runs are faster.
- **Memory issues**: Reduce `CHUNK_SIZE` or process fewer PDFs at once

## Requirements

- Works on CPU (no GPU needed)
- Optimized for MacBook Pro M2 with 8GB RAM
- Uses lightweight models suitable for local execution

## License

Part of the CredResolve Negotiator project.


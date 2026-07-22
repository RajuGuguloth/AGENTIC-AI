"""
ETL — Text Chunker
Splits LangChain Documents into overlapping chunks.
Implements Parent-Child chunking using Semantic Chunking for children.
"""

from typing import List, Tuple
from uuid import uuid4
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import Config


def _get_embeddings():
    """Lazy load embeddings to avoid loading multiple times."""
    return HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)


def _chunk_image_document(doc: Document) -> Tuple[Document, Document]:
    """Handle a single image document without text splitting."""
    parent_id = str(uuid4())
    image_path = doc.metadata.get("file_path") or doc.metadata.get("image_path", "")

    base_metadata = {
        **doc.metadata,
        "image_path": image_path,
        "parent_id": parent_id,
    }

    parent = Document(
        page_content="",
        metadata={**base_metadata, "doc_type": "parent"},
    )
    child = Document(
        page_content="",
        metadata={**base_metadata, "doc_type": "child"},
    )
    return parent, child


def chunk_documents_parent_child(
    documents: List[Document],
    parent_chunk_size: int = 1500,
    parent_chunk_overlap: int = 200,
    progress=None
) -> Tuple[List[Document], List[Document]]:
    """
    Split documents into large Parent chunks, and smaller Semantic Child chunks.
    Each child will have a 'parent_id' linking it back to its parent.

    Args:
        documents: List of LangChain Documents.
        parent_chunk_size: Max characters for parent chunk.
        parent_chunk_overlap: Overlap for parent chunk.
        progress: Optional Gradio progress.

    Returns:
        (parent_docs, child_docs)
    """
    image_docs = [d for d in documents if d.metadata.get("content_type") == "image"]
    text_docs = [d for d in documents if d.metadata.get("content_type") != "image"]

    parent_docs: List[Document] = []
    child_docs: List[Document] = []

    for doc in image_docs:
        parent, child = _chunk_image_document(doc)
        parent_docs.append(parent)
        child_docs.append(child)

    if not text_docs:
        if progress:
            progress(1.0, desc=f"Created {len(parent_docs)} parents and {len(child_docs)} children.")
        print(
            f"[chunker] Parent-Child generation: {len(documents)} raw docs -> "
            f"{len(parent_docs)} parents -> {len(child_docs)} semantic children."
        )
        return parent_docs, child_docs

    if progress:
        progress(0.3, desc="Generating Parent chunks...")
        
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    text_parent_docs = parent_splitter.split_documents(text_docs)
    
    # Assign UUIDs to parents
    for p in text_parent_docs:
        p.metadata["parent_id"] = str(uuid4())
        p.metadata["doc_type"] = "parent"

    parent_docs.extend(text_parent_docs)

    if progress:
        progress(0.5, desc="Initializing Semantic Chunker (loading embeddings)...")
        
    embeddings = _get_embeddings()
    semantic_chunker = SemanticChunker(
        embeddings, breakpoint_threshold_type="percentile"
    )

    total_parents = len(text_parent_docs)
    
    if progress:
        progress(0.6, desc="Generating Semantic Child chunks...")

    for i, parent in enumerate(text_parent_docs):
        # We split the parent's content semantically
        try:
            # SemanticChunker expects string or list of strings
            splits = semantic_chunker.split_text(parent.page_content)
            for split_text in splits:
                if not split_text.strip():
                    continue
                child = Document(
                    page_content=split_text,
                    metadata={
                        **parent.metadata,
                        "doc_type": "child",
                    }
                )
                child_docs.append(child)
        except Exception as e:
            print(f"[chunker] Warning: Semantic split failed on parent {i}: {e}")
            # Fallback to just using the parent as child if semantic chunker fails
            child = Document(page_content=parent.page_content, metadata={**parent.metadata, "doc_type": "child"})
            child_docs.append(child)
            
        if progress and i % 10 == 0:
            progress(0.6 + (0.4 * (i / total_parents)), desc=f"Semantic chunking ({i}/{total_parents} parents)...")

    if progress:
        progress(1.0, desc=f"Created {len(parent_docs)} parents and {len(child_docs)} children.")
        
    print(f"[chunker] Parent-Child generation: {len(documents)} raw docs -> {len(parent_docs)} parents -> {len(child_docs)} semantic children.")
    return parent_docs, child_docs

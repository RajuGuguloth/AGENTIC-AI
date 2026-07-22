"""
ETL — PDF Loader
Extracts text from PDF files page by page using pypdf.
Returns LangChain Document objects with source metadata.
"""

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf(file_path: str) -> List[Document]:
    """
    Extract text from a PDF file and return as a list of LangChain Documents,
    one Document per page.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of Document objects with page_content and metadata.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    reader = PdfReader(str(path))
    documents: List[Document] = []

    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception as e:
            print(f"[pdf_loader] Warning: Could not extract page {page_num + 1} from {path.name}: {e}")
            continue

        if not text or not text.strip():
            continue

        documents.append(Document(
            page_content=text.strip(),
            metadata={
                "source": path.name,
                "file_path": str(path),
                "page": page_num + 1,
                "total_pages": len(reader.pages),
            }
        ))

    if not documents:
        raise ValueError(f"No extractable text found in {path.name}. It may be a scanned PDF.")

    print(f"[pdf_loader] Loaded {len(documents)} pages from '{path.name}'")
    return documents


def load_pdfs_from_files(file_paths: List[str], progress=None) -> List[Document]:
    """
    Load PDFs from a list of file paths (e.g., from Gradio upload).

    Args:
        file_paths: List of absolute paths to PDF files.
        progress: Optional Gradio progress object.

    Returns:
        Combined list of Documents from all PDFs.
    """
    if not file_paths:
        raise ValueError("No PDF files provided.")

    all_docs: List[Document] = []
    total_files = len(file_paths)
    
    for i, file_path in enumerate(file_paths):
        path = Path(file_path)
        if progress:
            progress(i / total_files, desc=f"Loading {path.name}...")
            
        try:
            docs = load_pdf(str(path))
            all_docs.extend(docs)
        except Exception as e:
            print(f"[pdf_loader] Skipping {path.name}: {e}")

    if progress:
        progress(1.0, desc=f"Loaded {len(all_docs)} total pages.")
        
    print(f"[pdf_loader] Total: {len(all_docs)} pages from {total_files} PDFs")
    return all_docs

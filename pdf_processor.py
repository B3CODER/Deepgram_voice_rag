"""
PDF Processing Module for Voice RAG Pipeline
============================================
Handles PDF upload, text extraction, chunking, and vector embedding preparation.

Features:
- Extract text from PDF files (supports multi-page)
- Intelligent text chunking with overlap
- Metadata tracking (filename, page numbers, timestamps)
- Integration with sentence-transformers embedding models

Requirements:
- PyPDF2>=3.0.0
- pdfplumber>=0.10.0
"""

import os
import time
from typing import List, Dict, Tuple
from datetime import datetime

try:
    import PyPDF2
except ImportError:
    print("Please install PyPDF2: pip install PyPDF2")
    exit(1)

try:
    import pdfplumber
except ImportError:
    print("Please install pdfplumber: pip install pdfplumber")
    exit(1)


class PDFProcessor:
    """Handles PDF text extraction, chunking, and embedding preparation."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        """
        Initialize PDF processor.
        
        Args:
            chunk_size: Target size for text chunks (characters)
            chunk_overlap: Overlap between consecutive chunks (characters)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def extract_text_from_pdf(self, pdf_path: str, method: str = "pdfplumber") -> List[Dict]:
        """
        Extract text from PDF file with page-level metadata.
        
        Args:
            pdf_path: Path to PDF file
            method: Extraction method - "pdfplumber" (default) or "pypdf2"
            
        Returns:
            List of dicts with 'page_number' and 'text' keys
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        pages_data = []
        
        if method == "pdfplumber":
            pages_data = self._extract_with_pdfplumber(pdf_path)
        else:
            pages_data = self._extract_with_pypdf2(pdf_path)
        
        # Filter out empty pages
        pages_data = [p for p in pages_data if p['text'].strip()]
        
        print(f"✓ Extracted text from {len(pages_data)} pages")
        return pages_data
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[Dict]:
        """Extract text using pdfplumber (better formatting preservation)."""
        pages_data = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages_data.append({
                    'page_number': i + 1,
                    'text': text
                })
        
        return pages_data
    
    def _extract_with_pypdf2(self, pdf_path: str) -> List[Dict]:
        """Extract text using PyPDF2 (fallback method)."""
        pages_data = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text() or ""
                pages_data.append({
                    'page_number': i + 1,
                    'text': text
                })
        
        return pages_data
    
    def chunk_text(self, text: str, page_number: int = None) -> List[Dict]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            page_number: Optional page number for metadata
            
        Returns:
            List of dicts with 'text' and 'page_number' keys
        """
        chunks = []
        
        # Clean text
        text = text.strip()
        if not text:
            return chunks
        
        # Split into chunks with overlap
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If not the last chunk, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                search_start = max(start, end - 100)
                search_end = min(len(text), end + 100)
                search_text = text[search_start:search_end]
                
                # Find last sentence ending
                for delimiter in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    last_pos = search_text.rfind(delimiter)
                    if last_pos != -1:
                        end = search_start + last_pos + len(delimiter)
                        break
            
            # Extract chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'page_number': page_number,
                    'chunk_id': chunk_id
                })
                chunk_id += 1
            
            # Move start position (with overlap)
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start <= end - self.chunk_size:
                start = end
        
        return chunks
    
    def process_pdf_to_chunks(self, pdf_path: str) -> Tuple[List[Dict], Dict]:
        """
        Complete pipeline: PDF -> Text -> Chunks with metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (chunks_list, metadata_dict)
            - chunks_list: List of chunk dicts with text and page numbers
            - metadata_dict: Overall PDF metadata
        """
        print(f"\n📄 Processing PDF: {os.path.basename(pdf_path)}")
        
        # Extract text from PDF
        pages_data = self.extract_text_from_pdf(pdf_path)
        
        if not pages_data:
            raise ValueError("No text could be extracted from PDF")
        
        # Chunk each page
        all_chunks = []
        for page_data in pages_data:
            page_chunks = self.chunk_text(
                page_data['text'], 
                page_number=page_data['page_number']
            )
            all_chunks.extend(page_chunks)
        
        # Create metadata
        metadata = {
            'filename': os.path.basename(pdf_path),
            'filepath': os.path.abspath(pdf_path),
            'total_pages': len(pages_data),
            'total_chunks': len(all_chunks),
            'upload_timestamp': datetime.now().isoformat(),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
        
        print(f"  ✓ Created {len(all_chunks)} chunks from {len(pages_data)} pages")
        print(f"  ✓ Chunk size: {self.chunk_size} chars, Overlap: {self.chunk_overlap} chars")
        
        return all_chunks, metadata
    
    def prepare_for_milvus(self, chunks: List[Dict], metadata: Dict, 
                          embedding_model) -> List[Dict]:
        """
        Prepare chunks for Milvus insertion with embeddings.
        
        Args:
            chunks: List of chunk dicts from process_pdf_to_chunks
            metadata: PDF metadata dict
            embedding_model: SentenceTransformer model for embeddings
            
        Returns:
            List of dicts ready for Milvus insertion with fields:
            - text: chunk text
            - vector: embedding vector
            - source_type: "pdf"
            - filename: PDF filename
            - page_number: page number
            - upload_timestamp: ISO timestamp
        """
        print(f"\n🔢 Generating embeddings for {len(chunks)} chunks...")
        
        milvus_data = []
        
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = embedding_model.encode(chunk['text']).tolist()
            
            # Prepare Milvus entry
            entry = {
                'text': chunk['text'],
                'vector': embedding,
                'source_type': 'pdf',
                'filename': metadata['filename'],
                'page_number': chunk.get('page_number', 0),
                'upload_timestamp': metadata['upload_timestamp']
            }
            
            milvus_data.append(entry)
            
            # Progress indicator
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                print(f"  Progress: {i + 1}/{len(chunks)} chunks embedded", end='\r')
        
        print(f"\n  ✓ Generated {len(milvus_data)} embeddings")
        return milvus_data


def select_pdf_file() -> str:
    """
    Interactive PDF file selection.
    
    Returns:
        Path to selected PDF file
    """
    print("\n" + "="*50)
    print("📁 PDF FILE SELECTION")
    print("="*50)
    
    while True:
        pdf_path = input("\nEnter PDF file path (or 'cancel' to go back): ").strip()
        
        if pdf_path.lower() == 'cancel':
            return None
        
        # Remove quotes if present
        pdf_path = pdf_path.strip('"').strip("'")
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            print(f"❌ File not found: {pdf_path}")
            print("Please enter a valid file path.")
            continue
        
        # Check if it's a PDF
        if not pdf_path.lower().endswith('.pdf'):
            print(f"❌ Not a PDF file: {pdf_path}")
            print("Please select a .pdf file.")
            continue
        
        # File is valid
        print(f"✓ Selected: {os.path.basename(pdf_path)}")
        return pdf_path


if __name__ == "__main__":
    # Test the PDF processor
    print("PDF Processor Test")
    print("=" * 50)
    
    # Select PDF
    pdf_path = select_pdf_file()
    
    if pdf_path:
        # Process PDF
        processor = PDFProcessor(chunk_size=800, chunk_overlap=150)
        chunks, metadata = processor.process_pdf_to_chunks(pdf_path)
        
        # Display results
        print("\n" + "="*50)
        print("PROCESSING RESULTS")
        print("="*50)
        print(f"Filename: {metadata['filename']}")
        print(f"Total Pages: {metadata['total_pages']}")
        print(f"Total Chunks: {metadata['total_chunks']}")
        print(f"Timestamp: {metadata['upload_timestamp']}")
        
        # Show first chunk
        if chunks:
            print("\n" + "-"*50)
            print("SAMPLE CHUNK (first one):")
            print("-"*50)
            print(f"Page: {chunks[0].get('page_number', 'N/A')}")
            print(f"Text: {chunks[0]['text'][:200]}...")

"""
Document extraction module for PDFs, Markdown, and TXT files.
"""

import os
from typing import List, Optional, Tuple
import logging
from utils import Document, log_step, log_error

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

logger = logging.getLogger(__name__)


class DocumentExtractor:
    """Extract text and metadata from various document formats."""
    
    SUPPORTED_FORMATS = {'pdf', 'md', 'markdown', 'txt', 'text'}
    
    def __init__(self):
        log_step("EXTRACT", "Initializing DocumentExtractor")
    
    def extract(self, filepath: str, source_path: str) -> Optional[Document]:
        """
        Extract text from a document.
        
        Args:
            filepath: Full path to document
            source_path: Logical path for metadata (e.g., 'manuals/deviceA.pdf')
        
        Returns:
            Document object or None if extraction fails
        """
        if not os.path.exists(filepath):
            log_error("EXTRACT", f"File not found: {filepath}")
            return None
        
        _, ext = os.path.splitext(filepath)
        doc_type = ext.lstrip('.').lower()
        
        try:
            if doc_type == 'pdf':
                content = self._extract_pdf(filepath)
            elif doc_type in {'md', 'markdown'}:
                content = self._extract_markdown(filepath)
            elif doc_type in {'txt', 'text'}:
                content = self._extract_text(filepath)
            else:
                log_error("EXTRACT", f"Unsupported format: {doc_type}")
                return None
            
            if not content or not content.strip():
                log_error("EXTRACT", f"No content extracted from {filepath}")
                return None
            
            doc = Document(
                content=content,
                source=source_path,
                doc_type=doc_type.replace('markdown', 'md'),
                metadata={
                    'filename': os.path.basename(filepath),
                    'file_size_bytes': os.path.getsize(filepath),
                    'char_count': len(content)
                }
            )
            
            log_step("EXTRACT", f"Extracted {len(content)} chars from {source_path}")
            return doc
        
        except Exception as e:
            log_error("EXTRACT", f"Error extracting {filepath}: {str(e)}")
            return None
    
    def _extract_pdf(self, filepath: str) -> str:
        """Extract text from PDF file."""
        if PdfReader is None:
            raise ImportError("pypdf not installed. Install with: pip install pypdf")
        
        reader = PdfReader(filepath)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
        
        return "\n\n".join(text_parts)
    
    def _extract_markdown(self, filepath: str) -> str:
        """Extract text from Markdown file."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _extract_text(self, filepath: str) -> str:
        """Extract text from plain text file."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def extract_batch(
        self,
        file_list: List[Tuple[str, str]]
    ) -> List[Document]:
        """
        Extract multiple documents.
        
        Args:
            file_list: List of (filepath, source_path) tuples
        
        Returns:
            List of Document objects
        """
        documents = []
        for filepath, source_path in file_list:
            doc = self.extract(filepath, source_path)
            if doc:
                documents.append(doc)
        
        log_step("EXTRACT", f"Extracted {len(documents)} documents total")
        return documents


class AzureBlobExtractor:
    """Extract documents directly from Azure Blob Storage."""
    
    def __init__(self, connection_string: str):
        """
        Initialize Azure Blob extractor.
        
        Args:
            connection_string: Azure Storage connection string
        """
        try:
            from azure.storage.blob import BlobServiceClient
            self.blob_service_client = BlobServiceClient.from_connection_string(
                connection_string
            )
            self.extractor = DocumentExtractor()
            log_step("EXTRACT", "Initialized AzureBlobExtractor")
        except ImportError:
            raise ImportError("azure-storage-blob not installed")
    
    def extract_from_container(
        self,
        container_name: str,
        local_temp_dir: str = "./temp_downloads"
    ) -> List[Document]:
        """
        Extract all documents from a blob container.
        
        Args:
            container_name: Name of the blob container
            local_temp_dir: Temporary directory for downloads
        
        Returns:
            List of Document objects
        """
        import tempfile
        
        if not os.path.exists(local_temp_dir):
            os.makedirs(local_temp_dir)
        
        documents = []
        container_client = self.blob_service_client.get_container_client(
            container_name
        )
        
        try:
            blobs = container_client.list_blobs()
            
            for blob in blobs:
                blob_name = blob.name
                
                # Check if it's a supported format
                _, ext = os.path.splitext(blob_name)
                if ext.lstrip('.').lower() not in self.extractor.SUPPORTED_FORMATS:
                    continue
                
                # Download blob to temp directory
                local_path = os.path.join(local_temp_dir, blob_name.split('/')[-1])
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                blob_client = container_client.get_blob_client(blob_name)
                with open(local_path, 'wb') as f:
                    f.write(blob_client.download_blob().readall())
                
                # Extract document
                doc = self.extractor.extract(local_path, blob_name)
                if doc:
                    documents.append(doc)
            
            log_step("EXTRACT", f"Extracted {len(documents)} from Azure blob container")
        
        except Exception as e:
            log_error("EXTRACT", f"Error extracting from blob container: {str(e)}")
        
        return documents

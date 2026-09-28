import pytest
from src.rag.document_loader import Document
from src.rag.chunker import TextChunker
from src.rag.vector_store import VectorStore
from src.rag.pipeline import RAGPipeline

def test_chunker():
    doc = Document(content="word " * 400, doc_id="test_doc", metadata={"document_type": "sop"})
    chunker = TextChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1
    assert chunks[0].metadata["document_type"] == "sop"

def test_vector_store():
    vs = VectorStore(persistence_dir="vector_store_test")
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    doc = Document(content="Press-04 operating temperature above 180C requires immediate visual inspection.", doc_id="test_sop")
    chunks = chunker.chunk_document(doc)
    vs.add_chunks(chunks)
    
    results = vs.similarity_search("What is the temperature limit for Press-04?", top_k=1)
    assert len(results) > 0
    assert "180C" in results[0][0].text

def test_rag_pipeline():
    pipeline = RAGPipeline(vector_store_dir="vector_store_test")
    res = pipeline.answer_query("Why was RB-2041 flagged?")
    assert "answer" in res
    assert "sources" in res
    assert len(res["sources"]) > 0

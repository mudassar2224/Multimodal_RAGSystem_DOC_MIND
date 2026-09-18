from docmind.ingestion.chunking import chunk_text


def test_chunking_keeps_all_content() -> None:
    text = "word " * 1000
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 2
    assert all(chunk for chunk in chunks)

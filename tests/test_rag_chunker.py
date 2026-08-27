from stockM.rag.chunker import FixedSizeTextChunker
from stockM.rag.config import RAGSettings
from stockM.rag.schemas import Document, DocumentCategory


def test_chunker():
    # ---------------------------------------------------------
    # 1. Load existing StockM configuration
    # ---------------------------------------------------------
    settings = RAGSettings()

    print("\n==============================")
    print("STOCKM CHUNKER QUALITY TEST")
    print("==============================")

    print(f"\nChunk size     : {settings.chunk_size}")
    print(f"Chunk overlap  : {settings.chunk_overlap}")
    print(f"Minimum size   : {settings.min_chunk_size}")

    # ---------------------------------------------------------
    # 2. Create chunker
    # ---------------------------------------------------------
    chunker = FixedSizeTextChunker(settings)

    # ---------------------------------------------------------
    # 3. Create a long financial document
    # ---------------------------------------------------------
    text = (
        "Reliance Industries announced strong quarterly results. "
        "The company reported increased revenue and improved profitability. "
        "Management discussed investments in artificial intelligence, "
        "digital services, energy infrastructure, and new technology. "
        "Investors are closely monitoring the company's expansion plans. "
        "The company expects continued growth across its digital businesses. "
        "The management also discussed future capital expenditure plans. "
        "Analysts expect the company's technology investments to influence "
        "future earnings and long-term business growth. "
    ) * 15

    document = Document(
        doc_id="chunk-quality-test-001",
        text=text,
        source="test",
        url="https://example.com/test",
        company="RELIANCE",
        category=DocumentCategory.FINANCIAL_NEWS,
    )

    # ---------------------------------------------------------
    # 4. Generate chunks
    # ---------------------------------------------------------
    chunks = chunker.chunk(document)

    print(f"\nOriginal length : {len(text)}")
    print(f"Number of chunks: {len(chunks)}")

    # ---------------------------------------------------------
    # TEST 1 — Chunks exist
    # ---------------------------------------------------------
    assert len(chunks) > 1
    print("\nPASS: Multiple chunks generated.")

    # ---------------------------------------------------------
    # TEST 2 — IDs are deterministic
    # ---------------------------------------------------------
    for index, chunk in enumerate(chunks):
        expected_id = f"{document.doc_id}::{index}"

        assert chunk.chunk_id == expected_id
        assert chunk.doc_id == document.doc_id
        assert chunk.index == index

    print("PASS: Chunk IDs and indexes are correct.")

    # ---------------------------------------------------------
    # TEST 3 — No empty chunks
    # ---------------------------------------------------------
    for chunk in chunks:
        assert chunk.text.strip()

    print("PASS: No empty chunks.")

    # ---------------------------------------------------------
    # TEST 4 — Metadata preservation
    # ---------------------------------------------------------
    for chunk in chunks:
        assert chunk.metadata["doc_id"] == document.doc_id
        assert chunk.metadata["company"] == "RELIANCE"
        assert chunk.metadata["source"] == "test"
        assert chunk.metadata["category"] == "financial_news"

    print("PASS: Metadata preserved.")

    # ---------------------------------------------------------
    # TEST 5 — Chunk size sanity
    # ---------------------------------------------------------
    for chunk in chunks:
        assert len(chunk.text) <= settings.chunk_size

    print("PASS: No chunk exceeds configured chunk size.")

    # ---------------------------------------------------------
    # TEST 6 — Minimum final chunk size
    # ---------------------------------------------------------
    final_chunk = chunks[-1]

    assert (
        len(final_chunk.text) >= settings.min_chunk_size
        or len(chunks) == 1
    )

    print("PASS: Final chunk satisfies minimum size rule.")

    # ---------------------------------------------------------
    # TEST 7 — Check overlap
    # ---------------------------------------------------------
    print("\nChecking overlap...")

    overlap_found = False

    for i in range(len(chunks) - 1):
        current = chunks[i].text
        next_chunk = chunks[i + 1].text

        # Check whether the end of the current chunk
        # appears at the beginning of the next chunk.
        max_check = min(
            settings.chunk_overlap,
            len(current),
            len(next_chunk),
        )

        for size in range(max_check, 10, -1):
            suffix = current[-size:]

            if next_chunk.startswith(suffix):
                overlap_found = True

                print(
                    f"  Chunks {i} → {i + 1}: "
                    f"{size} characters overlap"
                )

                break

    assert overlap_found

    print("PASS: Chunk overlap detected.")

    # ---------------------------------------------------------
    # TEST 8 — Determinism
    # ---------------------------------------------------------
    chunks_again = chunker.chunk(document)

    assert len(chunks_again) == len(chunks)

    for first, second in zip(chunks, chunks_again):
        assert first.chunk_id == second.chunk_id
        assert first.text == second.text

    print("PASS: Chunking is deterministic.")

    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------
    print("\n==============================")
    print("ALL CHUNKER QUALITY TESTS PASSED ✅")
    print("==============================")


if __name__ == "__main__":
    test_chunker()
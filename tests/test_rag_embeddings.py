from stockM.rag.config import RAGSettings
from stockM.rag.embeddings import SentenceTransformerEmbeddingModel


def main():
    print("\n==============================")
    print("STOCKM RAG EMBEDDING TEST")
    print("==============================")

    # ---------------------------------------------------------
    # 1. Load existing StockM configuration
    # ---------------------------------------------------------
    settings = RAGSettings()

    print("\nConfigured model:")
    print(settings.embedding_model)

    # ---------------------------------------------------------
    # 2. Create embedding model
    # ---------------------------------------------------------
    embedding_model = SentenceTransformerEmbeddingModel(
        settings
    )

    # ---------------------------------------------------------
    # 3. Verify model information
    # ---------------------------------------------------------
    print("\nModel information:")
    print(f"Model name : {embedding_model.model_name}")
    print(f"Dimension  : {embedding_model.dimension}")

    # ---------------------------------------------------------
    # 4. Test multiple texts
    # ---------------------------------------------------------
    texts = [
        "Reliance Industries reported strong quarterly results.",
        "The company announced increased revenue and profitability.",
        "Investors are monitoring the company's technology investments.",
    ]

    embeddings = embedding_model.embed(texts)

    # ---------------------------------------------------------
    # 5. Display results
    # ---------------------------------------------------------
    print("\nEmbedding results:")
    print(f"Input texts : {len(texts)}")
    print(f"Vectors     : {len(embeddings)}")
    print(f"Dimension   : {len(embeddings[0])}")

    print("\nFirst vector preview:")
    print(embeddings[0][:10])

    # ---------------------------------------------------------
    # 6. Validation
    # ---------------------------------------------------------

    assert len(embeddings) == len(texts)

    print("\nPASS: One vector generated per input text.")

    for vector in embeddings:
        assert len(vector) == embedding_model.dimension

    print("PASS: All vectors have correct dimension.")

    for vector in embeddings:
        assert all(isinstance(value, float) for value in vector)

    print("PASS: Vector values are Python floats.")

    # ---------------------------------------------------------
    # 7. Test empty input
    # ---------------------------------------------------------

    empty_result = embedding_model.embed([])

    assert empty_result == []

    print("PASS: Empty input handled correctly.")

    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    print("\n==============================")
    print("EMBEDDING TEST PASSED ✅")
    print("==============================")


if __name__ == "__main__":
    main()
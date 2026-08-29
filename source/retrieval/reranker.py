
def rerank(
    query,
    results,
    reranker,
    top_n=7,
    expected_chunks=None,
    question_id=None,
    reranker_score_log=None
):
    documents = results["documents"][0]

    if not documents:
        return results

    pairs = [(query, document) for document in documents]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, range(len(documents))),
        key=lambda x: x[0],
        reverse=True
    )

    print("\n========== RERANK DEBUG ==========")
    print("Query:", query)

    for rank, (score, index) in enumerate(ranked, start=1):

        metadata = results["metadatas"][0][index]
        chunk_index = metadata["chunk_index"]

        # Check whether this chunk is expected
        is_expected = False

        if expected_chunks:
            is_expected = chunk_index in expected_chunks

        print(
            f"Rank {rank} | "
            f"Original index: {index} | "
            f"Score: {float(score):.4f} | "
            f"Chunk: {chunk_index} | "
            f"Expected: {is_expected}"
        )

        # Save score information for Excel
        if reranker_score_log is not None:

            reranker_score_log.append({
                "ID": question_id,
                "Question": query,
                "Rank": rank,
                "Chunk": chunk_index,
                "Reranker_Score": float(score),
                "Expected": is_expected,
                "Distance": results["distances"][0][index],
                "File": metadata.get("file_name", "")
            })

    # Keep current behavior: final Top-K = 7
    selected = ranked[:top_n]

    selected_indices = [
        index
        for score, index in selected
    ]

    reranked_results = results.copy()

    reranked_results["documents"] = [[
        results["documents"][0][i]
        for i in selected_indices
    ]]

    reranked_results["metadatas"] = [[
        results["metadatas"][0][i]
        for i in selected_indices
    ]]

    reranked_results["ids"] = [[
        results["ids"][0][i]
        for i in selected_indices
    ]]

    reranked_results["distances"] = [[
        results["distances"][0][i]
        for i in selected_indices
    ]]

    print("\nSelected chunks:")

    for score, index in selected:

        metadata = results["metadatas"][0][index]

        print(
            f"Chunk {metadata['chunk_index']} | "
            f"Score {float(score):.4f} | "
            f"{metadata.get('file_name', '')}"
        )

    print("==================================\n")

    return reranked_results

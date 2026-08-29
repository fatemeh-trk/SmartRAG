def search(query, embedder, collection_new, top_k):
    
    
    query_embedding = embedder.encode([query])
    results = collection_new.query(
        query_embeddings = query_embedding.tolist(),
        n_results=top_k)

    return results 
    

def process_retrieval_results(results,threshold):
    distances = results["distances"][0]
    docs = results["documents"][0]
    metadata = results["metadatas"][0]
    ids = results["ids"][0]

    valid_distances = []
    valid_docs = []
    valid_metadata = []
    valid_ids = []

    best_distance = min(distances) if distances else None
    
    for index , distance in enumerate(distances):
        if distance <= threshold:
          valid_distances.append(distance)
          valid_docs.append(docs[index])
          valid_metadata.append(metadata[index])
          valid_ids.append(ids[index]) 

    filtered_results = results.copy()

    filtered_results["distances"] = [valid_distances]
    filtered_results["documents"] = [valid_docs]
    filtered_results["metadatas"] = [valid_metadata]
    filtered_results["ids"] = [valid_ids]

    retrieved_count = len(valid_ids)
    print("Best distance:", best_distance)
    print("Retrieved count:", retrieved_count)
    return filtered_results
